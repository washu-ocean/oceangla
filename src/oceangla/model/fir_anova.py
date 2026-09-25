from itertools import product
import logging
from collections import defaultdict, OrderedDict
from pathlib import Path
import json
import math

import nibabel as nib
from nibabel.cifti2.cifti2_axes import ScalarAxis
import numpy as np
import pandas as pd
import progressbar
import matplotlib.pyplot as plt
from joblib import Parallel, Memory, delayed

# import ipdb
from nilearn.image import resample_img
from pathvalidate import sanitize_filename
from scipy import stats
from templateflow import api as tflow

# from .model import GroupLevelModel
from ..config import config
from ..image_utils.slice import get_spatial_slices
from ..image_utils.cifti import cifti_compatible_structures
from .correction import fdr_correct
from .surface_utils import (
    build_adjacency_from_faces,
    extract_hemi_values,
    get_biggest_surface_clusters,
    get_cluster_index_groups,
    get_template_midthicknesses_from_cifti_header,
)
from .volume_utils import (
    get_volume_array_from_cifti_array,
    get_biggest_voxel_cluster_sizes,
    get_voxel_clusters,
)
from .model import ModelDesc

logger = logging.getLogger(__name__)


def run_anova_model(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame
):
    match model_desc["model_type"]:
        case "fir_twoway_rm_anova":
            activation_img, num_frames = __get_anova_activation_img(model_desc, subject_activation_df)
            design_df = __get_twoway_anova_design_df(model_desc, subject_variables_df, num_frames)
            TwoWayAnovaModel(
                activation_img,
                design_df,
                model_desc,
                num_frames,
                alpha=config.alphas
            ).fit()
        case "fir_rm_anova":
            raise NotImplementedError("One-way ANOVA not yet implemented.")

def __get_oneway_anova_activation_img(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame,
):
    pass

def __get_anova_activation_img(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame,
):
    condition = model_desc["function_args"][0]
    space, task, session = (
        model_desc["space"],
        model_desc["task"],
        model_desc["session"],
    )
    paths = (
        subject_activation_df
        .query(
            "condition == @condition and "
            "frame_no == -1 and "
            "task == @task and "
            "space == @space and "
            "session == @session"
        )
        .sort_values(by=["subject"])
    )["path"].to_list()
    logger.debug("Loading activation...")
    # imgs = Parallel(n_jobs=10, verbose=10)(delayed(lambda p : nib.load(p))(p) for p in paths)
    img0 = nib.load(paths[0])
    if isinstance(img0, nib.Cifti2Image) and img0.dataobj.shape[0] == 1:
        raise ValueError("Image should contain multiple frames for an FIR two-way ANOVA.")
    elif isinstance(img0, nib.Nifti1Image) and img0.dataobj.shape[3] == 1:
        raise ValueError("Image should contain multiple frames for an FIR two-way ANOVA.")
    fdatas = [nib.load(p).get_fdata() for p in paths]
    # fdatas = Parallel(n_jobs=10, verbose=10)(delayed(lambda img : img.get_fdata())(img) for img in imgs)
    logger.debug("Done!")
    if isinstance(img0, nib.Cifti2Image):
        logger.info("Concatenating...")
        frame_count = img0.dataobj.shape[0]
        sub_count = len(subject_activation_df["subject"].unique())
        total_frames = frame_count * sub_count
        fdata_stacked = np.concatenate(fdatas, axis=0)
        logger.info("Making stacked image...")
        print(fdata_stacked.shape)
        return (
            nib.Cifti2Image(
                fdata_stacked,
                header=(
                    ScalarAxis(
                        name=["frame"] * total_frames
                    ),
                    img0.header.get_axis(1)
                ),
                nifti_header=img0.nifti_header
            ),
            frame_count
        )
    elif isinstance(img0, nib.Cifti2Image):
        fdata_stacked = np.concatenate(fdatas, axis=3)
        frame_count = img0.dataobj.shape[3]
        return (
            nib.Nifti1Image(
                fdata_stacked,
                affine=img0.affine,
                header=img0.header
            ),
            frame_count
        )
    else:
        raise ValueError(f"Unexpected image type {type(imgs[0])} (this shouldn't happen)")


def __get_twoway_anova_design_df(
    model_desc: ModelDesc,
    subject_variables_df: pd.DataFrame,
    num_frames: int
) -> pd.DataFrame:
    variable = model_desc["function_args"][1]
    design_df = (
        subject_variables_df[["subject", variable]]
        .sort_values(by="subject")
        .drop(columns=["subject"])
        .reset_index(drop=True)
    )
    design_df["intercept"] = 1
    design_df.insert(0, "intercept", design_df.pop("intercept"))
    design_arr = design_df.to_numpy()
    design_arr = np.repeat(design_arr, num_frames, axis=0)
    frame_arr = np.concatenate([np.eye(num_frames)] * (len(design_df)), axis=0)
    design_arr = np.concatenate((design_arr, frame_arr), axis=1)
    frame_column_names = [f"frame_{i}" for i in range(num_frames)]
    design_df = pd.DataFrame(
        design_arr, 
        columns=list(design_df.columns) + frame_column_names
    )
    for frame_no in frame_column_names:
        interaction_name = f"{variable}_{frame_no}_interaction"
        design_df[interaction_name] = design_df[variable] * design_df[frame_no]
    return design_df


class TwoWayAnovaModel:
    def __init__(
        self,
        activation_img: nib.Cifti2Image | nib.Nifti1Image,
        design_df: pd.DataFrame,
        model_desc: ModelDesc,
        num_frames: int,
        alpha: float | list[float] = 0.05,
        **kwargs,
    ):
        self.activation_img = activation_img
        self.fdata = activation_img.get_fdata()
        self.design_df = design_df
        self.num_frames = num_frames
        self.value_names = list(design_df.columns)
        self.model_desc = model_desc
        self.model_outdir = config.outdir_path / sanitize_filename(self.model_desc["model_name"])
        if not self.model_outdir.is_dir():
            self.model_outdir.mkdir(parents=True, exist_ok=True)
        self.alphas = [alpha] if isinstance(alpha, float) else alpha
        self.uncorr_pvals = None
        self.tstats = None
        self.betas = None
        self.ses = None
        self.fdr_corr_pvals = []

        # volume-specific variables
        self.volume_mask = None
        if isinstance(self.activation_img, nib.Nifti1Image):
            self.out_suffix = ".nii.gz"
            voxel_sizes = self.activation_img.header.get_zooms()[:3]
            # first check if any template resolution matches

            # TODO: handle cohorts
            # TODO: write test for making sure cohort-specific spaces
            for k, v in tflow.get_metadata(self.model_desc["space"])["res"].items():
                if np.allclose(voxel_sizes, v["zooms"]):
                    self.volume_mask = nib.load(
                        tflow.get(
                            self.model_desc["space"],
                            resolution=self.activation_img.header.get_zooms()[0],
                            desc="brain",
                            suffix="mask",
                        )
                    )
                    break
            # if not, try to upsample the template with the closest resolution under the target resolution
            if self.volume_mask is None:
                self.volume_mask = resample_img(
                    nib.load(
                        tflow.get(
                            self.model_desc["space"],
                            resolution=np.floor(voxel_sizes[0]),
                            desc="brain",
                            suffix="mask",
                        )
                    ),
                    target_affine=self.affine,
                    target_shape=self.fdata.shape[:3],
                    interpolation="nearest",
                )

        elif isinstance(self.activation_img, nib.Cifti2Image) and hasattr(self.activation_img.header.get_axis(1), 'vertex'):  # if doesn't have 'vertex' attr, then it has a ParcelAxis
            self.out_suffix = ".dscalar.nii"

    def fit(self):
        print(f"Running {self.model_desc}")
        self._fit()
        self._save()

    def _fit(self, permuted_design_matrix=None):
        design_matrix = self.design_df
        design_matrix_no_int = self.design_df[[col for col in self.design_df if "interaction" not in col]]
        design_matrix_arr = design_matrix.to_numpy()
        design_matrix_arr_no_int = design_matrix_no_int.to_numpy()
        if isinstance(self.activation_img, nib.Cifti2Image):
            img_shape = (1, self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.activation_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], 1)
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))
        else:
            raise ValueError(f"Cannot fit GLM for image type: {type(self.activation_img)}")
        Y = self.fdata.reshape(glm_input_shape)
        beta, _, _, _ = np.linalg.lstsq(
            design_matrix_arr,
            Y,
            rcond=None,
        )
        beta_no_int, _, _, _ = np.linalg.lstsq(
            design_matrix_arr_no_int,
            Y,
            rcond=None,
        )
        rss_full = np.sum((Y - design_matrix_arr @ beta)**2, axis=0)
        rss_reduced = np.sum((Y - design_matrix_arr_no_int @ beta_no_int)**2, axis=0)
        df_num = self.num_frames - 1
        df_denom = len(self.design_df) - design_matrix_arr.shape[1]
        fstat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_denom)
        pval = stats.f.sf(fstat, df_num, df_denom)
        self.fstat = fstat.reshape(img_shape)
        self.uncorr_pval = pval.reshape(img_shape)

    def _save(self):
        if isinstance(self.activation_img, nib.Nifti1Image):
            self._save_nifti()
        elif isinstance(self.activation_img, nib.Cifti2Image):
            self._save_cifti()
        self.design_df.to_csv(self.model_outdir / "design_matrix.tsv", sep="\t")

    def _save_cifti(self):
        for datatype, data in (
            ("uncorr_pvals", self.uncorr_pval),
            ("fstat", self.fstat)
        ):
            if data is None:
                continue
            img = nib.cifti2.cifti2.Cifti2Image(
                data,
                (
                    nib.cifti2.cifti2_axes.ScalarAxis(name=[datatype]),
                    self.activation_img.header.get_axis(1),
                ),
            )
            nib.save(
                img,
                p := self.model_outdir
                / f"{sanitize_filename(self.model_desc['model_name'])}_{datatype}.dscalar.nii",
            )
            logger.info(f"Saved {p!s}")
            del img

    def _save_nifti(self):
        for datatype, data in (
            ("uncorr_pvals", self.uncorr_pvals),
            ("fstat", self.fstat)
        ):
            if data is None:
                continue
            img = nib.Nifti1Image(data, self.affine, header=self.activation_img.header)
            nib.save(
                img,
                p := self.model_outdir
                / f"{sanitize_filename(self.model_desc['model_name'])}_{datatype}.nii.gz",
            )
            logger.info(f"Saved {p!s}")
            del img
