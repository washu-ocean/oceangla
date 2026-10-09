from itertools import product
import logging
import math
import re
import random
import subprocess
import shlex

import nibabel as nib
from nibabel.cifti2.cifti2_axes import ScalarAxis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    get_template_midthickness_paths_from_cifti_header
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
    get_img_cached = config.joblib_memory.cache(__get_anova_activation_img)
    match model_desc["model_type"]:
        case "fir_twoway_rm_anova":
            num_frames = __get_num_frames(model_desc, subject_activation_df)
            design_df, var_of_interest = __get_twoway_anova_design_df(model_desc, subject_variables_df, num_frames)
            activation_img = get_img_cached(model_desc, subject_activation_df, subject_variables_df, var_of_interest, num_frames)
            TwoWayAnovaModel(
                activation_img,
                design_df,
                model_desc,
                num_frames,
                var_of_interest,
                alpha=config.alphas
            ).fit()
        case "fir_rm_anova":
            raise NotImplementedError("One-way ANOVA not yet implemented.")


def __get_num_frames(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame
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
    img0 = nib.load(paths[0])
    if isinstance(img0, nib.Cifti2Image):
        num_frames = img0.dataobj.shape[0]
    elif isinstance(img0, nib.Nifti1Image):
        num_frames = img0.dataobj.shape[3]
    else:
        raise TypeError(f"Unexpected image type {img0}")
    del img0, paths
    return num_frames


def __get_anova_activation_img(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame,
    var_of_interest: str,
    num_frames: int
):
    condition = model_desc["function_args"][0]
    space, task, session = (
        model_desc["space"],
        model_desc["task"],
        model_desc["session"],
    )
    null_subjects = subject_variables_df.loc[subject_variables_df[var_of_interest].isna(), "subject"]
    subject_activation_df_ = subject_activation_df[~subject_activation_df["subject"].isin(null_subjects)]
    paths = (
        subject_activation_df_
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
        sub_count = len(subject_activation_df_["subject"].unique())
        total_frames = num_frames * sub_count
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
            )
        )
    elif isinstance(img0, nib.Cifti2Image):
        fdata_stacked = np.concatenate(fdatas, axis=3)
        return (
            nib.Nifti1Image(
                fdata_stacked,
                affine=img0.affine,
                header=img0.header
            )
        )
    else:
        raise TypeError(f"Unexpected image type {type(img0)} (this shouldn't happen)")


def __get_twoway_anova_design_df(
    model_desc: ModelDesc,
    subject_variables_df: pd.DataFrame,
    num_frames: int
) -> pd.DataFrame:
    variable = model_desc["function_args"][1]
    if len(variable) > 3 and variable[:2] == "C(" and variable[-1] == ")":
        categorical = True
        variable = variable[2:-1]
    else:
        categorical = isinstance(subject_variables_df[variable].dtype, pd.StringDtype)
    design_df = (
        subject_variables_df[["subject", variable]]
        .sort_values(by="subject")
        .reset_index(drop=True)
    )
    design_df = design_df[~design_df[variable].isna()]
    design_df["intercept"] = 1
    design_df.insert(1, "intercept", design_df.pop("intercept"))
    subj_series = pd.Series(np.repeat(design_df["subject"].to_numpy(), num_frames))
    if categorical:  # one-hot-encode categorical variable, dropping first
        design_df = pd.get_dummies(subject_variables_df, columns=[variable], drop_first=True, dtype=int)
    design_arr = design_df.drop(columns=["subject"]).to_numpy()
    design_arr = np.repeat(design_arr, num_frames, axis=0)
    frame_arr_ = (
        pd.get_dummies(pd.Series(list(range(num_frames))), drop_first=True, dtype=int)
    )
    frame_arr = np.vstack([frame_arr_] * len(design_df))
    design_arr = np.concatenate((design_arr, frame_arr), axis=1)
    frame_column_names = [f"frame_{i}" for i in range(1, num_frames)]
    design_df = pd.DataFrame(
        design_arr, 
        columns=list(design_df.columns[1:]) + frame_column_names
    )
    if categorical:
        cat_columns = [c for c in design_df.columns if c.startswith(variable)]
        for cat_column, frame_no in product(cat_columns, frame_column_names):
            interaction_name = f"{cat_column}_{frame_no}_interaction"
            design_df[interaction_name] = design_df[cat_column] * design_df[frame_no]
    else:
        for frame_no in frame_column_names:
            interaction_name = f"{variable}_{frame_no}_interaction"
            design_df[interaction_name] = design_df[variable] * design_df[frame_no]
    design_df.insert(0, "subject", subj_series)
    return design_df, variable


class TwoWayAnovaModel:
    def __init__(
        self,
        activation_img: nib.Cifti2Image | nib.Nifti1Image,
        design_df: pd.DataFrame,
        model_desc: ModelDesc,
        num_frames: int,
        var_of_interest: str,
        alpha: float | list[float] = 0.05,
        **kwargs,
    ):
        self.activation_img = activation_img
        self.fdata = activation_img.get_fdata()
        self.design_df_with_sub = design_df
        self.design_df = self.design_df_with_sub.drop(columns=["subject"])
        self.num_frames = num_frames
        self.var_of_interest = var_of_interest
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
            activation_img_path = self.model_outdir / "stacked_activation.dtseries.nii"
            nib.save(self.activation_img, activation_img_path)
            if config.fwhm > 0:
                l_surf_path, r_surf_path = get_template_midthickness_paths_from_cifti_header(
                    self.activation_img.header, self.model_desc["space"]
                )
                smoothed_activation_img_path = self.model_outdir / "smoothed_stacked_activation.dtseries.nii"
                subprocess.run(shlex.split(
                    "wb_command -cifti-smoothing "
                    f"{str(activation_img_path.resolve())} {config.fwhm} {config.fwhm} COLUMN {str(smoothed_activation_img_path.resolve())} -fwhm "
                    f"-left-surface {l_surf_path} -right-surface {r_surf_path}"
                ))
                self.activation_img = nib.load(smoothed_activation_img_path)
                self.fdata = self.activation_img.get_fdata()
        self.design_df_with_sub.to_csv(self.model_outdir / "design_matrix.tsv", sep="\t", index=False)

    def fit(self):
        print(f"Running {self.model_desc}")
        self._fit()
        self._save()

    def _fit(self):
        design_matrix = self.design_df.reset_index(drop=True)
        design_matrix_no_int = design_matrix[[col for col in self.design_df if "interaction" not in col]]
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
        # Greeenhouse-Geisser sphericity correction
        res = (Y - (design_matrix_arr @ beta)).reshape((-1, self.num_frames, beta.shape[-1]))
        epsilons = np.zeros(beta.shape[-1], dtype=np.float32)
        for v in range(res.shape[-1]):
            res_v = res[:, :, v]
            S = np.cov(res_v, rowvar=False)
            mean_diag, grand_mean, row_means = np.mean(np.diag(S)), np.mean(S), np.mean(S, axis=1)
            df_num_ = (self.num_frames**2) * ((mean_diag - grand_mean)**2)
            df_denom_ = (self.num_frames - 1) * (np.sum(S**2) - 2 * self.num_frames * np.sum(row_means**2) + (self.num_frames**2) * (grand_mean**2))
            if df_denom_ == 0:
                epsilons[v] = 1.0
            else:
                epsilons[v] = np.clip(df_num_ / df_denom_, 1.0 / (self.num_frames - 1), 1.0)
        pval_sphericity_corr = stats.f.sf(fstat, df_num*epsilons, df_denom*epsilons)
        self.fstat = fstat.reshape(img_shape)
        self.uncorr_pval = pval.reshape(img_shape)
        self.epsilons = epsilons.reshape(img_shape)
        self.pval_sphericity_corr = pval_sphericity_corr.reshape(img_shape)

    def _save(self):
        if isinstance(self.activation_img, nib.Nifti1Image):
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pvals),
                ("fstat", self.fstat),
                ("epsilons", self.epsilons),
                ("pval_sphericity_corr", self.pval_sphericity_corr),
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
        elif isinstance(self.activation_img, nib.Cifti2Image):
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pval),
                ("fstat", self.fstat),
                ("epsilons", self.epsilons),
                ("pval_sphericity_corr", self.pval_sphericity_corr),
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
        

    def __permute_design_matrix(self) -> pd.DataFrame:
        # Shuffle between-subject variable by whole-block, within-subject variables freely
        between_groups = []
        between_group_columns = (
            [self.var_of_interest] + 
            [c for c in self.design_df_with_sub.columns if any([
                re.match(r'frame_\d\d', c),
                re.match(rf'{self.var_of_interest}_frame_\d\d', c)
            ])]
        )
        for _, group_df in self.design_df_with_sub.groupby("subject"):
            between_groups.append(group_df[between_group_columns].sample(frac=1).reset_index(drop=True))
        random.shuffle(between_groups)
        df_ = self.design_df_with_sub.copy()
        df_[between_group_columns] = pd.concat(between_groups, ignore_index=True)
        return df_

