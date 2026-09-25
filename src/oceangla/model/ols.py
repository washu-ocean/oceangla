import itertools
import logging
from collections import defaultdict, OrderedDict
from pathlib import Path
import json
import math

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
)
from .volume_utils import (
    get_volume_array_from_cifti_array,
    get_biggest_voxel_cluster_sizes,
    get_voxel_clusters,
)
from .model import ModelDesc

logger = logging.getLogger(__name__)


def save_null_histogram(
    data: list | np.ndarray,
    title: str,
    out_path: str | Path,
    units: str = "mm"
):
    plt.hist(
        data,
        bins=math.floor(math.sqrt(len(data)))
    )
    plt.title(title, wrap=True)
    plt.xlabel(f"Cluster size, in {units}")
    plt.ylabel("Frequency")
    plt.savefig(out_path)
    plt.clf()


def run_ols_model(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame
):
    activation_img = __get_activation_img(model_desc, subject_activation_df)
    design_df = __get_design_df(model_desc, subject_variables_df)
    OLSModel(
        activation_img,
        design_df,
        model_desc,
        perms=config.perms,
        alpha=config.alphas,
        volume_cluster_strategy=config.volume_cluster_strategy,
        dlabel_paths=config.dlabel_paths,
        separate_null_by_hemisphere=config.separate_null_by_hemisphere,
        separate_null_by_parameter=config.separate_null_by_parameter,
    ).fit()

def __get_activation_img(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame,
):
    scalars, conditions = (
        [1 if c[0] == "+" else -1 for c in model_desc["depvars"]],
        [c[1:] for c in model_desc["depvars"]]
    )
    if len(scalars) > 1:  # Make sure we're scaling these in the right order after sorting the dataframe by subject/condition
        sorted_pairs = sorted(zip(scalars, conditions), key=lambda tup : tup[1])
        scalars, conditions = (
            [t[0] for t in sorted_pairs],
            [t[1] for t in sorted_pairs]
        )
    space, task, session = (
        model_desc["space"],
        model_desc["task"],
        model_desc["session"],
    )
    if len(conditions) > 1:  # compute a contrast for each subject, then concatenate
        paths = (
            subject_activation_df
            .query(
                "condition in @conditions and "
                "frame_no == -1 and "
                "task == @task and "
                "space == @space and "
                "session == @session "
            )
            .sort_values(by=["subject", "condition"])
        )["path"]
        imgs = [nib.load(p) for p in paths]
        if isinstance(imgs[0], nib.Cifti2Image):
            fdatas = [img.get_fdata() for img in imgs]
            fdata_stacked_list = []
            for i in range(len(conditions)):
                fdata_stacked_list.append(np.concatenate(fdatas[i::len(conditions)], axis=0, dtype=np.float32))
                fdata_stacked_list[-1] *= scalars[i]
            fdata_stacked = np.sum(fdata_stacked_list, axis=1, dtype=np.float32)
            return nib.Cifti2Image(
                fdata_stacked,
                header=(
                    ScalarAxis(name=sorted(subject_activation_df["subject"].unique())),
                    imgs[0].header.get_axis(1)
                ),
                nifti_header=imgs[0].nifti_header
            )
        elif isinstance(imgs[0], nib.Nifti1Image):
            fdatas = [img.get_fdata() for img in imgs]
            fdata_stacked_list = []
            for i in range(len(conditions)):
                fdata_stacked_list.append(np.concatenate(fdatas[i::len(conditions)], axis=3, dtype=np.float32))
                fdata_stacked_list[-1] *= scalars[i]
            fdata_stacked = np.sum(fdata_stacked_list, axis=(0, 1, 2), dtype=np.float32)
            return nib.Nifti1Image(
                fdata_stacked,
                affine=imgs[0].affine,
                header=imgs[0].header
            )
        else:
            raise ValueError(f"Unexpected image type {type(imgs[0])} (this shouldn't happen)")
    else:
        paths = (
            subject_activation_df
            .query(
                "condition == @conditions[0] and "
                "frame_no == -1 and "
                "task == @task and "
                "space == @space and "
                "session == @session "
            )
            .sort_values(by="subject")
        )["path"]
        imgs = [nib.load(p) for p in paths]
        fdatas = [img.get_fdata() for img in imgs]
        if isinstance(imgs[0], nib.Cifti2Image):
            fdata_stacked = np.concatenate(fdatas, axis=0)
            fdata_stacked *= scalars[0]
            return nib.Cifti2Image(
                fdata_stacked,
                header=(
                    ScalarAxis(name=sorted(subject_activation_df["subject"].unique())),
                    imgs[0].header.get_axis(1)
                ),
                nifti_header=imgs[0].nifti_header
            )
        elif isinstance(imgs[0], nib.Nifti1Image):
            fdata_stacked = np.concatenate(fdatas, axis=3)
            return nib.Nifti1Image(
                fdata_stacked,
                affine=imgs[0].affine,
                header=imgs[0].header
            )
        else:
            raise ValueError(f"Unexpected image type {type(imgs[0])} (this shouldn't happen)")


def __get_design_df(
    model_desc: ModelDesc,
    subject_variables_df: pd.DataFrame,
) -> pd.DataFrame:
    scalars, subject_variables = (
        [1 if c[0] == "+" else -1 for c in model_desc["indepvars"]],
        [c[1:] for c in model_desc["indepvars"]]
    )
    design_df = (
        subject_variables_df[["subject", *subject_variables]]
        .sort_values(by="subject")
        .drop(columns=["subject"])
        .reset_index(drop=True)
    )
    design_df["intercept"] = 1
    design_df.insert(0, "intercept", design_df.pop("intercept"))
    return design_df


    
class OLSModel:
    def __init__(
        self,
        activation_img: nib.Cifti2Image | nib.Nifti1Image,
        design_df: pd.DataFrame,
        model_desc: ModelDesc,
        perms: int = 0,
        alpha: float | list[float] = 0.05,
        l_area_path: Path | None = None,
        r_area_path: Path | None = None,
        volume_cluster_strategy: str = "NN1",
        dlabel_paths: list[Path] | None = None,
        separate_null_by_hemisphere: bool = False,
        separate_null_by_parameter: bool = False,
        **kwargs,
    ):
        self.activation_img = activation_img
        self.fdata = activation_img.get_fdata()
        self.design_df = design_df
        self.value_names = list(design_df.columns)
        self.model_desc = model_desc
        self.model_outdir = config.outdir_path / sanitize_filename(self.model_desc["model_name"])
        if not self.model_outdir.is_dir():
            self.model_outdir.mkdir(parents=True, exist_ok=True)
        self.perms = perms
        self.alphas = [alpha] if isinstance(alpha, float) else alpha
        self.uncorr_pvals = None
        self.tstats = None
        self.betas = None
        self.ses = None
        self.fdr_corr_pvals = []
        self.fwer_corr_pvals = []
        self.clus_corr_pvals = []
        self.volume_cluster_strategy = volume_cluster_strategy
        self.separate_null_by_hemisphere = separate_null_by_hemisphere
        self.separate_null_by_parameter = separate_null_by_parameter
        if dlabel_paths is not None:
            self.dlabel_paths = [str(p.resolve()) for p in dlabel_paths]
            self.dlabel_path_to_id = OrderedDict()
            for i, path in enumerate(self.dlabel_paths):
                self.dlabel_path_to_id[path] = f"network{i + 1}"
            with open(self.model_outdir / "network_mappings.json", "w") as f:
                json.dump(self.dlabel_path_to_id, f, indent=4)
                logger.info(f"Wrote {(self.model_outdir / 'network_mappings.json').resolve()!s}")

        # volume-specific variables
        self.volume_mask = None
        self.__biggest_vol_cluster_sizes = defaultdict(list)
        if isinstance(self.activation_img, nib.Nifti1Image):
            self.out_suffix = ".nii.gz"
            voxel_sizes = self.activation_img.header.get_zooms()[:3]
            # first check if any template resolution matches

            # TODO: handle cohorts
            # TODO: write test for making sure cohort-specific spaces
            # work
            for k, v in tflow.get_metadata(self.model_desc["space"])["res"].items():
                if np.allclose(self.activation_img.header.get_zooms()[:3], v["zooms"]):
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
                            resolution=np.floor(self.activation_img.header.get_zooms()[0]),
                            desc="brain",
                            suffix="mask",
                        )
                    ),
                    target_affine=self.affine,
                    target_shape=self.fdata.shape[:3],
                    interpolation="nearest",
                )

        # surface-specific variables
        self.__biggest_l_surf_cluster_sizes = defaultdict(list)
        self.__biggest_r_surf_cluster_sizes = defaultdict(list)
        if isinstance(self.activation_img, nib.Cifti2Image) and hasattr(self.activation_img.header.get_axis(1), 'vertex'):  # if doesn't have 'vertex' attr, then it has a ParcelAxis
            self.out_suffix = ".dscalar.nii"
            l_surf_img, r_surf_img = get_template_midthicknesses_from_cifti_header(
                self.activation_img.header, self.model_desc["space"]
            )
            self.l_faces, self.r_faces = (
                l_surf_img.darrays[1].data,
                r_surf_img.darrays[1].data,
            )
            self.l_numverts, self.r_numverts = (
                int(np.max(self.l_faces)) + 1,
                int(np.max(self.r_faces)) + 1,
            )
            self.l_neigh = build_adjacency_from_faces(self.l_numverts, self.l_faces)
            self.r_neigh = build_adjacency_from_faces(self.r_numverts, self.r_faces)
            self.l_area = (
                None if l_area_path is None else nib.load(l_area_path).darrays[0].data
            )
            self.r_area = (
                None if r_area_path is None else nib.load(r_area_path).darrays[0].data
            )
        elif isinstance(self.activation_img, nib.Cifti2Image) and isinstance(self.activation_img.header.get_axis(1), nib.cifti2.cifti2_axes.ParcelsAxis):
            self.out_suffix = ".pscalar.nii"

    def fit(self):
        if self.perms > 0:
            for perm in range(self.perms):
                print(f"Running permutation {perm + 1} of {self.model_desc}")
                self._fit(permuted_design_matrix=self.__get_permuted_design_matrix())
        print(f"Running {self.model_desc}")
        self._fit()
        self._save()
        if self.perms > 0:
            self._cluster_correct()
        self._fdr_correct()

    def _fit(self, permuted_design_matrix=None):
        design_matrix = (
            permuted_design_matrix
            if permuted_design_matrix is not None
            else self.design_df
        )
        design_matrix_arr = design_matrix.to_numpy()
        if isinstance(self.activation_img, nib.Cifti2Image):
            img_shape = (design_matrix_arr.shape[1], self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.activation_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], design_matrix_arr.shape[1])
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))
        else:
            raise ValueError(f"Cannot fit GLM for image type: {type(self.activation_img)}")
        n, p = design_matrix_arr.shape
        beta, ssr, rank, s = np.linalg.lstsq(
            design_matrix_arr,
            self.fdata.reshape(glm_input_shape),
            rcond=None,
        )
        sigma_sq = ssr / (n - p)
        v_cov_diag = np.repeat(
            np.diag(np.linalg.inv(design_matrix_arr.T @ design_matrix_arr))[:, np.newaxis],
            sigma_sq.shape[0],
            axis=1
        )
        se = np.sqrt(
            v_cov_diag * sigma_sq
        )
        tstat = beta / se
        pval = 2 * (1 - stats.t.cdf(np.abs(tstat), df=n - p))
        betas = beta.reshape(img_shape)
        ses = se.reshape(img_shape)
        tstats = tstat.reshape(img_shape)
        pvals = pval.reshape(img_shape)
        if permuted_design_matrix is not None:
            self.__add_cluster_sizes(pvals)
        else:
            self.uncorr_pvals = pvals
            self.tstats = tstats
            self.betas = betas
            self.ses = ses

    def __add_cluster_sizes(self, pvals: np.ndarray):
        if isinstance(self.activation_img, nib.Cifti2Image):
            self.__add_cifti_surf_cluster_sizes(pvals)
            self.__add_cifti_vol_cluster_sizes(pvals)
        elif isinstance(self.activation_img, nib.Nifti1Image):
            self.__add_nifti_cluster_sizes(pvals)

    def __add_cifti_surf_cluster_sizes(self, pvals: np.ndarray):
        l_pvals, r_pvals = extract_hemi_values(
            pvals, self.activation_img.header, self.l_numverts, self.r_numverts
        )
        for alpha in self.alphas:
            self.__biggest_l_surf_cluster_sizes[alpha].extend(
                get_biggest_surface_clusters(
                    l_pvals, alpha, self.l_neigh, self.l_area
                )
            )
            self.__biggest_r_surf_cluster_sizes[alpha].extend(
                get_biggest_surface_clusters(
                    r_pvals, alpha, self.r_neigh, self.r_area
                )
            )

    def __add_cifti_vol_cluster_sizes(self, pvals: np.ndarray):
        pval_volume = get_volume_array_from_cifti_array(pvals, self.activation_img.header)
        for alpha in self.alphas:
            threshold_mask = (pval_volume < alpha)
            self.__biggest_vol_cluster_sizes[alpha].extend([
                get_biggest_voxel_cluster_sizes(get_voxel_clusters(threshold_mask[..., vol], strategy=self.volume_cluster_strategy))
                for vol in range(threshold_mask.shape[-1])
            ])

    def __add_nifti_cluster_sizes(self, pvals: np.ndarray):
        for alpha in self.alphas:
            threshold_mask = (pvals < alpha)
            self.__biggest_vol_cluster_sizes[alpha].extend([
                get_biggest_voxel_cluster_sizes(get_voxel_clusters(threshold_mask[..., vol], strategy=self.volume_cluster_strategy))
                for vol in range(threshold_mask.shape[-1])
            ])

    def _fdr_correct(self):
        if isinstance(self.activation_img, nib.Cifti2Image):
            self._fdr_correct_cifti()
            if hasattr(self, 'dlabel_paths') and self.dlabel_paths is not None:
                for dlabel_path in self.dlabel_paths:
                    self._fdr_correct_cifti(dlabel_path=dlabel_path)
        elif isinstance(self.activation_img, nib.Nifti1Image):
            self._fdr_correct_nifti()

    def _fdr_correct_cifti(self, dlabel_path=None):
        fdr_corr_pvals = np.empty(
            (len(self.value_names) * len(self.alphas), self.uncorr_pvals.shape[1])
        )
        if dlabel_path is not None:
            logger.info(f"Running within-dlabel FDR correction with areas defined by {dlabel_path}")
            dlabel_img = nib.load(dlabel_path)
            common_structs = cifti_compatible_structures((self.activation_img.header, dlabel_img.header))
            if len(common_structs) == 0:
                raise ValueError(f"No compatible CIFTI structures between image and {dlabel_path}")
            dlabel_fdata = dlabel_img.get_fdata()
            unique_labels = np.unique(dlabel_img.get_fdata())
            for alpha_idx, alpha in enumerate(self.alphas):
                for value_idx in range(self.uncorr_pvals.shape[0]):
                    for label in unique_labels:
                        label_indices = np.argwhere(dlabel_fdata == label)
                        pval_vec = self.uncorr_pvals[value_idx, label_indices].copy()
                        pval_vec[np.isnan(pval_vec)] = 1
                        fdr_corr_pvals[value_idx * alpha_idx + value_idx, label_indices] = fdr_correct(
                            pval_vec, alpha=alpha
                        )
            fdr_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(
                fdr_corr_pvals,
                (
                    nib.cifti2.cifti2_axes.ScalarAxis(
                        [
                            f"{valname}_{alpha:.4f}"
                            for valname, alpha in itertools.product(
                                self.value_names, self.alphas
                            )
                        ]
                    ),
                    self.activation_img.header.get_axis(1),
                ),
            )
            nib.save(
                fdr_corr_pvals_cifti,
                p := self.model_outdir
                / f"{sanitize_filename(self.model_desc['model_name'])}_fdr_corr_{self.dlabel_path_to_id[dlabel_path]}{self.out_suffix}",
            )
            logger.info(f"Saved {p!s}")
        else:
            logger.info("Running FDR correction")
            for alpha_idx, alpha in enumerate(self.alphas):
                for value_idx in range(self.uncorr_pvals.shape[0]):
                    pval_vec = self.uncorr_pvals[value_idx, :].copy()
                    pval_vec[np.isnan(pval_vec)] = 1
                    fdr_corr_pvals[value_idx * alpha_idx + value_idx, :] = fdr_correct(
                        pval_vec, alpha=alpha
                    )

            fdr_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(
                fdr_corr_pvals,
                (
                    nib.cifti2.cifti2_axes.ScalarAxis(
                        [
                            f"{valname}_{alpha:.4f}"
                            for valname, alpha in itertools.product(
                                self.value_names, self.alphas
                            )
                        ]
                    ),
                    self.activation_img.header.get_axis(1),
                ),
            )
            nib.save(
                fdr_corr_pvals_cifti,
                p := self.model_outdir
                / f"{sanitize_filename(self.model_desc['model_name'])}_fdr_corr{self.out_suffix}",
            )
            logger.info(f"Saved {p!s}")

    def _fdr_correct_nifti(self):
        for alpha in self.alphas:
            for value_idx, value_name in enumerate(self.value_names):
                orig_shape = self.uncorr_pvals[..., value_idx].shape
                flattened_pvals = self.uncorr_pvals[..., value_idx].flatten()
                fdr_corr = fdr_correct(flattened_pvals, alpha=alpha).reshape(orig_shape)
                fdr_corr_img = nib.Nifti1Image(fdr_corr, self.affine, header=self.activation_img.header)
                nib.save(fdr_corr_img, p := self.model_outdir / f"{sanitize_filename(self.model_desc['model_name'])}_beta-{value_name}_fdr_corr_{alpha:.4f}.nii.gz")
                logger.info(f"Saved {p!s}")

    def _cluster_correct(self):
        if isinstance(self.activation_img, nib.Cifti2Image):
            self._cluster_correct_cifti()
        elif isinstance(self.activation_img, nib.Nifti1Image):
            self._cluster_correct_nifti()

    def _cluster_correct_cifti(self):
        l_pvals, r_pvals = extract_hemi_values(
            self.uncorr_pvals, self.activation_img.header, self.l_numverts, self.r_numverts
        )
        l_clus_corr = np.ones(
            (len(self.value_names) * len(self.alphas), self.l_numverts),
            dtype=np.float32,
        )
        r_clus_corr = np.ones(
            (len(self.value_names) * len(self.alphas), self.r_numverts),
            dtype=np.float32,
        )
        full_clus_corr = np.full(
            (len(self.value_names) * len(self.alphas), self.uncorr_pvals.shape[1]),
            np.nan,
        )
        volume = get_volume_array_from_cifti_array(self.uncorr_pvals, self.activation_img.header)
        volume_voxel_indices = self.activation_img.header.get_axis(1).voxel[(self.activation_img.header.get_axis(1).voxel != -1).all(axis=1)]
        volume_cifti_indices = np.argwhere((self.activation_img.header.get_axis(1).voxel != -1).all(axis=1)).flatten()
        for alpha_idx, alpha in enumerate(self.alphas):
            for value_idx, value_name in enumerate(self.value_names):
                l_mask = np.isfinite(l_pvals[value_idx, :]) & (
                    l_pvals[value_idx, :] < alpha
                )
                cluster_null_dist = (
                    self.__biggest_l_surf_cluster_sizes[alpha] + self.__biggest_r_surf_cluster_sizes[alpha]
                    if not self.separate_null_by_hemisphere
                    else self.__biggest_l_surf_cluster_sizes[alpha]
                )
                if self.separate_null_by_parameter:
                    cluster_null_dist = cluster_null_dist[value_idx::len(self.value_names)]
                save_null_histogram(
                    cluster_null_dist,
                    f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, tested against left hemisphere",
                    self.model_outdir / f"null_lh_alpha{alpha:.4f}_cond{value_name}.png"
                )
                for cluster in get_cluster_index_groups(l_mask, self.l_neigh):
                    cluster_size = (
                        len(cluster)
                        if self.l_area is None
                        else np.sum(self.l_area[cluster])
                    )
                    cluster_size = np.int64(cluster_size)
                    sizes_larger_than_this_cluster = np.float32(
                        np.sum(
                            cluster_null_dist >= cluster_size
                        )
                    )
                    l_clus_corr[value_idx * alpha_idx + value_idx, cluster] = (
                        sizes_larger_than_this_cluster / (len(cluster_null_dist) + 1)
                    )
                r_mask = np.isfinite(r_pvals[value_idx, :]) & (
                    r_pvals[value_idx, :] < alpha
                )
                cluster_null_dist = (
                    self.__biggest_l_surf_cluster_sizes[alpha] + self.__biggest_r_surf_cluster_sizes[alpha]
                    if not self.separate_null_by_hemisphere
                    else self.__biggest_r_surf_cluster_sizes[alpha]
                )
                if self.separate_null_by_parameter:
                    cluster_null_dist = cluster_null_dist[value_idx::len(self.value_names)]
                save_null_histogram(
                    cluster_null_dist,
                    f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, tested against right hemisphere",
                    self.model_outdir / f"null_rh_alpha{alpha:.4f}_cond{value_name}.png"
                )
                for cluster in get_cluster_index_groups(r_mask, self.r_neigh):
                    cluster_size = (
                        len(cluster)
                        if self.r_area is None
                        else np.sum(self.r_area[cluster])
                    )
                    cluster_size = np.int64(cluster_size)
                    sizes_larger_than_this_cluster = np.float32(
                        np.sum(
                            cluster_null_dist >= cluster_size
                        )
                    )
                    r_clus_corr[value_idx * alpha_idx + value_idx, cluster] = (
                        sizes_larger_than_this_cluster / (len(cluster_null_dist) + 1)
                    )
                volume_mask = volume[..., value_idx] < alpha
                clus_corr_volume = np.full_like(volume_mask, np.nan, dtype=np.float32)
                cluster_null_dist = (
                    self.__biggest_vol_cluster_sizes[alpha]
                    if not self.separate_null_by_parameter
                    else self.__biggest_vol_cluster_sizes[alpha][value_idx::len(self.value_names)]
                )
                save_null_histogram(
                    cluster_null_dist,
                    f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, tested against subcortical voxels",
                    self.model_outdir / f"null_subcort_alpha{alpha:.4f}_cond{value_name}.png"
                )
                for cluster in get_voxel_clusters(volume_mask):
                    cluster_size = np.int64(cluster.shape[0])
                    sizes_larger_than_this_cluster = np.sum(cluster_null_dist >= cluster_size)
                    clus_p = sizes_larger_than_this_cluster / (len(cluster_null_dist) + 1)
                    clus_corr_volume[tuple(cluster.T)] = clus_p
                full_clus_corr[value_idx * alpha_idx + value_idx, tuple(volume_cifti_indices.T)] = clus_corr_volume[tuple(volume_voxel_indices.T)].T

        for name, slc, bmodel in self.activation_img.header.get_axis(1).iter_structures():
            if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
                vidx = bmodel.vertex.astype(np.int64)
                full_clus_corr[:, slc] = l_clus_corr[:, vidx]
            elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
                vidx = bmodel.vertex.astype(np.int64)
                full_clus_corr[:, slc] = r_clus_corr[:, vidx]
        clus_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(
            full_clus_corr,
            (
                nib.cifti2.cifti2_axes.ScalarAxis(
                    [
                        f"{valname}_{alpha:.4f}"
                        for valname, alpha in itertools.product(
                            self.value_names, self.alphas
                        )
                    ]
                ),
                self.activation_img.header.get_axis(1),
            ),
        )
        nib.save(
            clus_corr_pvals_cifti,
            p := self.model_outdir
            / f"{sanitize_filename(self.model_desc['model_name'])}_clus_corr{self.out_suffix}",
        )
        logger.info(f"Saved {p!s}")

    def _cluster_correct_nifti(self):
        for alpha in self.alphas:
            for value_idx, value_name in enumerate(self.value_names):
                clus_corr = np.full_like(self.uncorr_pvals[..., value_idx], np.nan)
                cluster_null_dist = (
                    self.__biggest_vol_cluster_sizes[alpha]
                    if not self.separate_null_by_parameter
                    else self.__biggest_vol_cluster_sizes[alpha][value_idx::len(self.value_names)]
                )
                save_null_histogram(
                    cluster_null_dist,
                    f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, tested against voxels",
                    self.model_outdir / f"null_subcort_alpha{alpha:.4f}_cond{value_name}.png"
                )
                for cluster in get_voxel_clusters(self.uncorr_pvals[..., value_idx]):
                    cluster_size = np.int64(cluster.shape[0])
                    sizes_larger_than_this_cluster = np.sum(cluster_null_dist >= cluster_size)
                    clus_p = sizes_larger_than_this_cluster / (len(cluster_null_dist) + 1)
                    clus_corr[tuple(cluster.T)] = clus_p
                clus_corr_img = nib.Nifti1Image(clus_corr, self.affine, header=self.activation_img.header)
                nib.save(clus_corr_img, p := self.model_outdir / f"{sanitize_filename(self.model_desc['model_name'])}_beta-{value_name}_clus_corr_{alpha:.4f}.nii.gz")
                logger.info(f"Saved {p!s}")

    def _save(self):
        if isinstance(self.activation_img, nib.Nifti1Image):
            self._save_nifti()
        elif isinstance(self.activation_img, nib.Cifti2Image):
            self._save_cifti()
        self.design_df.to_csv(self.model_outdir / "design_matrix.tsv", sep="\t")

    def _save_cifti(self):
        for datatype, data in (
            ("uncorr_pvals", self.uncorr_pvals),
            ("betas", self.betas),
            ("ses", self.ses),
            ("tstats", self.tstats),
        ):
            if data is None:
                continue
            img = nib.cifti2.cifti2.Cifti2Image(
                data,
                (
                    nib.cifti2.cifti2_axes.ScalarAxis(self.value_names),
                    self.activation_img.header.get_axis(1),
                ),
            )
            nib.save(
                img,
                p := self.model_outdir
                / f"{sanitize_filename(self.model_desc['model_name'])}_{datatype}{self.out_suffix}",
            )
            logger.info(f"Saved {p!s}")
            del img

    def _save_nifti(self):
        for datatype, data in (
            ("uncorr_pvals", self.uncorr_pvals),
            ("betas", self.betas),
            ("ses", self.ses),
            ("tstats", self.tstats),
        ):
            if data is None:
                continue
            for idx, value_name in enumerate(
                self.value_names
            ):  # Have to save a different image per-beta bcuz NIFTI doesn't have volume labelling :-(
                img = nib.Nifti1Image(data[..., idx], self.affine, header=self.activation_img.header)
                nib.save(
                    img,
                    p := self.model_outdir
                    / f"{sanitize_filename(self.model_desc['model_name'])}_beta-{value_name}_{datatype}.nii.gz",
                )
                logger.info(f"Saved {p!s}")
                del img

    def __get_permuted_design_matrix(self) -> pd.DataFrame:
        """
        Return DataFrame, which is a copy of self.design_df with every column shuffled.
        """
        permuted_design_matrix = self.design_df.copy()
        for column in permuted_design_matrix.columns:
            if column != "intercept":
                permuted_design_matrix[column] = (
                    permuted_design_matrix[column].sample(frac=1).array
                )
        return permuted_design_matrix
