import itertools
import logging
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import progressbar

# import ipdb
from nilearn.image import resample_img
from pathvalidate import sanitize_filename
from scipy import stats
from templateflow import api as tflow

# from .model import GroupLevelModel
from ..config import config
from ..image.utils import get_spatial_slices
from .correction import fdr_correct
from .surface_utils import (
    build_adjacency_from_faces,
    extract_hemi_values,
    get_biggest_clusters_from_pmap,
    get_cluster_index_groups,
    get_template_midthicknesses_from_cifti_header,
)

logger = logging.getLogger(__name__)


class OLSModel:
    def __init__(
        self,
        activation: dict,
        design_matrix: pd.DataFrame,
        model_desc: str = "nondescript-model",
        perms: int = 0,
        alpha: float | list[float] = 0.05,
        l_area_path: Path = None,
        r_area_path: Path = None,
        **kwargs,
    ):
        self.design_matrix = design_matrix
        self.image_type = activation.get("type", "NA")
        self.value_names = tuple(design_matrix.reset_index(drop=True).columns)
        self.space = activation["space"]
        self.activation = activation["activation"]
        self.header = activation["header"]
        self.nifti_header = activation.get("nifti_header", None)
        self.affine = activation.get("affine", None)
        self.model_desc = model_desc
        self.model_outdir = config.outdir_path / sanitize_filename(self.model_desc)
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

        # volume-specific variables
        self.volume_mask = None
        self.__biggest_vol_cluster_sizes = defaultdict(list)
        if self.image_type == "NIFTI":
            self.voxel_sizes = self.header.get_zooms()[:3]
            # first check if any template resolution matches
            for k, v in tflow.get_metadata(self.space)["res"].items():
                if np.allclose(self.header.get_zooms()[:3], v["zooms"]):
                    self.volume_mask = nib.load(
                        tflow.get(
                            self.space,
                            resolution=self.header.get_zooms()[0],
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
                            self.space,
                            resolution=np.floor(self.header.get_zooms()[0]),
                            desc="brain",
                            suffix="mask",
                        )
                    ),
                    target_affine=self.affine,
                    target_shape=self.activation.shape[:3],
                    interpolation="nearest",
                )

        # surface-specific variables
        self.__biggest_l_surf_cluster_sizes = defaultdict(list)
        self.__biggest_r_surf_cluster_sizes = defaultdict(list)
        if self.image_type == "CIFTI":
            l_surf_img, r_surf_img = get_template_midthicknesses_from_cifti_header(
                self.header, self.space
            )
            # breakpoint()
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

    def fit(self):
        if self.perms > 0:
            for perm in range(self.perms):
                print(f"Running permutation {perm + 1} of {self.model_desc}")
                self._fit(permuted_design_matrix=self.get_permuted_design_matrix())
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
            else self.design_matrix
        )
        design_matrix_arr = design_matrix.to_numpy()
        if self.image_type == "CIFTI":
            img_shape = (design_matrix_arr.shape[1], self.activation.shape[1])
        elif self.image_type == "NIFTI":
            img_shape = (*self.activation.shape[:3], design_matrix_arr.shape[1])
        else:
            raise ValueError(f"Cannot fit GLM for image type: {self.image_type}")
        spatial_slices, no_of_spatial_slices = get_spatial_slices(
            self.activation, self.image_type, self.volume_mask
        )
        pvals, tstats, betas, ses = (np.full(img_shape, np.nan, dtype=np.float32),) * 4
        with progressbar.ProgressBar(
            max_value=no_of_spatial_slices, redirect_stdout=True
        ) as pbar:
            for spatial_slice in spatial_slices:
                try:
                    n, p = design_matrix_arr.shape
                    beta, ssr, rank, s = np.linalg.lstsq(
                        design_matrix_arr,
                        np.squeeze(self.activation[spatial_slice]),
                        rcond=None,
                    )
                    sigma_sq = ssr[0] / (n - p)
                    v_cov = (
                        np.linalg.inv(design_matrix_arr.T @ design_matrix_arr)
                        * sigma_sq
                    )
                    se = np.sqrt(np.diag(v_cov))
                    tstat = beta / se
                    pval = np.array(
                        [2 * (1 - stats.t.cdf(np.abs(t), df=n - p)) for t in tstat]
                    )
                    for value_arr, value_vec in (
                        (betas, beta),
                        (ses, se),
                        (tstats, tstat),
                        (pvals, pval),
                    ):
                        value_arr[spatial_slice] = value_vec
                except np.linalg.LinAlgError:
                    continue
                pbar += 1
        if permuted_design_matrix is not None:
            self.__add_cluster_sizes(pvals)
        else:
            self.uncorr_pvals = pvals
            self.tstats = tstats
            self.betas = betas
            self.ses = ses

    def __add_cluster_sizes(self, pvals: np.ndarray):
        if self.image_type == "CIFTI":
            self.__add_cifti_surf_cluster_sizes(pvals)
            self.__add_cifti_vol_cluster_sizes(pvals)
        elif self.image_type == "NIFTI":
            self.__add_nifti_cluster_sizes(pvals)

    def __add_cifti_surf_cluster_sizes(self, pvals: np.ndarray):
        l_pvals, r_pvals = extract_hemi_values(
            pvals, self.header, self.l_numverts, self.r_numverts
        )
        for alpha in self.alphas:
            self.__biggest_l_surf_cluster_sizes[alpha].extend(
                get_biggest_clusters_from_pmap(
                    l_pvals, alpha, self.l_neigh, self.l_area
                )
            )
            self.__biggest_r_surf_cluster_sizes[alpha].extend(
                get_biggest_clusters_from_pmap(
                    r_pvals, alpha, self.r_neigh, self.r_area
                )
            )

    def __add_cifti_vol_cluster_sizes(self, pvals: np.ndarray):
        pass

    def _fdr_correct(self):
        if self.image_type == "CIFTI":
            self._fdr_correct_cifti()
        elif self.image_type == "NIFTI":
            self._fdr_correct_nifti()

    def _fdr_correct_cifti(self):
        fdr_corr_pvals = np.empty(
            (len(self.value_names) * len(self.alphas), self.uncorr_pvals.shape[1])
        )
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
                self.header.get_axis(1),
            ),
        )
        nib.save(
            fdr_corr_pvals_cifti,
            p := self.model_outdir
            / f"{sanitize_filename(self.model_desc)}_fdr_corr.dscalar.nii",
        )
        logger.info(f"Saved {p!s}")

    def _fdr_correct_nifti(self):
        logger.warning("NIFTI FDR correction not implemented yet.")

    def _cluster_correct(self):
        if self.image_type == "CIFTI":
            self._cluster_correct_cifti()
        elif self.image_type == "NIFTI":
            self._cluster_correct_nifti()

    def _cluster_correct_cifti(self):
        l_pvals, r_pvals = extract_hemi_values(
            self.uncorr_pvals, self.header, self.l_numverts, self.r_numverts
        )
        l_clus_corr = np.ones(
            (len(self.value_names) * len(self.alphas), self.l_numverts),
            dtype=np.float32,
        )
        r_clus_corr = np.ones(
            (len(self.value_names) * len(self.alphas), self.r_numverts),
            dtype=np.float32,
        )
        for alpha_idx, alpha in enumerate(self.alphas):
            for value_idx in range(len(self.value_names)):
                l_mask = np.isfinite(l_pvals[value_idx, :]) & (
                    l_pvals[value_idx, :] < alpha
                )
                for cluster in get_cluster_index_groups(l_mask, self.l_neigh):
                    cluster_size = (
                        len(cluster)
                        if self.l_area is None
                        else np.sum(self.l_area[cluster])
                    )
                    sizes_larger_than_this_cluster = np.float32(
                        np.sum(self.__biggest_l_surf_cluster_sizes[alpha])
                        >= cluster_size
                    )
                    l_clus_corr[value_idx * alpha_idx + value_idx, cluster] = (
                        sizes_larger_than_this_cluster / (self.perms + 1)
                    )
                r_mask = np.isfinite(r_pvals[value_idx, :]) & (
                    r_pvals[value_idx, :] < alpha
                )
                for cluster in get_cluster_index_groups(r_mask, self.r_neigh):
                    cluster_size = (
                        len(cluster)
                        if self.r_area is None
                        else np.sum(self.r_area[cluster])
                    )
                    sizes_larger_than_this_cluster = np.float32(
                        np.sum(self.__biggest_r_surf_cluster_sizes[alpha])
                        >= cluster_size
                    )
                    r_clus_corr[value_idx * alpha_idx + value_idx, cluster] = (
                        sizes_larger_than_this_cluster / (self.perms + 1)
                    )
        full_clus_corr = np.full(
            (len(self.value_names) * len(self.alphas), self.uncorr_pvals.shape[1]),
            np.nan,
        )
        for name, slc, bmodel in self.header.get_axis(1).iter_structures():
            if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
                vidx = bmodel.vertex.astype(np.int64)
                full_clus_corr[:, slc] = l_clus_corr[:, vidx]
            elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
                vidx = bmodel.vertex.astype(np.int64)
                full_clus_corr[:, slc] = r_clus_corr[:, vidx]
            else:
                full_clus_corr[:, slc] = 1.0  # not clustered here
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
                self.header.get_axis(1),
            ),
        )
        nib.save(
            clus_corr_pvals_cifti,
            p := self.model_outdir
            / f"{sanitize_filename(self.model_desc)}_clus_corr.dscalar.nii",
        )
        logger.info(f"Saved {p!s}")

    def _cluster_correct_nifti(self):
        logger.warning("NIFTI cluster correction not yet implemented.")

    def _save(self):
        if self.image_type == "NIFTI":
            self._save_nifti()
        elif self.image_type == "CIFTI":
            self._save_cifti()

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
                    self.header.get_axis(1),
                ),
            )
            nib.save(
                img,
                p := self.model_outdir
                / f"{sanitize_filename(self.model_desc)}_{datatype}.dscalar.nii",
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
                img = nib.Nifti1Image(data[..., idx], self.affine, header=self.header)
                nib.save(
                    img,
                    p := self.model_outdir
                    / f"{sanitize_filename(self.model_desc)}_beta-{value_name}_{datatype}.nii.gz",
                )
                logger.info(f"Saved {p!s}")
                del img

    def get_permuted_design_matrix(self) -> pd.DataFrame:
        """
        Return DataFrame, which is a copy of self.design_matrix with every column shuffled.
        """
        permuted_design_matrix = self.design_matrix.copy()
        for column in permuted_design_matrix.columns:
            if column != "intercept":
                permuted_design_matrix[column] = (
                    permuted_design_matrix[column].sample(frac=1).array
                )
        return permuted_design_matrix
