from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import logging
import itertools

from scipy import stats
import nibabel as nib
import numpy as np
import pandas as pd
import progressbar
# import ipdb
from joblib import Memory, Parallel, delayed
from templateflow import api as tflow
from pathvalidate import sanitize_filename

# from .model import GroupLevelModel
from ..config import config
from .surface_utils import (
    get_faces_from_gifti_surf,
    build_adjacency_from_faces,
    extract_hemi_values,
    get_biggest_clusters_from_pmap,
    get_cluster_index_groups
)
from .correction import fdr_correct


logger = logging.getLogger(__name__)


class OLSModel:
    def __init__(self,
                 activation: dict,
                 design_matrix: pd.DataFrame,
                 model_desc: str = "nondescript-model",
                 perms: int = 0,
                 alpha: float | list[float] = 0.05,
                 l_surf_path: str | Path = None,
                 r_surf_path: str | Path = None,
                 l_area_path: str | Path = None,
                 r_area_path: str | Path = None,
                 **kwargs):
        self.design_matrix = design_matrix
        self.image_type = activation.get("type", "NA")
        print(self.image_type)
        self.value_names = tuple(design_matrix.reset_index(drop=True).columns)
        self.activation = activation["activation"]
        self.header = activation["header"]
        self.nifti_header = activation.get("nifti_header", None)
        self.affine = activation.get("affine", None)
        self.model_desc = model_desc
        self.model_outdir = config.outdir_path / sanitize_filename(self.model_desc)
        if not self.model_outdir.is_dir():
            self.model_outdir.mkdir(parents=True, exist_ok=True)
        self.perms = perms
        if isinstance(alpha, float):
            self.alphas = [alpha]
        else:
            self.alphas = alpha
        self.__orig_shape = None
        self.__biggest_l_cluster_sizes = defaultdict(list)
        self.__biggest_r_cluster_sizes = defaultdict(list)
        self.uncorr_pvals = None
        self.tstats = None
        self.betas = None
        self.ses = None
        self.fdr_corr_pvals = []
        self.fwer_corr_pvals = []
        self.clus_corr_pvals = []
        if l_surf_path is not None and r_surf_path is not None:
            self.l_surf = nib.load(l_surf_path)
            self.r_surf = nib.load(r_surf_path)
            self.l_faces = get_faces_from_gifti_surf(self.l_surf)
            self.r_faces = get_faces_from_gifti_surf(self.r_surf)
            self.l_numverts = int(np.max(self.l_faces))
            self.r_numverts = int(np.max(self.r_faces))
            self.l_neigh = build_adjacency_from_faces(self.l_numverts, self.l_faces)
            self.r_neigh = build_adjacency_from_faces(self.r_numverts, self.r_faces)
            if Path(l_area_path).is_file() and Path(r_area_path).is_file():
                self.l_area = nib.load(l_area_path).darrays[0].data
                self.r_area = nib.load(r_area_path).darrays[0].data
            else:
                self.l_area, self.r_area = None, None
        else:
            self.l_surf, self.r_surf = None, None
            self.l_area, self.r_area = None, None
        print(self.__dict__)
        exit()

    def fit(self):
        if self.image_type != "CIFTI":  # need to flatten into 2d array
            raise NotImplementedError("Cannot work with non-CIFTI data yet.")
        if self.perms > 0:
            for perm in range(self.perms):
                print(f"Running permutation {perm + 1} of {self.model_desc}")
                self._fit(permuted_design_matrix=self.get_permuted_design_matrix())
        print(f"Running {self.model_desc}")
        self._fit()
        self._save()
        if self.perms > 0 and self.l_surf is not None and self.r_surf is not None:
            self._cluster_correct()
        self._fdr_correct()

    def _fit(self, permuted_design_matrix=None):
        design_matrix = permuted_design_matrix if permuted_design_matrix is not None else self.design_matrix
        design_matrix_arr = design_matrix.to_numpy()
        pvals = np.empty((design_matrix_arr.shape[1], self.activation.shape[1]))
        tstats = np.empty((design_matrix_arr.shape[1], self.activation.shape[1]))
        betas = np.empty((design_matrix_arr.shape[1], self.activation.shape[1]))
        ses = np.empty((design_matrix_arr.shape[1], self.activation.shape[1]))
        for vtx in progressbar.progressbar(range(self.activation.shape[1]), redirect_stdout=True):
            n, p = design_matrix_arr.shape
            beta, ssr, rank, s = np.linalg.lstsq(
                design_matrix_arr,
                self.activation[:, vtx],
                rcond=None
            )
            sigma_sq = ssr[0] / (n - p)
            v_cov = np.linalg.inv(design_matrix_arr.T @ design_matrix_arr) * sigma_sq
            se = np.sqrt(np.diag(v_cov))
            tstat = beta / se
            pval = np.array([2 * (1 - stats.t.cdf(np.abs(t), df=n - p)) for t in tstat])
            for value_arr, value_vec in ((betas, beta), (ses, se), (tstats, tstat), (pvals, pval)):
                value_arr[:, vtx] = value_vec
        if permuted_design_matrix is not None and self.l_surf is not None and self.r_surf is not None:  # get biggest clusters
            l_pvals, r_pvals = extract_hemi_values(pvals, self.header, self.l_numverts, self.r_numverts)
            for alpha in self.alphas:
                self.__biggest_l_cluster_sizes[alpha].extend(get_biggest_clusters_from_pmap(l_pvals, alpha, self.l_neigh, self.l_area))
                self.__biggest_r_cluster_sizes[alpha].extend(get_biggest_clusters_from_pmap(r_pvals, alpha, self.r_neigh, self.r_area))
        else:
            self.uncorr_pvals = pvals
            self.tstats = tstats
            self.betas = betas
            self.ses = ses

    def _fdr_correct(self):
        fdr_corr_pvals = np.empty((len(self.value_names) * len(self.alphas), self.uncorr_pvals.shape[1]))
        for alpha_idx, alpha in enumerate(self.alphas):
            for value_idx in range(self.uncorr_pvals.shape[0]):
                pval_vec = self.uncorr_pvals[value_idx, :].copy()
                pval_vec[np.isnan(pval_vec)] = 1
                fdr_corr_pvals[value_idx * alpha_idx + value_idx, :] = fdr_correct(pval_vec, alpha=alpha)
        fdr_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(fdr_corr_pvals, (nib.cifti2.cifti2_axes.ScalarAxis([f"{valname}_{alpha:.4f}" for valname, alpha in itertools.product(self.value_names, self.alphas)]), self.header.get_axis(1)))
        nib.save(fdr_corr_pvals_cifti, p := self.model_outdir / f"{sanitize_filename(self.model_desc)}_fdr_corr.dscalar.nii")
        logger.info(f"Saved {p!s}")

    def _cluster_correct(self):
        l_pvals, r_pvals = extract_hemi_values(self.uncorr_pvals, self.header, self.l_numverts, self.r_numverts)
        l_clus_corr = np.ones((len(self.value_names) * len(self.alphas), self.l_numverts), dtype=np.float32)
        r_clus_corr = np.ones((len(self.value_names) * len(self.alphas), self.r_numverts), dtype=np.float32)
        for alpha_idx, alpha in enumerate(self.alphas):
            for value_idx in range(len(self.value_names)):
                l_mask = np.isfinite(l_pvals[value_idx, :]) & (l_pvals[value_idx, :] < alpha)
                for cluster in get_cluster_index_groups(l_mask, self.l_neigh):
                    cluster_size = len(cluster) if self.l_area is None else np.sum(self.l_area[cluster])
                    sizes_larger_than_this_cluster = np.float32(np.sum(self.__biggest_l_cluster_sizes[alpha]) >= cluster_size)
                    l_clus_corr[value_idx * alpha_idx + value_idx, cluster] = sizes_larger_than_this_cluster / (self.perms + 1)
                r_mask = np.isfinite(r_pvals[value_idx, :]) & (r_pvals[value_idx, :] < alpha)
                for cluster in get_cluster_index_groups(r_mask, self.r_neigh):
                    cluster_size = len(cluster) if self.r_area is None else np.sum(self.r_area[cluster])
                    sizes_larger_than_this_cluster = np.float32(np.sum(self.__biggest_r_cluster_sizes[alpha]) >= cluster_size)
                    r_clus_corr[value_idx * alpha_idx + value_idx, cluster] = sizes_larger_than_this_cluster / (self.perms + 1)
        full_clus_corr = np.full((len(self.value_names) * len(self.alphas), self.uncorr_pvals.shape[1]), np.nan)
        for (name, slc, bmodel) in self.header.get_axis(1).iter_structures():
            if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
                vidx = bmodel.vertex.astype(np.int64)
                full_clus_corr[:, slc] = l_clus_corr[:, vidx]
            elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
                vidx = bmodel.vertex.astype(np.int64)
                full_clus_corr[:, slc] = r_clus_corr[:, vidx]
            else:
                full_clus_corr[:, slc] = 1.0  # not clustered here
        clus_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(full_clus_corr, (nib.cifti2.cifti2_axes.ScalarAxis([f"{valname}_{alpha:.4f}" for valname, alpha in itertools.product(self.value_names, self.alphas)]), self.header.get_axis(1)))
        nib.save(clus_corr_pvals_cifti, p := self.model_outdir / f"{sanitize_filename(self.model_desc)}_clus_corr.dscalar.nii")
        logger.info(f"Saved {p!s}")

    def _save(self):
        if self.image_type != "CIFTI":
            raise NotImplementedError("Cannot work with non-CIFTI data yet.")
        else:
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pvals),
                ("betas", self.betas),
                ("ses", self.ses),
                ("tstats", self.tstats),
            ):
                if data is not None:
                    my_cifti = nib.cifti2.cifti2.Cifti2Image(
                        data,
                        (nib.cifti2.cifti2_axes.ScalarAxis(self.value_names), self.header.get_axis(1))
                    )
                    nib.save(my_cifti, p := self.model_outdir / f"{sanitize_filename(self.model_desc)}_{datatype}.dscalar.nii")
                    logger.info(f"Saved {p!s}")
                    del my_cifti

    def get_permuted_design_matrix(self) -> pd.DataFrame:
        """
        Return DataFrame, which is a copy of self.design_matrix with every column shuffled.
        """
        permuted_design_matrix = self.design_matrix.copy()
        for column in permuted_design_matrix.columns:
            if column != "intercept":
                permuted_design_matrix[column] = permuted_design_matrix[column].sample(frac=1).values
        return permuted_design_matrix
