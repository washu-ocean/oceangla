from __future__ import annotations

import itertools
import json
import logging
import math
import random
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Tuple, Any

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.cifti2.cifti2_axes import ScalarAxis
import numpy as np
import pandas as pd
from nilearn.image import resample_img
from pathvalidate import sanitize_filename
from patsy import dmatrix
from scipy import stats
from templateflow import api as tflow
from tqdm import tqdm

from oceangla.config import config
from oceangla.correction import fdr_correct
from oceangla.image_utils.cifti import cifti_compatible_structures
from .surface_utils import (
    build_adjacency_from_faces,
    extract_hemi_values,
    get_biggest_surface_clusters,
    get_cluster_index_groups,
    get_template_midthicknesses_from_cifti_header,
    get_template_midthickness_paths_from_cifti_header,
)
from .volume_utils import (
    get_volume_array_from_cifti_array,
    get_biggest_voxel_cluster_sizes,
    get_voxel_clusters,
)
from .gpu import get_xp, to_cpu, to_gpu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def save_null_histogram(
    data: list | np.ndarray,
    title: str,
    out_path: str | Path,
    units: str = "mm",
):
    if len(data) == 0:
        return
    plt.hist(
        data,
        bins=max(1, math.floor(math.sqrt(len(data)))),
    )
    plt.title(title, wrap=True)
    plt.xlabel(f"Cluster size, in {units}")
    plt.ylabel("Frequency")
    plt.savefig(out_path)
    plt.clf()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FirstLevelContrast:
    name: str = field(default_factory=str)
    scalar: int = 1


@dataclass
class ModelSpec:
    model_type: str = field(default_factory=str)
    model_name: str = field(default_factory=str)
    firstlevel_contrasts: List[FirstLevelContrast] = field(default_factory=list)
    subject_variables: str = field(default_factory=str)
    function_args: List[str] = field(default_factory=list)
    session: str = field(default_factory=str)
    space: str = field(default_factory=str)
    task: str = field(default_factory=str)
    frame_count: int = 1


# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

def run_models(
    model_specs: List[ModelSpec],
    firstlevel_contrasts_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame,
):
    for model_spec in model_specs:
        run_model(model_spec, firstlevel_contrasts_df, subject_variables_df)


def apply_corrections(results: ModelResults, vol_method: str):
    if getattr(config, "fdr", True):
        results.fdr_correct()
    if config.perms > 0 and getattr(config, "cluster_correct", True):
        results.cluster_correct(
            n_perms=config.perms,
            alpha=config.alphas,
            method=vol_method,
        )
    if config.perms > 0 and getattr(config, "tfce", False):
        alpha_val = config.alphas[0] if isinstance(config.alphas, list) else config.alphas
        results.tfce_correct(
            n_perms=config.perms,
            alpha=alpha_val,
            method=vol_method,
            H=getattr(config, "tfce_h", 2.0),
            E=getattr(config, "tfce_e", 0.5),
            dh=getattr(config, "tfce_dh", None),
        )


def run_model(
    model_spec: ModelSpec,
    firstlevel_contrasts_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame,
):
    vol_method = (config.volume_cluster_strategy.lower() if config.volume_cluster_strategy else "nn1")
    match model_spec.model_type:
        case "OLS":
            model = OLSModel(model_spec, firstlevel_contrasts_df, subject_variables_df)
            results = model.fit()
            results.save()
            apply_corrections(results, vol_method)
        case "onesampttest":
            if (num_args := len(model_spec.function_args)) != 1:
                raise ValueError(
                    "Function onesampttest() takes 1 argument: "
                    f"'beta'. Received {num_args}: {', '.join(model_spec.function_args)}"
                )
            model_spec.firstlevel_contrasts = [FirstLevelContrast(name=model_spec.function_args[0])]
            model_spec.subject_variables = "1"
            model = OLSModel(model_spec, firstlevel_contrasts_df, subject_variables_df)
            results = model.fit()
            results.save()
            apply_corrections(results, vol_method)
        case "fir_rm_anova":
            if (num_args := len(model_spec.function_args)) != 1:
                raise ValueError(
                    "Function fir_rm_anova() takes 1 argument: "
                    f"'beta'. Received {num_args}: {', '.join(model_spec.function_args)}"
                )
            model = RMAnovaModel(model_spec, firstlevel_contrasts_df, subject_variables_df)
            results = model.fit()
            results.save()
            apply_corrections(results, vol_method)
        case "fir_twoway_rm_anova":
            if (num_args := len(model_spec.function_args)) != 2:
                raise ValueError(
                    "Function fir_twoway_rm_anova() takes 2 arguments: "
                    f"'beta' and 'variable'. Received {num_args}: {', '.join(model_spec.function_args)}"
                )
            model = TwoWayAnovaModel(model_spec, firstlevel_contrasts_df, subject_variables_df)
            results = model.fit()
            results.save()
            apply_corrections(results, vol_method)
        case _:
            raise ValueError(f"Unknown function or model type {model_spec.model_type}.")


# ---------------------------------------------------------------------------
# ModelResults Base and Subclasses
# ---------------------------------------------------------------------------

class ModelResults(ABC):
    """Base class for results returned by a Model.

    Holds a reference to the parent Model instance that generated it in order to
    facilitate permutation testing, and provides member functions for:
      a) FDR correction
      b) Thresholded cluster correction via design matrix permutations
      c) TFCE-based cluster correction
    """

    def __init__(
        self,
        model: "Model",
        uncorr_pvals: np.ndarray,
        **kwargs,
    ):
        self.model = model
        self.uncorr_pvals = uncorr_pvals
        self.fdr_corr_pvals = None
        self.clus_corr_pvals = None
        self.tfce_corr_pvals = None

    @property
    def stat_names(self) -> list[str]:
        if isinstance(self.model.contrasts_img, nib.Cifti2Image):
            num_stats = self.uncorr_pvals.shape[0]
        else:
            num_stats = self.uncorr_pvals.shape[-1] if self.uncorr_pvals.ndim >= 4 else 1

        if len(self.model.value_names) == num_stats:
            return self.model.value_names
        elif num_stats == 1:
            return ["stat"]
        else:
            return [f"stat_{i}" for i in range(num_stats)]

    # ---- Abstract methods -------------------------------------------------

    @abstractmethod
    def save(self, outdir: Path | str | None = None) -> None:
        """Save primary uncorrected results to disk."""
        ...

    @abstractmethod
    def _get_stat_map(self) -> np.ndarray:
        """Return primary test-statistic map (tstats, fstats, etc.)."""
        ...

    # ---- a) FDR correction ------------------------------------------------

    def fdr_correct(self, alpha: float | list[float] | None = None) -> np.ndarray:
        """Perform FDR correction on uncorrected p-values.

        Parameters
        ----------
        alpha : float or list[float], optional
            Significance threshold(s). Defaults to config.alphas.

        Returns
        -------
        np.ndarray
            FDR-corrected p-value array.
        """
        alphas = [alpha] if isinstance(alpha, float) else (alpha if alpha is not None else config.alphas)
        if isinstance(self.model.contrasts_img, nib.Cifti2Image):
            self.fdr_corr_pvals = self._fdr_correct_cifti(alphas)
            if hasattr(self.model, "dlabel_paths") and self.model.dlabel_paths is not None:
                for dlabel_path in self.model.dlabel_paths:
                    self._fdr_correct_cifti(alphas, dlabel_path=dlabel_path)
        elif isinstance(self.model.contrasts_img, nib.Nifti1Image):
            self.fdr_corr_pvals = self._fdr_correct_nifti(alphas)
        return self.fdr_corr_pvals

    def _fdr_correct_cifti(self, alphas: list[float], dlabel_path: str | None = None) -> np.ndarray:
        value_names = self.stat_names
        num_vals = self.uncorr_pvals.shape[0]
        fdr_corr_pvals = np.empty(
            (num_vals * len(alphas), self.uncorr_pvals.shape[1]),
            dtype=np.float32,
        )
        if dlabel_path is not None:
            logger.info(f"Running within-dlabel FDR correction with areas defined by {dlabel_path}")
            dlabel_img = nib.load(dlabel_path)
            common_structs = cifti_compatible_structures((self.model.contrasts_img.header, dlabel_img.header))
            if len(common_structs) == 0:
                raise ValueError(f"No compatible CIFTI structures between image and {dlabel_path}")
            dlabel_fdata = dlabel_img.get_fdata()
            unique_labels = np.unique(dlabel_fdata)
            for alpha_idx, alpha in enumerate(alphas):
                for value_idx in range(num_vals):
                    for label in unique_labels:
                        label_indices = np.argwhere(dlabel_fdata == label)
                        pval_vec = self.uncorr_pvals[value_idx, label_indices].copy()
                        pval_vec[np.isnan(pval_vec)] = 1
                        fdr_corr_pvals[value_idx * len(alphas) + alpha_idx, label_indices] = fdr_correct(
                            pval_vec, alpha=alpha
                        )
            fdr_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(
                fdr_corr_pvals,
                (
                    nib.cifti2.cifti2_axes.ScalarAxis(
                        [
                            f"{valname}_{alpha:.4f}"
                            for valname, alpha in itertools.product(value_names, alphas)
                        ]
                    ),
                    self.model.contrasts_img.header.get_axis(1),
                ),
            )
            dlabel_id = self.model.dlabel_path_to_id.get(dlabel_path, "dlabel") if hasattr(self.model, "dlabel_path_to_id") and self.model.dlabel_path_to_id else "dlabel"
            p = (
                self.model.model_outdir
                / f"{sanitize_filename(self.model.model_spec.model_name)}_fdr_corr_{dlabel_id}{self.model.out_suffix}"
            )
            nib.save(fdr_corr_pvals_cifti, p)
            logger.info(f"Saved {p!s}")
        else:
            logger.info("Running FDR correction")
            for alpha_idx, alpha in enumerate(alphas):
                for value_idx in range(num_vals):
                    pval_vec = self.uncorr_pvals[value_idx, :].copy()
                    pval_vec[np.isnan(pval_vec)] = 1
                    fdr_corr_pvals[value_idx * len(alphas) + alpha_idx, :] = fdr_correct(
                        pval_vec, alpha=alpha
                    )

            fdr_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(
                fdr_corr_pvals,
                (
                    nib.cifti2.cifti2_axes.ScalarAxis(
                        [
                            f"{valname}_{alpha:.4f}"
                            for valname, alpha in itertools.product(value_names, alphas)
                        ]
                    ),
                    self.model.contrasts_img.header.get_axis(1),
                ),
            )
            p = (
                self.model.model_outdir
                / f"{sanitize_filename(self.model.model_spec.model_name)}_fdr_corr{self.model.out_suffix}"
            )
            nib.save(fdr_corr_pvals_cifti, p)
            logger.info(f"Saved {p!s}")
        return fdr_corr_pvals

    def _fdr_correct_nifti(self, alphas: list[float]) -> np.ndarray:
        value_names = self.stat_names
        out_pvals = np.empty_like(self.uncorr_pvals)
        for alpha in alphas:
            for value_idx, value_name in enumerate(value_names):
                orig_shape = self.uncorr_pvals[..., value_idx].shape
                flattened_pvals = self.uncorr_pvals[..., value_idx].flatten()
                fdr_corr = fdr_correct(flattened_pvals, alpha=alpha).reshape(orig_shape)
                out_pvals[..., value_idx] = fdr_corr
                fdr_corr_img = nib.Nifti1Image(
                    fdr_corr, self.model.affine, header=self.model.contrasts_img.header
                )
                p = (
                    self.model.model_outdir
                    / f"{sanitize_filename(self.model.model_spec.model_name)}_beta-{value_name}_fdr_corr_{alpha:.4f}.nii.gz"
                )
                nib.save(fdr_corr_img, p)
                logger.info(f"Saved {p!s}")
        return out_pvals

    # ---- b) Thresholded cluster correction ---------------------------------

    def cluster_correct(
        self,
        n_perms: int | None = None,
        alpha: float | list[float] | None = None,
        method: str = "nn1",
        **kwargs,
    ) -> np.ndarray:
        """Thresholded cluster correction via permuting the design matrix N times.

        Parameters
        ----------
        n_perms : int, optional
            Number of permutations. Defaults to config.perms or 1000.
        alpha : float or list[float], optional
            Cluster-forming threshold(s). Defaults to config.alphas.
        method : str, default "nn1"
            Volumetric clustering neighborhood method: "nn1" (faces),
            "nn2" (faces + edges), or "nn3" (faces + edges + corners).

        Returns
        -------
        np.ndarray
            Cluster-corrected p-value array.
        """
        method_clean = method.lower()
        if method_clean not in ("nn1", "nn2", "nn3"):
            raise ValueError(f"method must be one of 'nn1', 'nn2', or 'nn3', received '{method}'")
        strategy = method_clean.upper()

        perms = n_perms if n_perms is not None else getattr(config, "perms", 1000)
        if perms <= 0:
            perms = kwargs.get("perms", 1000)
        alphas = [alpha] if isinstance(alpha, float) else (alpha if alpha is not None else config.alphas)

        biggest_l_surf = defaultdict(list)
        biggest_r_surf = defaultdict(list)
        biggest_vol = defaultdict(list)

        for perm in range(perms):
            logger.info(f"Running permutation {perm + 1} of {perms} for {self.model.model_spec.model_name}")
            permuted_dm = self.model.get_permuted_design_matrix()
            perm_pvals = self.model.fit_permuted(permuted_dm)
            self._accumulate_cluster_sizes(
                perm_pvals, alphas, strategy, biggest_l_surf, biggest_r_surf, biggest_vol
            )

        if isinstance(self.model.contrasts_img, nib.Cifti2Image):
            self.clus_corr_pvals = self._cluster_correct_cifti(
                alphas, biggest_l_surf, biggest_r_surf, biggest_vol, strategy
            )
        elif isinstance(self.model.contrasts_img, nib.Nifti1Image):
            self.clus_corr_pvals = self._cluster_correct_nifti(
                alphas, biggest_vol, strategy
            )
        return self.clus_corr_pvals

    def _accumulate_cluster_sizes(
        self,
        pvals: np.ndarray,
        alphas: list[float],
        strategy: str,
        biggest_l_surf: dict,
        biggest_r_surf: dict,
        biggest_vol: dict,
    ):
        if isinstance(self.model.contrasts_img, nib.Cifti2Image):
            if hasattr(self.model, "l_numverts"):
                l_pvals, r_pvals = extract_hemi_values(
                    pvals, self.model.contrasts_img.header, self.model.l_numverts, self.model.r_numverts
                )
                for alpha in alphas:
                    biggest_l_surf[alpha].extend(
                        get_biggest_surface_clusters(
                            l_pvals, alpha, self.model.l_neigh, self.model.l_area
                        )
                    )
                    biggest_r_surf[alpha].extend(
                        get_biggest_surface_clusters(
                            r_pvals, alpha, self.model.r_neigh, self.model.r_area
                        )
                    )
            # Subcortical volume inside CIFTI
            try:
                pval_volume = get_volume_array_from_cifti_array(pvals, self.model.contrasts_img.header)
                for alpha in alphas:
                    threshold_mask = (pval_volume < alpha)
                    biggest_vol[alpha].extend([
                        get_biggest_voxel_cluster_sizes(
                            get_voxel_clusters(threshold_mask[..., vol], strategy=strategy)
                        )
                        for vol in range(threshold_mask.shape[-1])
                    ])
            except (TypeError, AttributeError):
                pass
        elif isinstance(self.model.contrasts_img, nib.Nifti1Image):
            for alpha in alphas:
                threshold_mask = (pvals < alpha)
                biggest_vol[alpha].extend([
                    get_biggest_voxel_cluster_sizes(
                        get_voxel_clusters(threshold_mask[..., vol], strategy=strategy)
                    )
                    for vol in range(threshold_mask.shape[-1])
                ])

    def _cluster_correct_cifti(
        self,
        alphas: list[float],
        biggest_l_surf: dict,
        biggest_r_surf: dict,
        biggest_vol: dict,
        strategy: str,
    ) -> np.ndarray:
        value_names = self.stat_names
        separate_hemi = getattr(self.model, "separate_null_by_hemisphere", False)
        separate_param = getattr(self.model, "separate_null_by_parameter", False)

        l_pvals, r_pvals = extract_hemi_values(
            self.uncorr_pvals, self.model.contrasts_img.header, self.model.l_numverts, self.model.r_numverts
        )
        l_clus_corr = np.ones(
            (len(value_names) * len(alphas), self.model.l_numverts),
            dtype=np.float32,
        )
        r_clus_corr = np.ones(
            (len(value_names) * len(alphas), self.model.r_numverts),
            dtype=np.float32,
        )
        full_clus_corr = np.full(
            (len(value_names) * len(alphas), self.uncorr_pvals.shape[1]),
            np.nan,
            dtype=np.float32,
        )

        has_volume = False
        try:
            volume = get_volume_array_from_cifti_array(self.uncorr_pvals, self.model.contrasts_img.header)
            axis1 = self.model.contrasts_img.header.get_axis(1)
            volume_voxel_indices = axis1.voxel[(axis1.voxel != -1).all(axis=1)]
            volume_cifti_indices = np.argwhere((axis1.voxel != -1).all(axis=1)).flatten()
            has_volume = len(volume_cifti_indices) > 0
        except Exception:
            has_volume = False

        for alpha_idx, alpha in enumerate(alphas):
            for value_idx, value_name in enumerate(value_names):
                # Left hemisphere
                l_mask = np.isfinite(l_pvals[value_idx, :]) & (l_pvals[value_idx, :] < alpha)
                cluster_null_dist = (
                    biggest_l_surf[alpha] + biggest_r_surf[alpha]
                    if not separate_hemi
                    else biggest_l_surf[alpha]
                )
                if separate_param and len(cluster_null_dist) >= len(value_names):
                    cluster_null_dist = cluster_null_dist[value_idx::len(value_names)]
                save_null_histogram(
                    cluster_null_dist,
                    f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, LH",
                    self.model.model_outdir / f"null_lh_alpha{alpha:.4f}_cond{value_name}.png",
                )
                for cluster in get_cluster_index_groups(l_mask, self.model.l_neigh):
                    cluster_size = (
                        len(cluster)
                        if self.model.l_area is None
                        else np.sum(self.model.l_area[cluster])
                    )
                    cluster_size = np.int64(cluster_size)
                    sizes_larger = np.float32(np.sum(cluster_null_dist >= cluster_size))
                    l_clus_corr[value_idx * len(alphas) + alpha_idx, cluster] = (
                        sizes_larger / (len(cluster_null_dist) + 1)
                    )

                # Right hemisphere
                r_mask = np.isfinite(r_pvals[value_idx, :]) & (r_pvals[value_idx, :] < alpha)
                cluster_null_dist = (
                    biggest_l_surf[alpha] + biggest_r_surf[alpha]
                    if not separate_hemi
                    else biggest_r_surf[alpha]
                )
                if separate_param and len(cluster_null_dist) >= len(value_names):
                    cluster_null_dist = cluster_null_dist[value_idx::len(value_names)]
                save_null_histogram(
                    cluster_null_dist,
                    f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, RH",
                    self.model.model_outdir / f"null_rh_alpha{alpha:.4f}_cond{value_name}.png",
                )
                for cluster in get_cluster_index_groups(r_mask, self.model.r_neigh):
                    cluster_size = (
                        len(cluster)
                        if self.model.r_area is None
                        else np.sum(self.model.r_area[cluster])
                    )
                    cluster_size = np.int64(cluster_size)
                    sizes_larger = np.float32(np.sum(cluster_null_dist >= cluster_size))
                    r_clus_corr[value_idx * len(alphas) + alpha_idx, cluster] = (
                        sizes_larger / (len(cluster_null_dist) + 1)
                    )

                # Subcortical volume
                if has_volume:
                    volume_mask = volume[..., value_idx] < alpha
                    clus_corr_volume = np.full_like(volume_mask, np.nan, dtype=np.float32)
                    vol_null_dist = (
                        biggest_vol[alpha]
                        if not separate_param
                        else biggest_vol[alpha][value_idx::len(value_names)]
                    )
                    save_null_histogram(
                        vol_null_dist,
                        f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, subcortical",
                        self.model.model_outdir / f"null_subcort_alpha{alpha:.4f}_cond{value_name}.png",
                    )
                    for cluster in get_voxel_clusters(volume_mask, strategy=strategy):
                        cluster_size = np.int64(cluster.shape[0])
                        sizes_larger = np.sum(vol_null_dist >= cluster_size)
                        clus_p = sizes_larger / (len(vol_null_dist) + 1)
                        clus_corr_volume[tuple(cluster.T)] = clus_p
                    full_clus_corr[
                        value_idx * len(alphas) + alpha_idx,
                        tuple(volume_cifti_indices.T),
                    ] = clus_corr_volume[tuple(volume_voxel_indices.T)].T

        # Transfer surface clusters to full CIFTI
        for name, slc, bmodel in self.model.contrasts_img.header.get_axis(1).iter_structures():
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
                        for valname, alpha in itertools.product(value_names, alphas)
                    ]
                ),
                self.model.contrasts_img.header.get_axis(1),
            ),
        )
        p = (
            self.model.model_outdir
            / f"{sanitize_filename(self.model.model_spec.model_name)}_clus_corr{self.model.out_suffix}"
        )
        nib.save(clus_corr_pvals_cifti, p)
        logger.info(f"Saved {p!s}")
        return full_clus_corr

    def _cluster_correct_nifti(
        self,
        alphas: list[float],
        biggest_vol: dict,
        strategy: str,
    ) -> np.ndarray:
        value_names = self.stat_names
        separate_param = getattr(self.model, "separate_null_by_parameter", False)
        clus_corr_full = np.full_like(self.uncorr_pvals, np.nan, dtype=np.float32)

        for alpha in alphas:
            for value_idx, value_name in enumerate(value_names):
                clus_corr = np.full_like(self.uncorr_pvals[..., value_idx], np.nan, dtype=np.float32)
                vol_null_dist = (
                    biggest_vol[alpha]
                    if not separate_param
                    else biggest_vol[alpha][value_idx::len(value_names)]
                )
                save_null_histogram(
                    vol_null_dist,
                    f"Null distribution of cluster sizes at alpha {alpha:.4f}, condition {value_name}, voxels",
                    self.model.model_outdir / f"null_subcort_alpha{alpha:.4f}_cond{value_name}.png",
                )
                mask = self.uncorr_pvals[..., value_idx] < alpha
                for cluster in get_voxel_clusters(mask, strategy=strategy):
                    cluster_size = np.int64(cluster.shape[0])
                    sizes_larger = np.sum(vol_null_dist >= cluster_size)
                    clus_p = sizes_larger / (len(vol_null_dist) + 1)
                    clus_corr[tuple(cluster.T)] = clus_p
                clus_corr_full[..., value_idx] = clus_corr
                clus_corr_img = nib.Nifti1Image(
                    clus_corr, self.model.affine, header=self.model.contrasts_img.header
                )
                p = (
                    self.model.model_outdir
                    / f"{sanitize_filename(self.model.model_spec.model_name)}_beta-{value_name}_clus_corr_{alpha:.4f}.nii.gz"
                )
                nib.save(clus_corr_img, p)
                logger.info(f"Saved {p!s}")
        return clus_corr_full

    # ---- c) TFCE-based cluster correction -----------------------------------

    def tfce_correct(
        self,
        n_perms: int | None = None,
        alpha: float = 0.05,
        method: str = "nn1",
        H: float = 2.0,
        E: float = 0.5,
        dh: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Threshold-Free Cluster Enhancement (TFCE) permutation correction.

        Parameters
        ----------
        n_perms : int, optional
            Number of permutations. Defaults to config.perms or 1000.
        alpha : float, default 0.05
            Significance threshold.
        method : str, default "nn1"
            Volumetric clustering neighborhood: "nn1", "nn2", or "nn3".
        H : float, default 2.0
            Height exponent (Smith & Nichols 2009).
        E : float, default 0.5
            Extent exponent (Smith & Nichols 2009).
        dh : float, optional
            Threshold step size. If None, defaults to max(|stat|) / 100.

        Returns
        -------
        np.ndarray
            TFCE-corrected p-value map.
        """
        method_clean = method.lower()
        if method_clean not in ("nn1", "nn2", "nn3"):
            raise ValueError(f"method must be one of 'nn1', 'nn2', or 'nn3', received '{method}'")
        strategy = method_clean.upper()

        perms = n_perms if n_perms is not None else getattr(config, "perms", 1000)
        if perms <= 0:
            perms = kwargs.get("perms", 1000)

        stat_map = self._get_stat_map()
        obs_tfce = self._compute_tfce(stat_map, strategy=strategy, H=H, E=E, dh=dh)

        null_max = np.empty(perms, dtype=np.float64)
        for perm in range(perms):
            logger.info(f"Running TFCE permutation {perm + 1} of {perms}")
            permuted_dm = self.model.get_permuted_design_matrix()
            perm_stat = self.model.fit_permuted_stat(permuted_dm)
            perm_tfce = self._compute_tfce(perm_stat, strategy=strategy, H=H, E=E, dh=dh)
            null_max[perm] = np.nanmax(np.abs(perm_tfce))

        # Compute TFCE corrected p-values
        tfce_pvals = np.ones_like(obs_tfce, dtype=np.float32)
        for idx in np.ndindex(obs_tfce.shape):
            val = np.abs(obs_tfce[idx])
            if np.isfinite(val) and val > 0:
                tfce_pvals[idx] = (np.sum(null_max >= val) + 1) / (perms + 1)

        self.tfce_corr_pvals = tfce_pvals
        self._save_tfce(tfce_pvals, alpha)
        return tfce_pvals

    def _compute_tfce(
        self,
        stat_map: np.ndarray,
        strategy: str = "NN1",
        H: float = 2.0,
        E: float = 0.5,
        dh: float | None = None,
    ) -> np.ndarray:
        """Compute TFCE score map for a given statistic map."""
        abs_stat = np.abs(stat_map)
        max_val = np.nanmax(abs_stat)
        if max_val == 0 or np.isnan(max_val):
            return np.zeros_like(stat_map, dtype=np.float32)
        if dh is None:
            dh = float(max_val / 100.0)
        if dh <= 0:
            dh = 0.1

        thresholds = np.arange(dh, max_val + dh, dh)
        tfce = np.zeros_like(stat_map, dtype=np.float32)

        if isinstance(self.model.contrasts_img, nib.Cifti2Image) and hasattr(self.model, "l_neigh"):
            # Surface + subcortical TFCE
            for h in thresholds:
                suprathresh = abs_stat >= h
                for row_idx in range(stat_map.shape[0]):
                    row_mask = suprathresh[row_idx, :]
                    l_vals = np.zeros(self.model.l_numverts, dtype=bool)
                    r_vals = np.zeros(self.model.r_numverts, dtype=bool)
                    for name, slc, bmodel in self.model.contrasts_img.header.get_axis(1).iter_structures():
                        vidx = bmodel.vertex.astype(np.int64)
                        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
                            l_vals[vidx] = row_mask[slc]
                        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
                            r_vals[vidx] = row_mask[slc]
                    for cluster in get_cluster_index_groups(l_vals, self.model.l_neigh):
                        extent = len(cluster) if self.model.l_area is None else np.sum(self.model.l_area[cluster])
                        score = float((h ** H) * (extent ** E) * dh)
                        for name2, slc2, bmodel2 in self.model.contrasts_img.header.get_axis(1).iter_structures():
                            if name2 == "CIFTI_STRUCTURE_CORTEX_LEFT":
                                vidx2 = bmodel2.vertex.astype(np.int64)
                                cifti_mask = np.isin(vidx2, cluster)
                                tfce[row_idx, slc2][cifti_mask] += score
                                break
                    for cluster in get_cluster_index_groups(r_vals, self.model.r_neigh):
                        extent = len(cluster) if self.model.r_area is None else np.sum(self.model.r_area[cluster])
                        score = float((h ** H) * (extent ** E) * dh)
                        for name2, slc2, bmodel2 in self.model.contrasts_img.header.get_axis(1).iter_structures():
                            if name2 == "CIFTI_STRUCTURE_CORTEX_RIGHT":
                                vidx2 = bmodel2.vertex.astype(np.int64)
                                cifti_mask = np.isin(vidx2, cluster)
                                tfce[row_idx, slc2][cifti_mask] += score
                                break
        elif isinstance(self.model.contrasts_img, nib.Nifti1Image):
            for h in thresholds:
                for vol_idx in range(stat_map.shape[-1]):
                    vol_stat = abs_stat[..., vol_idx]
                    suprathresh = vol_stat >= h
                    for cluster in get_voxel_clusters(suprathresh, strategy=strategy):
                        extent = cluster.shape[0]
                        score = float((h ** H) * (extent ** E) * dh)
                        tfce[tuple(cluster.T) + (vol_idx,)] += score

        return tfce

    def _save_tfce(self, tfce_pvals: np.ndarray, alpha: float):
        value_names = self.stat_names
        if isinstance(self.model.contrasts_img, nib.Cifti2Image):
            img = nib.cifti2.cifti2.Cifti2Image(
                tfce_pvals,
                (
                    nib.cifti2.cifti2_axes.ScalarAxis(value_names),
                    self.model.contrasts_img.header.get_axis(1),
                ),
            )
            p = (
                self.model.model_outdir
                / f"{sanitize_filename(self.model.model_spec.model_name)}_tfce_corr_{alpha:.4f}{self.model.out_suffix}"
            )
            nib.save(img, p)
            logger.info(f"Saved {p!s}")
        elif isinstance(self.model.contrasts_img, nib.Nifti1Image):
            for idx, vn in enumerate(value_names):
                val = tfce_pvals[..., idx] if tfce_pvals.ndim >= 4 else tfce_pvals
                img = nib.Nifti1Image(
                    val, self.model.affine, header=self.model.contrasts_img.header
                )
                p = (
                    self.model.model_outdir
                    / f"{sanitize_filename(self.model.model_spec.model_name)}_beta-{vn}_tfce_corr_{alpha:.4f}.nii.gz"
                )
                nib.save(img, p)
                logger.info(f"Saved {p!s}")


class OLSModelResults(ModelResults):
    """Results container for OLSModel."""

    def __init__(
        self,
        model: "Model",
        uncorr_pvals: np.ndarray,
        tstats: np.ndarray,
        betas: np.ndarray,
        ses: np.ndarray,
    ):
        super().__init__(model, uncorr_pvals)
        self.tstats = tstats
        self.betas = betas
        self.ses = ses

    def _get_stat_map(self) -> np.ndarray:
        return self.tstats

    def save(self, outdir: Path | str | None = None) -> None:
        target_dir = Path(outdir) if outdir is not None else self.model.model_outdir
        if isinstance(self.model.contrasts_img, nib.Nifti1Image):
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pvals),
                ("betas", self.betas),
                ("ses", self.ses),
                ("tstats", self.tstats),
            ):
                if data is None:
                    continue
                for idx, value_name in enumerate(self.model.value_names):
                    img = nib.Nifti1Image(data[..., idx], self.model.affine, header=self.model.contrasts_img.header)
                    p = target_dir / f"{sanitize_filename(self.model.model_spec.model_name)}_beta-{value_name}_{datatype}.nii.gz"
                    nib.save(img, p)
                    logger.info(f"Saved {p!s}")
        elif isinstance(self.model.contrasts_img, nib.Cifti2Image):
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
                        nib.cifti2.cifti2_axes.ScalarAxis(self.model.value_names),
                        self.model.contrasts_img.header.get_axis(1),
                    ),
                )
                p = target_dir / f"{sanitize_filename(self.model.model_spec.model_name)}_{datatype}{self.model.out_suffix}"
                nib.save(img, p)
                logger.info(f"Saved {p!s}")


class TwoWayAnovaModelResults(ModelResults):
    """Results container for TwoWayAnovaModel."""

    def __init__(
        self,
        model: "Model",
        uncorr_pvals: np.ndarray,
        fstats: np.ndarray,
        epsilons: np.ndarray,
        pval_sphericity_corr: np.ndarray,
    ):
        super().__init__(model, uncorr_pvals)
        self.fstats = fstats
        self.epsilons = epsilons
        self.pval_sphericity_corr = pval_sphericity_corr

    def _get_stat_map(self) -> np.ndarray:
        return self.fstats

    def save(self, outdir: Path | str | None = None) -> None:
        target_dir = Path(outdir) if outdir is not None else self.model.model_outdir
        if isinstance(self.model.contrasts_img, nib.Nifti1Image):
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pvals),
                ("fstat", self.fstats),
                ("epsilons", self.epsilons),
                ("pval_sphericity_corr", self.pval_sphericity_corr),
            ):
                if data is None:
                    continue
                img = nib.Nifti1Image(data, self.model.affine, header=self.model.contrasts_img.header)
                p = target_dir / f"{sanitize_filename(self.model.model_spec.model_name)}_{datatype}.nii.gz"
                nib.save(img, p)
                logger.info(f"Saved {p!s}")
        elif isinstance(self.model.contrasts_img, nib.Cifti2Image):
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pvals),
                ("fstat", self.fstats),
                ("epsilons", self.epsilons),
                ("pval_sphericity_corr", self.pval_sphericity_corr),
            ):
                if data is None:
                    continue
                img = nib.cifti2.cifti2.Cifti2Image(
                    data,
                    (
                        nib.cifti2.cifti2_axes.ScalarAxis(name=[datatype]),
                        self.model.contrasts_img.header.get_axis(1),
                    ),
                )
                p = target_dir / f"{sanitize_filename(self.model.model_spec.model_name)}_{datatype}{self.model.out_suffix}"
                nib.save(img, p)
                logger.info(f"Saved {p!s}")


class RMAnovaModelResults(ModelResults):
    """Results container for RMAnovaModel."""

    def __init__(
        self,
        model: "Model",
        uncorr_pvals: np.ndarray,
        fstats: np.ndarray,
        epsilons: np.ndarray,
        pval_sphericity_corr: np.ndarray,
    ):
        super().__init__(model, uncorr_pvals)
        self.fstats = fstats
        self.epsilons = epsilons
        self.pval_sphericity_corr = pval_sphericity_corr

    def _get_stat_map(self) -> np.ndarray:
        return self.fstats

    def save(self, outdir: Path | str | None = None) -> None:
        target_dir = Path(outdir) if outdir is not None else self.model.model_outdir
        if isinstance(self.model.contrasts_img, nib.Nifti1Image):
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pvals),
                ("fstat", self.fstats),
                ("epsilons", self.epsilons),
                ("pval_sphericity_corr", self.pval_sphericity_corr),
            ):
                if data is None:
                    continue
                img = nib.Nifti1Image(data, self.model.affine, header=self.model.contrasts_img.header)
                p = target_dir / f"{sanitize_filename(self.model.model_spec.model_name)}_{datatype}.nii.gz"
                nib.save(img, p)
                logger.info(f"Saved {p!s}")
        elif isinstance(self.model.contrasts_img, nib.Cifti2Image):
            for datatype, data in (
                ("uncorr_pvals", self.uncorr_pvals),
                ("fstat", self.fstats),
                ("epsilons", self.epsilons),
                ("pval_sphericity_corr", self.pval_sphericity_corr),
            ):
                if data is None:
                    continue
                img = nib.cifti2.cifti2.Cifti2Image(
                    data,
                    (
                        nib.cifti2.cifti2_axes.ScalarAxis(name=[datatype]),
                        self.model.contrasts_img.header.get_axis(1),
                    ),
                )
                p = target_dir / f"{sanitize_filename(self.model.model_spec.model_name)}_{datatype}{self.model.out_suffix}"
                nib.save(img, p)
                logger.info(f"Saved {p!s}")


# Alias for backward compatibility
AnovaModelResults = TwoWayAnovaModelResults


# ---------------------------------------------------------------------------
# Model Base and Subclasses
# ---------------------------------------------------------------------------

class Model(ABC):
    """Common Abstract Base Class for group-level statistical models.

    The constructor's only responsibility is to build out the necessary stacked
    first-level image and design matrices for the model, resolve geometry, and
    prepare for fitting.

    Member function fit() fits the model and returns a ModelResults instance.
    """

    model_spec: ModelSpec
    firstlevel_contrasts_df: pd.DataFrame
    subject_variables_df: pd.DataFrame
    contrasts_img: nib.Cifti2Image | nib.Nifti1Image
    fdata: np.ndarray
    design_df: pd.DataFrame
    value_names: list[str]
    model_outdir: Path
    out_suffix: str

    @abstractmethod
    def __init__(
        self,
        model_spec: ModelSpec,
        firstlevel_contrasts_df: pd.DataFrame,
        subject_variables_df: pd.DataFrame,
    ):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> ModelResults:
        """Fit the model and return a ModelResults instance."""
        pass

    @abstractmethod
    def get_permuted_design_matrix(self) -> Any:
        """Return a permuted design matrix (or permuted condition labels)."""
        pass

    @abstractmethod
    def fit_permuted(self, permuted_design_matrix: Any) -> np.ndarray:
        """Fit the model with a permuted design matrix and return p-values."""
        pass

    @abstractmethod
    def fit_permuted_stat(self, permuted_design_matrix: Any) -> np.ndarray:
        """Fit the model with a permuted design matrix and return test statistics."""
        pass

    # ---- Common geometry setup --------------------------------------------

    def _setup_outdir(self):
        outdir_base = config.outdir_path if config.outdir_path is not None else Path("./output")
        self.model_outdir = outdir_base / sanitize_filename(self.model_spec.model_name)
        if not self.model_outdir.is_dir():
            self.model_outdir.mkdir(parents=True, exist_ok=True)

    def _setup_image_geometry(
        self,
        l_area_path: Path | None = None,
        r_area_path: Path | None = None,
    ):
        self.volume_mask = None
        self.l_area = None
        self.r_area = None

        if isinstance(self.contrasts_img, nib.Nifti1Image):
            self.out_suffix = ".nii.gz"
            self.affine = self.contrasts_img.affine
            voxel_sizes = self.contrasts_img.header.get_zooms()[:3]
            try:
                for k, v in tflow.get_metadata(self.model_spec.space)["res"].items():
                    if np.allclose(self.contrasts_img.header.get_zooms()[:3], v["zooms"]):
                        self.volume_mask = nib.load(
                            tflow.get(
                                self.model_spec.space,
                                resolution=self.contrasts_img.header.get_zooms()[0],
                                desc="brain",
                                suffix="mask",
                            )
                        )
                        break
                if self.volume_mask is None:
                    self.volume_mask = resample_img(
                        nib.load(
                            tflow.get(
                                self.model_spec.space,
                                resolution=np.floor(self.contrasts_img.header.get_zooms()[0]),
                                desc="brain",
                                suffix="mask",
                            )
                        ),
                        target_affine=self.affine,
                        target_shape=self.fdata.shape[:3],
                        interpolation="nearest",
                    )
            except Exception as e:
                logger.warning(f"Could not load templateflow mask for space {self.model_spec.space}: {e}")

        elif isinstance(self.contrasts_img, nib.Cifti2Image) and hasattr(self.contrasts_img.header.get_axis(1), "vertex"):
            self.out_suffix = ".dscalar.nii"
            try:
                l_surf_img, r_surf_img = get_template_midthicknesses_from_cifti_header(
                    self.contrasts_img.header, self.model_spec.space
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
            except Exception as e:
                logger.warning(f"Could not load surface midthickness for space {self.model_spec.space}: {e}")
        elif isinstance(self.contrasts_img, nib.Cifti2Image) and isinstance(self.contrasts_img.header.get_axis(1), nib.cifti2.cifti2_axes.ParcelsAxis):
            self.out_suffix = ".pscalar.nii"


class OLSModel(Model):
    """Ordinary Least Squares group-level GLM.

    Constructor responsibility:
      Build out the necessary stacked first-level image and design matrix from
      the provided ModelSpec, firstlevel_contrasts_df, and subject_variables_df.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        firstlevel_contrasts_df: pd.DataFrame,
        subject_variables_df: pd.DataFrame,
        **kwargs,
    ):
        self.model_spec = model_spec
        self.firstlevel_contrasts_df = firstlevel_contrasts_df
        self.subject_variables_df = subject_variables_df
        self._setup_outdir()

        # Build stacked first-level image
        get_img_fn = get_assumed_contrasts_img
        if config.joblib_memory is not None:
            get_img_fn = config.joblib_memory.cache(get_assumed_contrasts_img)
        self.contrasts_img = get_img_fn(self.model_spec, self.firstlevel_contrasts_df)
        self.fdata = self.contrasts_img.get_fdata()

        # Build design matrix
        self.design_df = get_ols_design_df(self.model_spec, self.subject_variables_df)
        self.value_names = list(self.design_df.columns)
        self.design_df.to_csv(self.model_outdir / "design_matrix.tsv", sep="\t", index=False)

        # Setup geometry
        l_area_path = kwargs.get("l_area_path", None)
        r_area_path = kwargs.get("r_area_path", None)
        self._setup_image_geometry(l_area_path=l_area_path, r_area_path=r_area_path)

        # Options for null separation & dlabel
        self.separate_null_by_hemisphere = kwargs.get(
            "separate_null_by_hemisphere", getattr(config, "separate_null_by_hemisphere", False)
        )
        self.separate_null_by_parameter = kwargs.get(
            "separate_null_by_parameter", getattr(config, "separate_null_by_parameter", False)
        )

        dlabel_paths = kwargs.get("dlabel_paths", getattr(config, "dlabel_paths", None))
        if dlabel_paths is not None:
            self.dlabel_paths = [str(p.resolve()) for p in dlabel_paths]
            self.dlabel_path_to_id = OrderedDict()
            for i, path in enumerate(self.dlabel_paths):
                self.dlabel_path_to_id[path] = f"network{i + 1}"
            with open(self.model_outdir / "network_mappings.json", "w") as f:
                json.dump(self.dlabel_path_to_id, f, indent=4)
                logger.info(f"Wrote {(self.model_outdir / 'network_mappings.json').resolve()!s}")

    def fit(self, *args, **kwargs) -> OLSModelResults:
        """Fit OLS GLM and return an OLSModelResults instance."""
        xp = get_xp()
        design_matrix_arr = self.design_df.to_numpy()
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (design_matrix_arr.shape[1], self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], design_matrix_arr.shape[1])
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))
        else:
            raise ValueError(f"Cannot fit GLM for image type: {type(self.contrasts_img)}")

        n, p = design_matrix_arr.shape
        X = xp.asarray(design_matrix_arr)
        Y = xp.asarray(self.fdata.reshape(glm_input_shape))

        beta, ssr, rank, s = xp.linalg.lstsq(X, Y, rcond=None)
        if ssr.size == 0 or ssr.shape[0] == 0:
            ssr = xp.sum((Y - X @ beta) ** 2, axis=0)

        sigma_sq = ssr / (n - p)
        v_cov_diag = xp.repeat(
            xp.diag(xp.linalg.inv(X.T @ X))[:, xp.newaxis],
            sigma_sq.shape[0],
            axis=1,
        )
        se = xp.sqrt(v_cov_diag * sigma_sq)
        tstat = beta / se

        tstat_cpu = to_cpu(tstat)
        pval = 2 * (1 - stats.t.cdf(np.abs(tstat_cpu), df=n - p))

        betas = to_cpu(beta).reshape(img_shape)
        ses = to_cpu(se).reshape(img_shape)
        tstats = tstat_cpu.reshape(img_shape)
        pvals = pval.reshape(img_shape)

        return OLSModelResults(
            model=self,
            uncorr_pvals=pvals,
            tstats=tstats,
            betas=betas,
            ses=ses,
        )

    def get_permuted_design_matrix(self) -> pd.DataFrame:
        """Return a copy of self.design_df with every non-intercept column shuffled."""
        permuted = self.design_df.copy()
        for column in permuted.columns:
            if column != "intercept":
                permuted[column] = permuted[column].sample(frac=1).array
        return permuted

    def fit_permuted(self, permuted_design_matrix: pd.DataFrame) -> np.ndarray:
        """Re-fit OLS with permuted design matrix and return p-values."""
        xp = get_xp()
        design_matrix_arr = permuted_design_matrix.to_numpy()
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (design_matrix_arr.shape[1], self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], design_matrix_arr.shape[1])
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))
        n, p = design_matrix_arr.shape
        X = xp.asarray(design_matrix_arr)
        Y = xp.asarray(self.fdata.reshape(glm_input_shape))

        beta, ssr, rank, s = xp.linalg.lstsq(X, Y, rcond=None)
        if ssr.size == 0 or ssr.shape[0] == 0:
            ssr = xp.sum((Y - X @ beta) ** 2, axis=0)

        sigma_sq = ssr / (n - p)
        v_cov_diag = xp.repeat(
            xp.diag(xp.linalg.inv(X.T @ X))[:, xp.newaxis],
            sigma_sq.shape[0],
            axis=1,
        )
        se = xp.sqrt(v_cov_diag * sigma_sq)
        tstat = beta / se
        tstat_cpu = to_cpu(tstat)
        pval = 2 * (1 - stats.t.cdf(np.abs(tstat_cpu), df=n - p))
        return pval.reshape(img_shape)

    def fit_permuted_stat(self, permuted_design_matrix: pd.DataFrame) -> np.ndarray:
        """Re-fit OLS with permuted design matrix and return t-statistics."""
        xp = get_xp()
        design_matrix_arr = permuted_design_matrix.to_numpy()
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (design_matrix_arr.shape[1], self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], design_matrix_arr.shape[1])
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))
        n, p = design_matrix_arr.shape
        X = xp.asarray(design_matrix_arr)
        Y = xp.asarray(self.fdata.reshape(glm_input_shape))

        beta, ssr, rank, s = xp.linalg.lstsq(X, Y, rcond=None)
        if ssr.size == 0 or ssr.shape[0] == 0:
            ssr = xp.sum((Y - X @ beta) ** 2, axis=0)

        sigma_sq = ssr / (n - p)
        v_cov_diag = xp.repeat(
            xp.diag(xp.linalg.inv(X.T @ X))[:, xp.newaxis],
            sigma_sq.shape[0],
            axis=1,
        )
        se = xp.sqrt(v_cov_diag * sigma_sq)
        tstat = beta / se
        return to_cpu(tstat).reshape(img_shape)


class TwoWayAnovaModel(Model):
    """Two-way repeated-measures ANOVA on FIR time-courses.

    Constructor responsibility:
      Build out the stacked FIR image and design matrix from ModelSpec,
      firstlevel_contrasts_df, and subject_variables_df.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        firstlevel_contrasts_df: pd.DataFrame,
        subject_variables_df: pd.DataFrame,
        **kwargs,
    ):
        self.model_spec = model_spec
        self.firstlevel_contrasts_df = firstlevel_contrasts_df
        self.subject_variables_df = subject_variables_df
        self._setup_outdir()

        self.num_frames = get_num_frames(self.model_spec, self.firstlevel_contrasts_df)
        self.design_df_with_sub, self.var_of_interest = get_twoway_anova_design_df(
            self.model_spec, self.subject_variables_df, self.num_frames
        )
        self.design_df = self.design_df_with_sub.drop(columns=["subject"])
        self.value_names = list(self.design_df.columns)
        self.design_df_with_sub.to_csv(self.model_outdir / "design_matrix.tsv", sep="\t", index=False)

        get_img_fn = get_anova_contrasts_img
        if config.joblib_memory is not None:
            get_img_fn = config.joblib_memory.cache(get_anova_contrasts_img)
        self.contrasts_img = get_img_fn(
            self.model_spec,
            self.firstlevel_contrasts_df,
            self.num_frames,
            subject_variables_df=self.subject_variables_df,
            var_of_interest=self.var_of_interest,
        )
        self.fdata = self.contrasts_img.get_fdata()

        self._setup_image_geometry()
        self._maybe_smooth_cifti()

        self.separate_null_by_hemisphere = kwargs.get(
            "separate_null_by_hemisphere", getattr(config, "separate_null_by_hemisphere", False)
        )
        self.separate_null_by_parameter = kwargs.get(
            "separate_null_by_parameter", getattr(config, "separate_null_by_parameter", False)
        )

    def _maybe_smooth_cifti(self):
        if isinstance(self.contrasts_img, nib.Cifti2Image) and hasattr(self.contrasts_img.header.get_axis(1), "vertex"):
            self.out_suffix = ".dscalar.nii"
            contrasts_img_path = self.model_outdir / "stacked_contrasts.dtseries.nii"
            nib.save(self.contrasts_img, contrasts_img_path)
            if config.fwhm > 0:
                l_surf_path, r_surf_path = get_template_midthickness_paths_from_cifti_header(
                    self.contrasts_img.header, self.model_spec.space
                )
                smoothed_contrasts_img_path = self.model_outdir / "smoothed_stacked_contrasts.dtseries.nii"
                subprocess.run(shlex.split(
                    "wb_command -cifti-smoothing "
                    f"{str(contrasts_img_path.resolve())} {config.fwhm} {config.fwhm} COLUMN {str(smoothed_contrasts_img_path.resolve())} -fwhm "
                    f"-left-surface {l_surf_path} -right-surface {r_surf_path}"
                ))
                self.contrasts_img = nib.load(smoothed_contrasts_img_path)
                self.fdata = self.contrasts_img.get_fdata()

    def fit(self, *args, **kwargs) -> TwoWayAnovaModelResults:
        """Fit TwoWayAnovaModel and return TwoWayAnovaModelResults."""
        xp = get_xp()
        design_matrix = self.design_df.reset_index(drop=True)
        design_matrix_no_int = design_matrix[[col for col in self.design_df if "interaction" not in col]]
        design_matrix_arr = design_matrix.to_numpy()
        design_matrix_arr_no_int = design_matrix_no_int.to_numpy()
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (1, self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], 1)
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))
        else:
            raise ValueError(f"Cannot fit GLM for image type: {type(self.contrasts_img)}")

        Y = xp.asarray(self.fdata.reshape(glm_input_shape))
        X_full = xp.asarray(design_matrix_arr)
        X_red = xp.asarray(design_matrix_arr_no_int)

        beta, _, _, _ = xp.linalg.lstsq(X_full, Y, rcond=None)
        beta_no_int, _, _, _ = xp.linalg.lstsq(X_red, Y, rcond=None)
        rss_full = xp.sum((Y - X_full @ beta) ** 2, axis=0)
        rss_reduced = xp.sum((Y - X_red @ beta_no_int) ** 2, axis=0)
        df_num = self.num_frames - 1
        df_denom = len(self.design_df) - design_matrix_arr.shape[1]
        fstat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_denom)

        # Greenhouse-Geisser sphericity correction
        res = (Y - (X_full @ beta)).reshape((-1, self.num_frames, beta.shape[-1]))
        n_subs = res.shape[0]
        k_frames = self.num_frames
        res_c = res - res.mean(axis=0, keepdims=True)
        S = xp.einsum("sik,sjk->ijk", res_c, res_c) / (n_subs - 1)
        s_diag = xp.diagonal(S, axis1=0, axis2=1)
        s_mean = xp.mean(S, axis=(0, 1))
        s_row_mean = xp.mean(S, axis=1)

        df_num_ = (k_frames ** 2) * ((xp.mean(s_diag, axis=1) - s_mean) ** 2)
        df_denom_ = (k_frames - 1) * (
            xp.sum(S ** 2, axis=(0, 1))
            - 2 * k_frames * xp.sum(s_row_mean ** 2, axis=0)
            + (k_frames ** 2) * (s_mean ** 2)
        )
        epsilons = xp.where(df_denom_ == 0, 1.0, xp.clip(df_num_ / df_denom_, 1.0 / (k_frames - 1), 1.0))
        epsilons = xp.nan_to_num(epsilons, nan=1.0 / (k_frames - 1))

        fstat_cpu = to_cpu(fstat)
        epsilons_cpu = to_cpu(epsilons)
        pval = stats.f.sf(fstat_cpu, df_num, df_denom)
        pval_sphericity_corr = stats.f.sf(fstat_cpu, df_num * epsilons_cpu, df_denom * epsilons_cpu)

        return TwoWayAnovaModelResults(
            model=self,
            uncorr_pvals=pval.reshape(img_shape),
            fstats=fstat_cpu.reshape(img_shape),
            epsilons=epsilons_cpu.reshape(img_shape),
            pval_sphericity_corr=pval_sphericity_corr.reshape(img_shape),
        )

    def get_permuted_design_matrix(self) -> pd.DataFrame:
        """Shuffle between-subject variable by whole-block, within-subject variables freely."""
        between_group_columns = [
            c for c in self.design_df_with_sub.columns
            if c != "subject" and (
                c.startswith(self.var_of_interest)
                or c == self.var_of_interest
                or re.match(r"frame_\d+", c)
                or "interaction" in c
            )
        ]
        between_groups = []
        for _, group_df in self.design_df_with_sub.groupby("subject", sort=False):
            between_groups.append(group_df[between_group_columns].sample(frac=1).reset_index(drop=True))
        random.shuffle(between_groups)
        df_ = self.design_df_with_sub.copy()
        df_[between_group_columns] = pd.concat(between_groups, ignore_index=True)
        return df_.drop(columns=["subject"])

    def fit_permuted(self, permuted_design_matrix: pd.DataFrame) -> np.ndarray:
        """Re-fit TwoWayAnova with permuted design matrix and return p-values."""
        xp = get_xp()
        design_matrix = permuted_design_matrix.reset_index(drop=True)
        design_matrix_no_int = design_matrix[[col for col in design_matrix.columns if "interaction" not in col]]
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (1, self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], 1)
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))

        Y = xp.asarray(self.fdata.reshape(glm_input_shape))
        X_full = xp.asarray(design_matrix.to_numpy())
        X_red = xp.asarray(design_matrix_no_int.to_numpy())

        beta, _, _, _ = xp.linalg.lstsq(X_full, Y, rcond=None)
        beta_no_int, _, _, _ = xp.linalg.lstsq(X_red, Y, rcond=None)
        rss_full = xp.sum((Y - X_full @ beta) ** 2, axis=0)
        rss_reduced = xp.sum((Y - X_red @ beta_no_int) ** 2, axis=0)
        df_num = self.num_frames - 1
        df_denom = len(design_matrix) - X_full.shape[1]
        fstat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_denom)
        fstat_cpu = to_cpu(fstat)
        pval = stats.f.sf(fstat_cpu, df_num, df_denom)
        return pval.reshape(img_shape)

    def fit_permuted_stat(self, permuted_design_matrix: pd.DataFrame) -> np.ndarray:
        """Re-fit TwoWayAnova with permuted design matrix and return F-stats."""
        xp = get_xp()
        design_matrix = permuted_design_matrix.reset_index(drop=True)
        design_matrix_no_int = design_matrix[[col for col in design_matrix.columns if "interaction" not in col]]
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (1, self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], 1)
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))

        Y = xp.asarray(self.fdata.reshape(glm_input_shape))
        X_full = xp.asarray(design_matrix.to_numpy())
        X_red = xp.asarray(design_matrix_no_int.to_numpy())

        beta, _, _, _ = xp.linalg.lstsq(X_full, Y, rcond=None)
        beta_no_int, _, _, _ = xp.linalg.lstsq(X_red, Y, rcond=None)
        rss_full = xp.sum((Y - X_full @ beta) ** 2, axis=0)
        rss_reduced = xp.sum((Y - X_red @ beta_no_int) ** 2, axis=0)
        df_num = self.num_frames - 1
        df_denom = len(design_matrix) - X_full.shape[1]
        fstat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_denom)
        return to_cpu(fstat).reshape(img_shape)


class RMAnovaModel(Model):
    """One-way repeated-measures ANOVA on FIR time-courses.

    Constructor responsibility:
      Build out the stacked FIR image and frame structures from ModelSpec,
      firstlevel_contrasts_df, and subject_variables_df.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        firstlevel_contrasts_df: pd.DataFrame,
        subject_variables_df: pd.DataFrame,
        **kwargs,
    ):
        self.model_spec = model_spec
        self.firstlevel_contrasts_df = firstlevel_contrasts_df
        self.subject_variables_df = subject_variables_df
        self._setup_outdir()

        self.num_frames = get_num_frames(self.model_spec, self.firstlevel_contrasts_df)
        get_img_fn = get_anova_contrasts_img
        if config.joblib_memory is not None:
            get_img_fn = config.joblib_memory.cache(get_anova_contrasts_img)
        self.contrasts_img = get_img_fn(
            self.model_spec, self.firstlevel_contrasts_df, self.num_frames
        )
        self.fdata = self.contrasts_img.get_fdata()

        self.design_df = pd.DataFrame({"frame": [f"frame_{i}" for i in range(self.num_frames)]})
        self.value_names = ["condition"]

        if isinstance(self.contrasts_img, nib.Nifti1Image):
            self.num_subjects = self.contrasts_img.shape[-1] // self.num_frames
        elif isinstance(self.contrasts_img, nib.Cifti2Image):
            self.num_subjects = self.contrasts_img.shape[0] // self.num_frames

        self._setup_image_geometry()
        self._maybe_smooth_cifti()

        self.separate_null_by_hemisphere = kwargs.get(
            "separate_null_by_hemisphere", getattr(config, "separate_null_by_hemisphere", False)
        )
        self.separate_null_by_parameter = kwargs.get(
            "separate_null_by_parameter", getattr(config, "separate_null_by_parameter", False)
        )

    def _maybe_smooth_cifti(self):
        if isinstance(self.contrasts_img, nib.Cifti2Image) and hasattr(self.contrasts_img.header.get_axis(1), "vertex"):
            self.out_suffix = ".dscalar.nii"
            contrasts_img_path = self.model_outdir / "stacked_contrasts.dscalar.nii"
            nib.save(self.contrasts_img, contrasts_img_path)
            if config.fwhm > 0:
                l_surf_path, r_surf_path = get_template_midthickness_paths_from_cifti_header(
                    self.contrasts_img.header, self.model_spec.space
                )
                smoothed_contrasts_img_path = self.model_outdir / "smoothed_stacked_contrasts.dscalar.nii"
                subprocess.run(shlex.split(
                    "wb_command -cifti-smoothing "
                    f"{str(contrasts_img_path.resolve())} {config.fwhm} {config.fwhm} COLUMN {str(smoothed_contrasts_img_path.resolve())} -fwhm "
                    f"-left-surface {l_surf_path} -right-surface {r_surf_path}"
                ))
                self.contrasts_img = nib.load(smoothed_contrasts_img_path)
                self.fdata = self.contrasts_img.get_fdata()

    def fit(self, *args, **kwargs) -> RMAnovaModelResults:
        """Fit RMAnovaModel and return RMAnovaModelResults."""
        xp = get_xp()
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (1, self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], 1)
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))
        else:
            raise ValueError(f"Cannot fit GLM for image type: {type(self.contrasts_img)}")

        Y = xp.asarray(self.fdata.reshape(glm_input_shape))
        n, k = self.num_subjects, self.num_frames
        Y_sep = Y.reshape((n, k, Y.shape[1]))
        grand_mean = Y_sep.mean(axis=(0, 1), keepdims=True)
        sub_mean = Y_sep.mean(axis=1, keepdims=True)
        frame_mean = Y_sep.mean(axis=0, keepdims=True)
        ss_cond = n * xp.sum((frame_mean - grand_mean) ** 2, axis=(0, 1))
        ss_sub = k * xp.sum((sub_mean - grand_mean) ** 2, axis=(0, 1))
        ss_total = xp.sum((Y_sep - grand_mean) ** 2, axis=(0, 1))
        ss_error = xp.maximum(ss_total - ss_sub - ss_cond, 0.0)
        df_cond = k - 1
        df_error = (n - 1) * (k - 1)
        ms_cond = ss_cond / df_cond
        ms_error = ss_error / df_error
        fstat = xp.where(ms_error > 0, ms_cond / ms_error, 0.0)
        fstat = xp.nan_to_num(fstat, nan=0.0)

        # Greenhouse-Geisser sphericity correction
        Y_c = Y_sep - Y_sep.mean(axis=0, keepdims=True)
        S = xp.einsum("sik,sjk->ijk", Y_c, Y_c) / (n - 1)
        s_diag = xp.diagonal(S, axis1=0, axis2=1)
        s_mean = xp.mean(S, axis=(0, 1))
        s_row_mean = xp.mean(S, axis=1)
        num = (k * (xp.mean(s_diag, axis=1) - s_mean)) ** 2
        den = (k - 1) * (
            xp.sum(S ** 2, axis=(0, 1))
            - 2 * k * xp.sum(s_row_mean ** 2, axis=0)
            + (k ** 2) * (s_mean ** 2)
        )
        eps_gg = xp.where(den == 0, 1.0, xp.clip(num / den, 1.0 / (k - 1), 1.0))
        eps_gg = xp.nan_to_num(eps_gg, nan=1.0 / (k - 1))

        fstat_cpu = to_cpu(fstat)
        eps_gg_cpu = to_cpu(eps_gg)

        pval = stats.f.sf(fstat_cpu, df_cond, df_error)
        df_cond_adj = eps_gg_cpu * df_cond
        df_error_adj = eps_gg_cpu * df_error
        pval_sphericity_corr = stats.f.sf(fstat_cpu, df_cond_adj, df_error_adj)

        return RMAnovaModelResults(
            model=self,
            uncorr_pvals=pval.reshape(img_shape),
            fstats=fstat_cpu.reshape(img_shape),
            epsilons=eps_gg_cpu.reshape(img_shape),
            pval_sphericity_corr=pval_sphericity_corr.reshape(img_shape),
        )

    def get_permuted_design_matrix(self) -> np.ndarray:
        """Return a random permutation indices matrix for within-subject frames."""
        n, k = self.num_subjects, self.num_frames
        perm_indices = np.empty((n, k), dtype=np.int64)
        for sub in range(n):
            perm_indices[sub] = np.random.permutation(k)
        return perm_indices

    def fit_permuted(self, permuted_design_matrix: np.ndarray) -> np.ndarray:
        """Re-fit RMAnova with permuted within-subject frames and return p-values."""
        xp = get_xp()
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (1, self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], 1)
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))

        Y = xp.asarray(self.fdata.reshape(glm_input_shape))
        n, k = self.num_subjects, self.num_frames
        Y_sep = Y.reshape((n, k, Y.shape[1]))
        perm_idx = xp.asarray(permuted_design_matrix)
        sub_idx = xp.arange(n)[:, None]
        Y_sep = Y_sep[sub_idx, perm_idx, :]

        grand_mean = Y_sep.mean(axis=(0, 1), keepdims=True)
        sub_mean = Y_sep.mean(axis=1, keepdims=True)
        frame_mean = Y_sep.mean(axis=0, keepdims=True)
        ss_cond = n * xp.sum((frame_mean - grand_mean) ** 2, axis=(0, 1))
        ss_sub = k * xp.sum((sub_mean - grand_mean) ** 2, axis=(0, 1))
        ss_total = xp.sum((Y_sep - grand_mean) ** 2, axis=(0, 1))
        ss_error = xp.maximum(ss_total - ss_sub - ss_cond, 0.0)
        df_cond = k - 1
        df_error = (n - 1) * (k - 1)
        ms_cond = ss_cond / df_cond
        ms_error = ss_error / df_error
        fstat = xp.where(ms_error > 0, ms_cond / ms_error, 0.0)
        fstat = xp.nan_to_num(fstat, nan=0.0)
        fstat_cpu = to_cpu(fstat)
        pval = stats.f.sf(fstat_cpu, df_cond, df_error)
        return pval.reshape(img_shape)

    def fit_permuted_stat(self, permuted_design_matrix: np.ndarray) -> np.ndarray:
        """Re-fit RMAnova with permuted within-subject frames and return F-stats."""
        xp = get_xp()
        if isinstance(self.contrasts_img, nib.Cifti2Image):
            img_shape = (1, self.fdata.shape[1])
            glm_input_shape = self.fdata.shape
        elif isinstance(self.contrasts_img, nib.Nifti1Image):
            img_shape = (*self.fdata.shape[:3], 1)
            glm_input_shape = (self.fdata.shape[3], math.prod(self.fdata.shape[:3]))

        Y = xp.asarray(self.fdata.reshape(glm_input_shape))
        n, k = self.num_subjects, self.num_frames
        Y_sep = Y.reshape((n, k, Y.shape[1]))
        perm_idx = xp.asarray(permuted_design_matrix)
        sub_idx = xp.arange(n)[:, None]
        Y_sep = Y_sep[sub_idx, perm_idx, :]

        grand_mean = Y_sep.mean(axis=(0, 1), keepdims=True)
        sub_mean = Y_sep.mean(axis=1, keepdims=True)
        frame_mean = Y_sep.mean(axis=0, keepdims=True)
        ss_cond = n * xp.sum((frame_mean - grand_mean) ** 2, axis=(0, 1))
        ss_sub = k * xp.sum((sub_mean - grand_mean) ** 2, axis=(0, 1))
        ss_total = xp.sum((Y_sep - grand_mean) ** 2, axis=(0, 1))
        ss_error = xp.maximum(ss_total - ss_sub - ss_cond, 0.0)
        df_cond = k - 1
        df_error = (n - 1) * (k - 1)
        ms_cond = ss_cond / df_cond
        ms_error = ss_error / df_error
        fstat = xp.where(ms_error > 0, ms_cond / ms_error, 0.0)
        fstat = xp.nan_to_num(fstat, nan=0.0)
        return to_cpu(fstat).reshape(img_shape)


# ---------------------------------------------------------------------------
# Image & Design helpers
# ---------------------------------------------------------------------------

def get_assumed_contrasts_img(
    model_spec: ModelSpec,
    firstlevel_contrasts_df: pd.DataFrame,
) -> nib.Cifti2Image | nib.Nifti1Image:
    """Stack first-level contrast maps into a single 4D or CIFTI image."""
    scalars, conditions = (
        [c.scalar for c in model_spec.firstlevel_contrasts],
        [c.name for c in model_spec.firstlevel_contrasts],
    )
    if len(scalars) > 1:
        sorted_pairs = sorted(zip(scalars, conditions), key=lambda tup: tup[1])
        scalars, conditions = map(list, zip(*sorted_pairs))
    space, task, session = (
        model_spec.space,
        model_spec.task,
        model_spec.session,
    )
    if len(conditions) > 1:
        paths = (
            firstlevel_contrasts_df
            .query(
                "condition in @conditions and "
                "frame_no == -1 and "
                "task == @task and "
                "space == @space and "
                "session == @session "
            )
            .sort_values(by=["subject", "condition"])
        )["path"]
        imgs = []
        for p in tqdm(paths, desc="Loading contrast maps..."):
            imgs.append(nib.load(p))
        img0 = imgs[0]
        logger.info("Stacking contrast maps...")
        if isinstance(img0, nib.Cifti2Image):
            fdata_stacked_list = []
            for i in range(len(conditions)):
                fdata_stacked_list.append(
                    np.empty((len(imgs) // len(conditions), img0.shape[1]), dtype=np.float32)
                )
                sub_idx = 0
                for j in range(i, len(imgs), len(conditions)):
                    fdata_stacked_list[-1][sub_idx, :] = imgs[j].get_fdata(dtype=np.float32)
                    sub_idx += 1
                fdata_stacked_list[-1] *= scalars[i]
            fdata_stacked = np.sum(fdata_stacked_list, axis=1, dtype=np.float32) / len(conditions)
            return nib.Cifti2Image(
                fdata_stacked,
                header=(
                    ScalarAxis(name=sorted(firstlevel_contrasts_df["subject"].unique())),
                    img0.header.get_axis(1),
                ),
                nifti_header=img0.nifti_header,
            )
        elif isinstance(img0, nib.Nifti1Image):
            fdata_stacked_list = []
            for i in range(len(conditions)):
                fdata_stacked_list.append(
                    np.empty((*img0.shape[:3], len(imgs) // len(conditions)), dtype=np.float32)
                )
                sub_idx = 0
                for j in range(i, len(imgs), len(conditions)):
                    fdata_stacked_list[-1][sub_idx, :] = imgs[j].get_fdata(dtype=np.float32)
                    sub_idx += 1
                fdata_stacked_list[-1] *= scalars[i]
            fdata_stacked = np.sum(fdata_stacked_list, axis=(0, 1, 2), dtype=np.float32) / len(conditions)
            return nib.Nifti1Image(
                fdata_stacked,
                affine=img0.affine,
                header=img0.header,
            )
        else:
            raise ValueError(f"Unexpected image type {type(img0)} (this shouldn't happen)")
    else:
        paths = (
            firstlevel_contrasts_df
            .query(
                "condition == @conditions[0] and "
                "frame_no == -1 and "
                "task == @task and "
                "space == @space and "
                "session == @session "
            )
            .sort_values(by="subject")
        )["path"]
        imgs = []
        for p in tqdm(paths, desc="Loading contrast maps..."):
            imgs.append(nib.load(p))
        img0 = imgs[0]
        if isinstance(img0, nib.Cifti2Image):
            fdata_stacked = np.empty((len(imgs), img0.shape[1]), dtype=np.float32)
            for i, img in tqdm(enumerate(imgs), desc="Stacking contrast maps..."):
                fdata_stacked[i, :] = img.get_fdata(dtype=np.float32)
            fdata_stacked *= scalars[0]
            return nib.Cifti2Image(
                fdata_stacked,
                header=(
                    ScalarAxis(name=sorted(firstlevel_contrasts_df["subject"].unique())),
                    img0.header.get_axis(1),
                ),
                nifti_header=img0.nifti_header,
            )
        elif isinstance(img0, nib.Nifti1Image):
            fdata_stacked = np.empty((*img0.shape[:3], len(imgs)), dtype=np.float32)
            for i, img in tqdm(enumerate(imgs), desc="Stacking contrast maps..."):
                fdata_stacked[..., i] = img.get_fdata(dtype=np.float32)
            fdata_stacked *= scalars[0]
            return nib.Nifti1Image(
                fdata_stacked,
                affine=img0.affine,
                header=img0.header,
            )
        else:
            raise ValueError(f"Unexpected image type {type(img0)} (this shouldn't happen)")


def get_ols_design_df(
    model_spec: ModelSpec,
    subject_variables_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build design matrix from patsy formula and subject variables."""
    design_df = dmatrix(model_spec.subject_variables, subject_variables_df, return_type="dataframe")
    design_df["subject"] = subject_variables_df["subject"]
    design_df = (
        design_df
        .sort_values(by="subject")
        .reset_index(drop=True)
    )
    design_df.drop(columns=["subject"], inplace=True)
    return design_df


# For backwards compatibility with assumed.py
get_design_df = get_ols_design_df


def get_num_frames(
    model_spec: ModelSpec,
    firstlevel_contrasts_df: pd.DataFrame,
) -> int:
    """Extract number of FIR frames from first subject's image."""
    condition = model_spec.function_args[0]
    space, task, session = (
        model_spec.space,
        model_spec.task,
        model_spec.session,
    )
    paths = (
        firstlevel_contrasts_df
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


def get_anova_contrasts_img(
    model_spec: ModelSpec,
    firstlevel_contrasts_df: pd.DataFrame,
    num_frames: int,
    subject_variables_df: pd.DataFrame | None = None,
    var_of_interest: str | None = None,
) -> nib.Cifti2Image | nib.Nifti1Image:
    """Stack FIR contrast maps for ANOVA."""
    condition = model_spec.function_args[0]
    space, task, session = (
        model_spec.space,
        model_spec.task,
        model_spec.session,
    )
    if var_of_interest is not None and subject_variables_df is not None:
        null_subjects = subject_variables_df.loc[subject_variables_df[var_of_interest].isna(), "subject"]
        firstlevel_contrasts_df_ = firstlevel_contrasts_df[~firstlevel_contrasts_df["subject"].isin(null_subjects)]
    else:
        firstlevel_contrasts_df_ = firstlevel_contrasts_df.copy()
    paths = (
        firstlevel_contrasts_df_
        .query(
            "condition == @condition and "
            "frame_no == -1 and "
            "task == @task and "
            "space == @space and "
            "session == @session"
        )
        .sort_values(by=["subject"])
    )["path"].to_list()
    logger.debug("Loading contrasts...")
    img0 = nib.load(paths[0])
    if isinstance(img0, nib.Cifti2Image) and img0.dataobj.shape[0] == 1:
        raise ValueError("Image should contain multiple frames for an FIR ANOVA.")
    elif isinstance(img0, nib.Nifti1Image) and img0.dataobj.shape[3] == 1:
        raise ValueError("Image should contain multiple frames for an FIR ANOVA.")
    fdatas = [nib.load(p).get_fdata() for p in paths]
    logger.debug("Done!")
    if isinstance(img0, nib.Cifti2Image):
        logger.info("Concatenating...")
        sub_count = len(firstlevel_contrasts_df_["subject"].unique())
        total_frames = num_frames * sub_count
        fdata_stacked = np.concatenate(fdatas, axis=0)
        logger.info("Making stacked image...")
        return nib.Cifti2Image(
            fdata_stacked,
            header=(
                ScalarAxis(name=["frame"] * total_frames),
                img0.header.get_axis(1),
            ),
            nifti_header=img0.nifti_header,
        )
    elif isinstance(img0, nib.Nifti1Image):
        fdata_stacked = np.concatenate(fdatas, axis=3)
        return nib.Nifti1Image(
            fdata_stacked,
            affine=img0.affine,
            header=img0.header,
        )
    else:
        raise TypeError(f"Unexpected image type {type(img0)} (this shouldn't happen)")


def get_twoway_anova_design_df(
    model_spec: ModelSpec,
    subject_variables_df: pd.DataFrame,
    num_frames: int,
) -> tuple[pd.DataFrame, str]:
    """Build design matrix for two-way repeated-measures ANOVA."""
    variable = model_spec.function_args[1]
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
    if categorical:
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
        columns=list(design_df.columns[1:]) + frame_column_names,
    )
    if categorical:
        cat_columns = [c for c in design_df.columns if c.startswith(variable)]
        for cat_column, frame_no in itertools.product(cat_columns, frame_column_names):
            interaction_name = f"{cat_column}_{frame_no}_interaction"
            design_df[interaction_name] = design_df[cat_column] * design_df[frame_no]
    else:
        for frame_no in frame_column_names:
            interaction_name = f"{variable}_{frame_no}_interaction"
            design_df[interaction_name] = design_df[variable] * design_df[frame_no]
    design_df.insert(0, "subject", subj_series)
    return design_df, variable