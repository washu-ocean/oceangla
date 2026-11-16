from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from oceangla.config import config
from .model import (
    ModelSpec,
    OLSModel,
    OLSModelResults,
    get_assumed_contrasts_img,
    get_ols_design_df,
    get_design_df,
    save_null_histogram,
    apply_corrections,
)

logger = logging.getLogger(__name__)


def run_ols_model(
    model_spec: ModelSpec,
    firstlevel_contrasts_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame,
) -> OLSModelResults:
    """Run an OLS model: instantiate, fit, save, and apply corrections."""
    model = OLSModel(
        model_spec=model_spec,
        firstlevel_contrasts_df=firstlevel_contrasts_df,
        subject_variables_df=subject_variables_df,
        perms=config.perms,
        alpha=config.alphas,
        volume_cluster_strategy=config.volume_cluster_strategy,
        dlabel_paths=config.dlabel_paths,
        separate_null_by_hemisphere=config.separate_null_by_hemisphere,
        separate_null_by_parameter=config.separate_null_by_parameter,
    )
    results = model.fit()
    results.save()
    vol_method = config.volume_cluster_strategy.lower() if config.volume_cluster_strategy else "nn1"
    apply_corrections(results, vol_method)
    return results


__all__ = [
    "OLSModel",
    "OLSModelResults",
    "run_ols_model",
    "get_assumed_contrasts_img",
    "get_ols_design_df",
    "get_design_df",
    "save_null_histogram",
]
