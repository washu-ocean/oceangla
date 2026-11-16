from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from oceangla.config import config
from .model import (
    ModelSpec,
    TwoWayAnovaModel,
    RMAnovaModel,
    TwoWayAnovaModelResults,
    RMAnovaModelResults,
    AnovaModelResults,
    get_num_frames,
    get_anova_contrasts_img,
    get_twoway_anova_design_df,
    apply_corrections,
)

logger = logging.getLogger(__name__)


def run_anova_model(
    model_spec: ModelSpec,
    firstlevel_contrasts_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame,
):
    """Run an ANOVA model: instantiate, fit, save, and apply corrections."""
    match model_spec.model_type:
        case "fir_twoway_rm_anova":
            model = TwoWayAnovaModel(
                model_spec=model_spec,
                firstlevel_contrasts_df=firstlevel_contrasts_df,
                subject_variables_df=subject_variables_df,
                alpha=config.alphas,
            )
        case "fir_rm_anova":
            model = RMAnovaModel(
                model_spec=model_spec,
                firstlevel_contrasts_df=firstlevel_contrasts_df,
                subject_variables_df=subject_variables_df,
                alpha=config.alphas,
            )
        case _:
            raise NotImplementedError(f"{model_spec.model_type} not implemented yet.")

    results = model.fit()
    results.save()
    vol_method = config.volume_cluster_strategy.lower() if config.volume_cluster_strategy else "nn1"
    apply_corrections(results, vol_method)
    return results


__all__ = [
    "TwoWayAnovaModel",
    "RMAnovaModel",
    "TwoWayAnovaModelResults",
    "RMAnovaModelResults",
    "AnovaModelResults",
    "run_anova_model",
    "get_num_frames",
    "get_anova_contrasts_img",
    "get_twoway_anova_design_df",
]
