from typing import TypedDict, Required
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ModelDesc = TypedDict(
    "ModelDesc",
    {
        "model_type": Required[str],
        "model_name": str,
        "depvars": list[str],
        "indepvars": str,
        "function_args": list[str],
        "session": str,
        "space": str,
        "task": str,
        "frame_count": np.uint32
    },
    total=False
)

def run_models(
    model_descs: list[ModelDesc],
    subject_activation_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame
):
    for model_desc in model_descs:
        _run_model(model_desc, subject_activation_df, subject_variables_df)


def _run_model(
    model_desc: ModelDesc,
    subject_activation_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame
):
    match model_desc["model_type"]:
        case "OLS":
            from .ols import run_ols_model
            run_ols_model(model_desc, subject_activation_df, subject_variables_df)
        case "onesampttest":
            from .ols import run_ols_model
            if (num_args := len(model_desc["function_args"])) != 1:
                raise ValueError(
                    "Function fir_twoway_rm_anova() takes 2 arguments: "
                    f"'beta' and 'variable'. Received {num_args}: {', '.join(model_desc['function_args'])}"
                )
            model_desc["depvars"] = ["+" + model_desc["function_args"][0]] # will find a better way of adding scalars to this soon
            model_desc["indepvars"] = []
            run_ols_model(model_desc, subject_activation_df, subject_variables_df)
        case "fir_rm_anova":
            pass
        case "fir_twoway_rm_anova":
            if (num_args := len(model_desc["function_args"])) != 2:
                raise ValueError(
                    "Function fir_twoway_rm_anova() takes 2 arguments: "
                    f"'beta' and 'variable'. Received {num_args}: {', '.join(model_desc['function_args'])}"
                )
            from .fir_anova import run_anova_model
            run_anova_model(model_desc, subject_activation_df, subject_variables_df)
        case _:
            raise ValueError(f"Unknown function or model type {model_desc['model_type']}.")