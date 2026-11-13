from typing import TypedDict, Required, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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

def run_models(
    model_specs: List[ModelSpec],
    firstlevel_contrasts_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame
):
    for model_spec in model_specs:
        _run_model(model_spec, firstlevel_contrasts_df, subject_variables_df)


def _run_model(
    model_spec: ModelSpec,
    firstlevel_contrasts_df: pd.DataFrame,
    subject_variables_df: pd.DataFrame
):
    match model_spec.model_type:
        case "OLS":
            from .assumed import run_ols_model
            run_ols_model(model_spec, firstlevel_contrasts_df, subject_variables_df)
        case "onesampttest":
            from .assumed import run_ols_model
            if (num_args := len(model_spec.function_args)) != 1:
                raise ValueError(
                    "Function fir_twoway_rm_anova() takes 2 arguments: "
                    f"'beta' and 'variable'. Received {num_args}: {', '.join(model_spec['function_args'])}"
                )
            model_spec.firstlevel_contrasts = [FirstLevelContrast(name=model_spec.function_args[0])]
            model_spec.subject_variables = "1"
            run_ols_model(model_spec, firstlevel_contrasts_df, subject_variables_df)
        case "fir_rm_anova":
            if (num_args := len(model_spec.function_args)) != 1:
                raise ValueError(
                    "Function fir_rm_anova() takes 1 argument: "
                    f"'beta'. Received {num_args}: {', '.join(model_spec.function_args)}"
                )
            from .fir_anova import run_anova_model
            run_anova_model(model_spec, firstlevel_contrasts_df, subject_variables_df)
        case "fir_twoway_rm_anova":
            if (num_args := len(model_spec.function_args)) != 2:
                raise ValueError(
                    "Function fir_twoway_rm_anova() takes 2 arguments: "
                    f"'beta' and 'variable'. Received {num_args}: {', '.join(model_spec['function_args'])}"
                )
            from .fir_anova import run_anova_model
            run_anova_model(model_spec, firstlevel_contrasts_df, subject_variables_df)
        case _:
            raise ValueError(f"Unknown function or model type {model_spec['model_type']}.")

class ModelResults(ABC):
    pass


class Model(ABC):
    @abstractmethod
    def __init__(self,
                 model_spec: ModelSpec,
                 firstlevel_contrasts_df: pd.DataFrame,
                 subject_variables_df: pd.DataFrame):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> ModelResults:
        pass


class AssumedResponseOLS(Model):
    def __init__(self,
                 model_spec: ModelSpec,
                 firstlevel_contrasts_df: pd.DataFrame,
                 subject_variables_df: pd.DataFrame):
        self.model_spec = model_spec
        from .assumed import get_assumed_contrasts_img
        self.firstlevel_contrasts_df = firstlevel_contrasts_df
        self.subject_variables_df = subject_variables_df

    def fit(self, *args, **kwargs) -> ModelResults:
        pass


class FIRResponseModel():
    pass