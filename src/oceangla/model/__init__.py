from .ols import OLSModel
from typing import TypedDict, Required

VALID_FUNCS = ("onesampttest", "fir_rmanova")

ModelDesc = TypedDict(
    "ModelDesc",
    {
        "model_type": Required[str],
        "depvars": list[str],
        "indepvars": list[str],
        "function_args": list[str]
    },
    total=False
)

__all__ = ["OLSModel"]
