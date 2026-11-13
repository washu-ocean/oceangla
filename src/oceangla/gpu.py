"""GPU acceleration utilities using CuPy when a GPU is present."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# CuPy detection
try:
    import cupy as cp
    _CUPY_INSTALLED = True
except (ImportError, Exception):
    cp = None
    _CUPY_INSTALLED = False


def is_cupy_installed() -> bool:
    """Return True if CuPy package is installed."""
    return _CUPY_INSTALLED and cp is not None


def is_gpu_available() -> bool:
    """Return True if CuPy is installed, CUDA is available, and >=1 GPU exists."""
    if not is_cupy_installed():
        return False
    try:
        return bool(cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0)
    except Exception as e:
        logger.debug(f"CUDA device check failed: {e}")
        return False


def should_use_gpu(use_gpu: bool | None = None) -> bool:
    """Determine whether GPU acceleration should be used.

    Parameters
    ----------
    use_gpu : bool or None, optional
        If True, force GPU (raises RuntimeError if GPU not available).
        If False, force CPU.
        If None, check config.use_gpu or auto-detect if GPU is available.
    """
    from .config import config

    flag = use_gpu if use_gpu is not None else getattr(config, "use_gpu", None)

    if flag is True:
        if not is_gpu_available():
            raise RuntimeError(
                "GPU execution was requested (--gpu), but CuPy is not installed or no CUDA GPU was detected."
            )
        return True
    elif flag is False:
        return False
    else:
        # Auto-detect GPU presence
        return is_gpu_available()


def get_xp(use_gpu: bool | None = None):
    """Return cupy if GPU is used, otherwise numpy."""
    if should_use_gpu(use_gpu):
        return cp
    return np


def to_gpu(arr: Any) -> Any:
    """Move an array to GPU using CuPy if GPU is available, else return array."""
    if arr is None:
        return None
    if should_use_gpu():
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Any) -> Any:
    """Move an array from CuPy GPU to NumPy CPU, or return as np.ndarray."""
    if arr is None:
        return None
    if is_cupy_installed() and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    if hasattr(arr, "get") and callable(arr.get):
        return arr.get()
    return np.asarray(arr)
