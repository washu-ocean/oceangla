from pathlib import Path
import logging
import sqlite3

import numpy as np
import nibabel as nib
from joblib import Memory, Parallel, delayed

from .config import config

logger = logging.getLogger(__name__)


def find_beta_maps(fladirs: list[str] | list[Path] | str | Path) -> np.ndarray:
    if isinstance(fladirs, (str, Path)):
        fladirs = [fladirs]
    fladirs = [Path(d) for d in fladirs]

    beta_dict = {
        "nifti": [],
        "cifti": []
    }

    for fladir in fladirs:
        def _find_fla_maps(suffix):
            try:
                sub_fla_dir = fladir / f"sub-{sub}"
                sub_activation_map = next(sub_fla_dir.rglob(f"sub-{sub}_*_condition-{condition}_stat-effect_boldmap{suffix}"))
                logger.debug(f'Found a {suffix} map for sub {sub}, '
                             f'condition {condition} in {sub_fla_dir}')
                return sub_activation_map
            # if a map is not found
            except StopIteration:
                logger.warning(f'Could not find {suffix} map for sub {sub}, '
                               f'condition {condition} in {sub_fla_dir}')
                return None
            # if it couldn't load for some reason (this shouldn't happen)
            except nib.filebasedimages.ImageFileError:
                logger.error(f'Error loading {suffix} map for sub {sub}, '
                             f'condition {condition} in {sub_fla_dir}')
                return None
