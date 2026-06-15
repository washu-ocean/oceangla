from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np


def cifti_compatible_structures(
    headers_or_paths: Sequence[nib.cifti2.cifti2.Cifti2Header] | Sequence[str] | Sequence[Path],
) -> list[str]:
    """
    Return a list of common CIFTI structure names (e.g. 'CIFTI_STRUCTURE_CORTEX_LEFT')
    between all elements in `headers_or_paths`.

    Parameters
    ==========
    headers_or_paths: Sequence[nib.cifti2.cifti2.Cifti2Header] | Sequence[str] | Sequence[Path]
        Sequence of Cifti2Header, or str/Path pointing to a valid CIFTI image

    Returns
    =======
    list[str]
        List of common CIFTI structure names between all elements in `headers_or_paths`
    """
    if len(headers_or_paths) < 2:
        raise ValueError(
            "`headers_or_paths` must contain at least 2 of Cifti2Header, str, or Path objects."
        )
    if not all([isinstance(headers_or_paths[i], type(headers_or_paths[0])) for i in range(1, len(headers_or_paths))]):
        raise ValueError(
            "All elements of `headers_or_paths` must be of the same type (Cifti2Header, str, or Path)"
        )
    if type(headers_or_paths[0]) not in (str, Path, nib.cifti2.cifti2.Cifti2Header):
        raise ValueError(
            f"All elements of `headers_or_paths` must of type Cifti2Header, str, or Path -- received {type(headers_or_paths[0])}"
        )
    cifti_headers = (
        [nib.load(p).header for p in headers_or_paths]
        if isinstance(type(headers_or_paths[0]), str) or isinstance(type(headers_or_paths[0]), Path)
        else headers_or_paths
    )
    unique_struct_names = set.intersection(
        *(set(np.unique(header.get_axis(1).name)) for header in cifti_headers)
    )
    common_struct_names = []
    for name in unique_struct_names:
        if all((
            len(header.get_axis(1).name[header.get_axis(1).name == name]) == len(cifti_headers[0].get_axis(1).name[cifti_headers[0].get_axis(1).name == name])
            for header in cifti_headers[1:]
        )):
            common_struct_names.append(str(name))

    return common_struct_names
