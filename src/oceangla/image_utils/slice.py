from itertools import product

import nibabel as nib
import numpy as np


def get_spatial_slices(
    activation: np.ndarray,
    image_type: str = "CIFTI",
    volume_mask: nib.Nifti1Image | None = None,
) -> (tuple, int):
    if image_type == "CIFTI":
        return (
            tuple((np.s_[:, idx] for idx in range(activation.shape[1]))),
            int(activation.shape[1]),
        )
    elif image_type == "NIFTI":
        if volume_mask is None:
            return (
                tuple(
                    (
                        np.s_[x_idx, y_idx, z_idx, :]
                        for x_idx, y_idx, z_idx in product(
                            *[range(shape) for shape in activation.shape[:3]]
                        )
                    )
                ),
                int(np.prod(activation.shape[:3])),
            )
        else:
            return (
                tuple(
                    (
                        np.s_[indices[0], indices[1], indices[2], :]
                        for indices in np.argwhere(volume_mask.dataobj != 0)
                    )
                ),
                int(len(np.argwhere(volume_mask.dataobj != 0))),
            )
    else:
        raise ValueError(
            "spatial_index_iter image type must be either 'NIFTI' or 'CIFTI', "
            f"received {image_type}"
        )
