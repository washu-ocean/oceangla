from collections.abc import Generator
import logging

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_surf_data(img: nib.cifti2.cifti2.Cifti2Image) -> dict[str, np.ndarray]:
    fdata = img.get_fdata()
    bm_axis = img.header.get_axis(1)
    surf_data_dict = {}
    for name, data_indices, model in bm_axis.iter_structures():
        if name in ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"):
            fdata_t = fdata.copy().T[data_indices]
            vtx_indices = model.vertex
            surf_data = np.full(
                (vtx_indices.max() + 1,) + fdata_t.shape[1:],
                np.nan,
                dtype=fdata_t.dtype
            )
            surf_data[vtx_indices] = fdata_t
            surf_data_dict[name] = surf_data.T
    return surf_data_dict


def __slice_to_index_array(my_slice: slice, max_size: int) -> np.ndarray:
    return np.array(range(*my_slice.indices(max_size)))


def __check_network_df_is_valid(network_df: pd.DataFrame) -> None:
    if not all(
        (
            "index" in network_df.columns,
            network_df["index"].dtype in (np.int32, np.int64),
            "network_label" in network_df.columns,
        )
    ):
        raise ValueError(
            "DataFrame describing networks must have an 'index' column "
            "of type np.int32 or np.int64, and a 'network_label' column. "
            f"Columns in dataframe: {network_df.columns}"
        )


def get_network_slices(
    dlabel_img: nib.cifti2.cifti2.Cifti2Image,
    network_df: pd.DataFrame,
) -> Generator[np.ndarray, None, None]:
    __check_network_df_is_valid(network_df)
    dlabel_surf_data = get_surf_data(dlabel_img)
    for network_name, network_group_df in network_df.groupby("network_label"):
        l_label_indices = np.argwhere(
            np.isin(dlabel_surf_data["CIFTI_STRUCTURE_CORTEX_LEFT"], network_group_df["index"]).flatten()
        ).flatten()
        r_label_indices = np.argwhere(
            np.isin(dlabel_surf_data["CIFTI_STRUCTURE_CORTEX_RIGHT"], network_group_df["index"]).flatten()
        ).flatten()
        yield (
            np.concat(
                (
                    l_label_indices,
                    r_label_indices + np.max(l_label_indices) + 1
                )
            )
        )
