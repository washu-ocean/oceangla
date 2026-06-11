import logging
from functools import partial

import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

VALID_NEIGHBOR_STRATEGIES = ("NN1", "NN2", "NN3")


def get_volume_array_from_cifti_array(arr: np.ndarray, header: nib.cifti2.cifti2.Cifti2Header) -> np.ndarray:
    """
    Return a 3-D ndarray from a CIFTI with a volume component, where values of
    1 represent unmasked (valid) subcortical voxels, and values of -1 represent
    invalid voxels.

    Parameters
    ----------
    img : nib.cifti2.cifti2.Cifti2Image
        A CIFTI image with a volumetric component.

    Returns
    -------
    np.ndarray
        A 3-D ndarray representing a subcortical voxel mask
    """
    bm_axis = header.get_axis(1)
    if bm_axis.volume_shape is None:
        raise TypeError("CIFTI has no volumetric component.")
    volume = np.full((*bm_axis.volume_shape, arr.shape[0]), np.nan, dtype=arr.dtype)
    voxel_indices = bm_axis.voxel[(bm_axis.voxel != -1).all(axis=1)]
    cifti_indices = np.argwhere((bm_axis.voxel != -1).all(axis=1)).flatten()
    volume[tuple(voxel_indices.T)] = arr[:, tuple(cifti_indices.T)].T
    return volume


def get_volume_array_from_cifti(img: nib.cifti2.cifti2.Cifti2Image) -> np.ndarray:
    """
    Return a 3-D ndarray from a CIFTI with a volume component, where values of
    1 represent unmasked (valid) subcortical voxels, and values of -1 represent
    invalid voxels.

    Parameters
    ----------
    img : nib.cifti2.cifti2.Cifti2Image
        A CIFTI image with a volumetric component.

    Returns
    -------
    np.ndarray
        A 3-D ndarray representing a subcortical voxel mask
    """
    bm_axis = img.header.get_axis(1)
    if bm_axis.volume_shape is None:
        raise TypeError("CIFTI has no volumetric component.")
    volume = np.full(bm_axis.volume_shape, 0, dtype=np.int64)
    voxel_indices = bm_axis.voxel[np.where((bm_axis.voxel != [-1, -1, -1]).all(axis=1))[0], :]
    volume[tuple(voxel_indices.T)] = 1
    return volume


def get_neighboring_voxels_at_index(idx: np.ndarray[np.int64], volume_shape: tuple[int], strategy: str = "NN3") -> np.ndarray[np.int64]:
    """
    Return a 2-D array of neighboring voxels' indices at a given voxel index.

    Parameters
    ----------
    idx : np.ndarray[np.int64]
        A voxel index of shape (3,)
    volume_shape: tuple[int]
        Shape of the volume this index is in
    strategy: str, default "NN3"
        Strategy for determining which voxels are considered neighbors. Choices are "NN1"
        (neighbors touch faces, maximum of 6), "NN2" (neighbors touch faces or edges, maximum
        of 18), or "NN3" (neighbors touch faces, edges, or corners, maximum 26)

    Returns
    -------
    np.ndarray[np.int64]
        2-D array of neighboring indices, of shape (# of neighbors, 3).
    """
    if not idx.shape == (3,):
        raise ValueError(f"idx must be an array of np.int64 of size (3,), received array {idx=}")
    if not len(volume_shape) == 3:
        raise ValueError(f"volume_shape must be a 3-tuple of integers, received {volume_shape=}")
    volume_shape = np.array(volume_shape, dtype=np.int64)
    match strategy:
        case "NN1":
            rel_indices = np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            ])
        case "NN2":
            rel_indices = np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                [1, 1, 0], [0, 1, 1], [1, 0, 1],
                [-1, 0, -1], [-1, -1, 0], [0, -1, -1],
                [-1, 1, 0], [0, -1, 1], [-1, 0, 1],
                [1, 0, -1], [1, -1, 0], [0, 1, -1],
            ])
        case "NN3":
            rel_indices = np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                [1, 1, 0], [0, 1, 1], [1, 0, 1],
                [-1, 0, -1], [-1, -1, 0], [0, -1, -1],
                [-1, 1, 0], [0, -1, 1], [-1, 0, 1],
                [1, 0, -1], [1, -1, 0], [0, 1, -1],
                [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
                [-1, -1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]
            ])
        case _:
            raise ValueError(f"strategy must be one of 'NN1', 'NN2', or 'NN3', received {strategy}")
    neighbors = (idx + rel_indices)
    neighbors = neighbors[np.argwhere(
        (neighbors >= 0).all(axis=1) & (volume_shape - neighbors > 0).all(axis=1)
    )].squeeze()
    return neighbors


def get_voxel_clusters(threshold_mask: np.ndarray, strategy: str = "NN3") -> list[np.ndarray[np.int64]]:
    """
    Get clusters of voxels via a volume mask.

    Parameters
    ----------
    threshold_mask: np.ndarray
        A mask the same shape as the target volume, where indices with a value of 1
        are considered a significant voxel.
    strategy: str, default "NN3"
        Strategy for determining which voxels are considered neighbors. Choices are "NN1"
        (neighbors touch faces, maximum of 6), "NN2" (neighbors touch faces or edges, maximum
        of 18), or "NN3" (neighbors touch faces, edges, or corners, maximum 26)

    Returns
    -------
    list[np.ndarray[np.int64]]
        List of arrays of voxel indices representing contiguous clusters.
    """
    if not len(threshold_mask.shape) == 3:
        raise ValueError(f"threshold_mask must be 3-dimensional, received mask of shape {threshold_mask.shape}")
    get_neighboring_voxels_at_index_ = partial(get_neighboring_voxels_at_index, volume_shape=threshold_mask.shape, strategy=strategy)
    seen = np.zeros_like(threshold_mask, dtype=bool)
    clusters = []
    signif_vtxs = np.argwhere(threshold_mask == 1)
    for signif_vtx in signif_vtxs:
        if seen[*signif_vtx]:
            continue
        stack = [signif_vtx]
        seen[*signif_vtx] = True
        cluster = []
        while stack:
            cur_idx = stack.pop()
            cluster.append(cur_idx)
            for neighbor in get_neighboring_voxels_at_index_(cur_idx):
                if threshold_mask[*neighbor] == 1 and not seen[*neighbor]:
                    seen[*neighbor] = True
                    stack.append(neighbor)
        clusters.append(np.array(cluster, dtype=np.int64))
    return clusters



def get_voxel_cluster_sizes(clusters: list[np.ndarray[np.int64]] | list[list[np.ndarray[np.int64]]]) -> list[np.int64]:
    """
    From a list of cluster arrays, return the number of voxels in the largest cluster.

    Parameters
    ----------
    clusters: list[np.ndarray[np.int64]]
        List of arrays representing clusters

    Returns
    -------
    np.int64
        Size, in voxels, of the largest cluster.
    """
    return np.array([cluster.shape[0] for cluster in clusters]).astype(np.int64)


def get_biggest_voxel_cluster_sizes(clusters: list[np.ndarray[np.int64]] | list[list[np.ndarray[np.int64]]]) -> list[np.int64]:
    """
    From a list of cluster arrays, return the number of voxels in the largest cluster.

    Parameters
    ----------
    clusters: list[np.ndarray[np.int64]]
        List of arrays representing clusters

    Returns
    -------
    np.int64
        Size, in voxels, of the largest cluster.
    """
    return np.max([cluster.shape[0] for cluster in clusters]).astype(np.int64)


