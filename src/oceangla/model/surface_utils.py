from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
from templateflow import api as tflow


# Not sure how robust this is, but should work for fsLR case at least
def get_template_midthicknesses_from_cifti_header(
    header: nib.cifti2.cifti2.Cifti2Header, space: str
) -> tuple[nib.cifti2.cifti2.Cifti2Image, nib.cifti2.cifti2.Cifti2Image]:
    n_verts = np.max(header.get_axis(1).vertex)
    n_verts_str = f"{(n_verts - (n_verts % 1000)) // 1000}k"  # for example 32k for fsLR
    l_search_kwargs = {
        "density": n_verts_str,
        "hemi": "L",
        "suffix": "midthickness",
        "desc": None,
    }
    r_search_kwargs = deepcopy(l_search_kwargs)
    r_search_kwargs["hemi"] = "R"
    l_hem_path, r_hem_path = (
        tflow.get(space, **l_search_kwargs),
        tflow.get(space, **r_search_kwargs),
    )
    if isinstance(l_hem_path, list):
        raise ValueError(
            f"{'More than 1' if len(l_hem_path) > 0 else 'No'} possible surfaces found for left-hemisphere {': ' + ','.join(l_hem_path) if len(l_hem_path) > 0 else ''}. Searched with following params: "
            + ",".join([f"{k}={v}" for k, v in l_search_kwargs.items()])
        )
    return nib.load(l_hem_path), nib.load(r_hem_path)


def extract_hemi_values(
    values_full: np.ndarray, hdr: nib.cifti2.cifti2.Cifti2Header, nL: int, nR: int
) -> tuple[np.ndarray, np.ndarray]:
    bm_axis = hdr.get_axis(1)
    L_vals = np.full((*values_full.shape[:-1], nL), np.nan, dtype=np.float32)
    R_vals = np.full((*values_full.shape[:-1], nR), np.nan, dtype=np.float32)
    seen_structures = []
    for name, slc, bmodel in bm_axis.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            seen_structures.append(name)
            vidx = bmodel.vertex.astype(np.int64)
            if np.max(vidx) >= nL:
                raise RuntimeError(
                    f"Left surface has {nL} verts but CIFTI references vertex {np.max(vidx)}. Wrong L_surf?"
                )
            L_vals[..., vidx] = values_full[..., slc].astype(np.float32)
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            seen_structures.append(name)
            vidx = bmodel.vertex.astype(np.int64)
            if np.max(vidx) >= nR:
                raise RuntimeError(
                    f"Right surface has {nR} verts but CIFTI references vertex {np.max(vidx)}. Wrong R_surf?"
                )
            R_vals[..., vidx] = values_full[..., slc].astype(np.float32)
        if len(seen_structures) == 2:
            break
    return L_vals, R_vals


def build_adjacency_from_faces(n_verts: int, faces: np.ndarray) -> list[np.ndarray]:
    neigh = [set() for _ in range(n_verts)]
    f = faces.astype(np.int64)
    for a, b, c in f:
        neigh[a].add(b)
        neigh[a].add(c)
        neigh[b].add(a)
        neigh[b].add(c)
        neigh[c].add(a)
        neigh[c].add(b)
    return [np.fromiter(sorted(s), dtype=np.int32) for s in neigh]


def get_cluster_index_groups(
    mask: np.ndarray, neighbors: list[np.ndarray]
) -> list[np.ndarray]:
    seen = np.zeros(mask.shape[0], dtype=bool)
    comps = []
    for v in np.where(mask)[0]:
        if seen[v]:
            continue
        stack = [int(v)]
        seen[v] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in neighbors[u]:
                if mask[w] and not seen[w]:
                    seen[w] = True
                    stack.append(int(w))
        comps.append(np.array(comp, dtype=np.int32))
    return comps


def get_cluster_sizes_from_pmap(
    pval: np.ndarray,
    pthr: float,
    neighbors: list[np.ndarray],
    area_map: np.ndarray | None = None,
) -> list[int]:
    if len(pval.shape) == 2:
        cluster_sizes_2d = []
        for beta in range(pval.shape[0]):
            thresholded_mask = np.squeeze(
                np.isfinite(pval[beta, :]) & (pval[beta, :] < pthr)
            )
            cluster_idx_groups = get_cluster_index_groups(thresholded_mask, neighbors)
            if area_map is not None:
                cluster_sizes_2d.append(
                    np.array(
                        [
                            np.sum(area_map[cluster_idx_group])
                            for cluster_idx_group in cluster_idx_groups
                        ]
                    )
                )
            else:
                cluster_sizes_2d.append(
                    np.array(
                        [
                            len(cluster_idx_group)
                            for cluster_idx_group in cluster_idx_groups
                        ]
                    )
                )
        return cluster_sizes_2d
    elif len(pval.shape) == 1:
        thresholded_mask = np.isfinite(pval) & (pval < pthr)
        cluster_idx_groups = get_cluster_index_groups(thresholded_mask, neighbors)
        if area_map is not None:
            return np.array(
                [
                    np.sum(area_map[cluster_idx_group])
                    for cluster_idx_group in cluster_idx_groups
                ]
            )
        else:
            return np.array(
                [len(cluster_idx_group) for cluster_idx_group in cluster_idx_groups]
            )


def get_biggest_clusters_from_pmap(
    pval: np.ndarray,
    pthr: float,
    neighbors: list[np.ndarray],
    area_map: np.ndarray | None = None,
) -> list[int]:
    if len(pval.shape) == 2:
        return [
            np.max(cluster_sizes)
            for cluster_sizes in get_cluster_sizes_from_pmap(
                pval, pthr, neighbors, area_map
            )
        ]
    elif len(pval.shape) == 1:
        return [np.max(get_cluster_sizes_from_pmap(pval, pthr, neighbors, area_map))]
