from pathlib import Path
import logging

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

CLUSTER_CORR_METHODS = (
    "weighted_surf_and_volume",
    "weighed_surf_only",
    "nonweighted_surf_and_volume",
    "nonweighted_surf_only",
    "volume_only",
    "none"
)


def cifti_contains_surface_and_volume_component(activation: dict) -> bool:
    """
    Helper function to determine cluster correction algorithm
    """
    return (
        activation["type"] == "CIFTI" and
        len([
            str(struct[0]) for struct in super().activation["header"].get_axis(1).iter_structures()
            if str(struct[0]) not in ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT")
        ]) > 0
    )


# TODO: figure out if we eventually want separate lists of 'biggest' clusters per-hemisphere,
# instead of one big list containing the max sizes from both
class ClusterCorrectionMixin:
    def __init__(self):
        if hasattr(self, "l_surf_path") and hasattr(self, "r_surf_path"):
            self.__get_faces()
            self.l_numverts, self.r_numverts = (
                int(np.max(self.l_faces)),
                int(np.max(self.r_faces))
            )
            self.__build_adjacency_from_faces(self.l_numverts, self.l_faces)
            if hasattr(self, "l_area_path") and hasattr(self, "r_area_path"):
                self.l_area = nib.load(self.l_area_path).darrays[0].data
                self.r_area = nib.load(self.r_area_path).darrays[0].data

    def __get_faces(self):
        if not hasattr(self, "l_surf_path") or not hasattr(self, "r_surf_path"):
            return
        l_surf_img, r_surf_img = nib.load(self.l_surf_path), nib.load(self.r_surf_path)
        if not isinstance(l_surf_img, nib.gifti.GiftiImage):
            raise TypeError(f"Not a GIFTI surface: {self.l_surf_path}")
        elif not isinstance(r_surf_img, nib.gifti.GiftiImage):
            raise TypeError(f"Not a GIFTI surface: {self.r_surf_path}")
        l_faces, r_faces = l_surf_img.darrays[1].data, r_surf_img.darrays[1].data
        self.l_faces = l_faces
        self.r_faces = r_faces

    def __build_adjacency_from_faces(self):
        l_neigh = [set() for _ in range(self.l_numverts)]
        l_faces = self.l_faces.astype(np.int64)
        for a, b, c in l_faces:
            l_neigh[a].add(b)
            l_neigh[a].add(c)
            l_neigh[b].add(a)
            l_neigh[b].add(c)
            l_neigh[c].add(a)
            l_neigh[c].add(b)
        self.l_neigh = [np.fromiter(sorted(s), dtype=np.int32) for s in l_neigh]
        r_neigh = [set() for _ in range(self.r_numverts)]
        r_faces = self.r_faces.astype(np.int64)
        for a, b, c in r_faces:
            r_neigh[a].add(b)
            r_neigh[a].add(c)
            r_neigh[b].add(a)
            r_neigh[b].add(c)
            r_neigh[c].add(a)
            r_neigh[c].add(b)
        self.r_neigh = [np.fromiter(sorted(s), dtype=np.int32) for s in r_neigh]

    def __extract_hemi_values(self):
        """
        Calling this expects the following:

        - the parent class is in a state where `self.perm_pvals` is an `np.ndarray`
        - `self.activation["type"]` == `"CIFTI"` and `self.activation["header"]` exists
        """
        bm_axis = self.activation["header"].get_axis(1)
        L_vals = np.full((*values_full.shape[:-1], nL), np.nan, dtype=np.float32)
        R_vals = np.full((*values_full.shape[:-1], nR), np.nan, dtype=np.float32)
        seen_structures = []
        for (name, slc, bmodel) in bm_axis.iter_structures():
            if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
                seen_structures.append(name)
                vidx = bmodel.vertex.astype(np.int64)
                if np.max(vidx) >= nL:
                    raise RuntimeError(f"Left surface has {nL} verts but CIFTI references vertex {np.max(vidx)}. Wrong L_surf?")
                L_vals[..., vidx] = values_full[..., slc].astype(np.float32)
            elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
                seen_structures.append(name)
                vidx = bmodel.vertex.astype(np.int64)
                if np.max(vidx) >= nR:
                    raise RuntimeError(f"Right surface has {nR} verts but CIFTI references vertex {np.max(vidx)}. Wrong R_surf?")
                R_vals[..., vidx] = values_full[..., slc].astype(np.float32)
            if len(seen_structures) == 2:
                break

        return L_vals, R_vals


