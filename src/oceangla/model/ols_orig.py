#!/usr/bin/env python
# coding: utf-8

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from scipy import stats
import nibabel as nib
import numpy as np
import pandas as pd
import progressbar
# import ipdb
from joblib import Memory, Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection

SCRIPT_DIR = Path(__file__).parent
cachedir = SCRIPT_DIR / ".ols_v2_cache"
cachedir.mkdir(parents=True, exist_ok=True)
memory = Memory(cachedir)


L_SURF = "/data/sylvester/data1/ref/atlas/surf/conte/Conte69.L.midthickness.32k_fs_LR.surf.gii"
R_SURF = "/data/sylvester/data1/ref/atlas/surf/conte/Conte69.R.midthickness.32k_fs_LR.surf.gii"
L_AREA = SCRIPT_DIR / "surface/L_area.func.gii"
R_AREA = SCRIPT_DIR / "surface/R_area.func.gii"

CSV_PATH = SCRIPT_DIR / "combined_scores.csv"
# CATEGORICAL_COLUMNS = ("childs_gender", "Race", "childethnic")  # Break these out into columns in design mat


def _formula_var_list(value) -> list[str]:
    formula_str = str(value)
    formula_tokens = formula_str.split(' ')
    var_list = []
    cur_sign = ''
    for tok in formula_tokens:
        if tok == '-':
            cur_sign = tok
            continue
        elif tok == '+':
            cur_sign = ''
            continue
        else:
            var_list.append(f"{cur_sign}{tok}")
            cur_sign = ''
    return var_list


def get_faces_from_gifti_surf(surf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    g = nib.load(str(surf_path))
    if not isinstance(g, nib.gifti.GiftiImage):
        raise TypeError(f"Not a GIFTI surface: {surf_path}")
    faces = g.darrays[1].data
    return faces


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


def get_parser():
    parser = ArgumentParser(
        prog="ols.py",
        description="Run group-level OLS model with given variables"
    )
    parser.add_argument("-o", "--outdir",
                        type=Path,
                        default=Path(__file__).parent / "models",
                        help="Path to output directory",)
    parser.add_argument("-f", "--fladir",
                        type=Path,
                        default=Path("/data/sylvester/data1/datasets/aaa/derivatives/fla_assumed_robust/"),
                        help="Path to FLA directory",)
    parser.add_argument("-c", "--csv_path",
                        type=Path,
                        default=CSV_PATH,
                        help="Path to FLA directory",)
    parser.add_argument("-a", "--alpha",
                        type=float,
                        default=0.05,
                        help="Maximum p-value to use for defining clusters.",)
    parser.add_argument("-p", "--perms",
                        type=int,
                        default=0,
                        help="Number of permutations used to perform cluster correction -- default 0 (no permutations)")
    return parser


def extract_hemi_values(values_full: np.ndarray,
                        hdr: nib.cifti2.cifti2.Cifti2Header,
                        nL: int,
                        nR: int) -> tuple[np.ndarray, np.ndarray]:
    bm_axis = hdr.get_axis(1)
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


def get_cluster_index_groups(mask: np.ndarray, neighbors: list[np.ndarray]) -> list[np.ndarray]:
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


def get_cluster_sizes_from_pmap(pval: np.ndarray, pthr: float, neighbors: list[np.ndarray], area_map: np.ndarray) -> list[int]:
    if len(pval.shape) == 2:
        cluster_sizes_2d = []
        for beta in range(pval.shape[0]):
            thresholded_mask = np.squeeze(np.isfinite(pval[beta, :]) & (pval[beta, :] < pthr))
            cluster_idx_groups = get_cluster_index_groups(thresholded_mask, neighbors)
            cluster_sizes_2d.append(np.array([np.sum(area_map[cluster_idx_group]) for cluster_idx_group in cluster_idx_groups]))
        return cluster_sizes_2d
    elif len(pval.shape) == 1:
        thresholded_mask = np.isfinite(pval) & (pval < pthr)
        cluster_idx_groups = get_cluster_index_groups(thresholded_mask, neighbors)
        return np.array([np.sum(area_map[cluster_idx_group]) for cluster_idx_group in cluster_idx_groups])


def get_biggest_clusters_from_pmap(pval: np.ndarray, pthr: float, neighbors: list[np.ndarray], area_map: np.ndarray) -> list[int]:
    if len(pval.shape) == 2:
        return np.array([np.max(cluster_sizes) for cluster_sizes in get_cluster_sizes_from_pmap(pval, pthr, neighbors, area_map)])
    elif len(pval.shape) == 1:
        return np.max(get_cluster_sizes_from_pmap(pval, pthr, neighbors, area_map))


def _fdr_correct_1d(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    pvals_ = pvals.copy()
    pvals_[np.isnan(pvals_)] = 1
    pvals_sortind = np.argsort(pvals_)
    pvals_sorted = np.take(pvals_, pvals_sortind)
    ecdffactor = np.arange(1, len(pvals_sorted) + 1) / float(len(pvals_sorted))
    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1
    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[pvals_sortind] = pvals_corrected
    del pvals_corrected
    return pvals_corrected_


def fdr_correct(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    '''
    FDR-correct a pval map
    '''
    if pvals.ndim == 2:
        qvals = np.empty_like(pvals)
        for beta in range(pvals.shape[0]):
            qvals[beta, :] = _fdr_correct_1d(pvals[beta, :], alpha)
        return qvals
    elif pvals.ndim == 1:
        return _fdr_correct_1d(pvals)
    else:
        raise np.exceptions.AxisError("We should only have 1 or 2 axes in our pvalue map.")


def run_ols(design_matrix_df: pd.DataFrame,
            activation: np.ndarray,
            hdr: nib.cifti2.cifti2.Cifti2Header,
            out_folder: Path,
            model_desc: str,
            perms: int = 0,
            alpha: float = 0.05,
            condition: str = "oddball"):
    '''
    Run an OLS model.
    '''
    short_condition_name = condition.replace('condition-', '').replace('_stat-effect', '')
    l_faces, r_faces = get_faces_from_gifti_surf(L_SURF), get_faces_from_gifti_surf(R_SURF)
    l_numverts, r_numverts = int(np.max(l_faces)) + 1, int(np.max(r_faces)) + 1
    l_neigh, r_neigh = build_adjacency_from_faces(l_numverts, l_faces), build_adjacency_from_faces(r_numverts, r_faces)
    l_area, r_area = nib.load(L_AREA).darrays[0].data, nib.load(R_AREA).darrays[0].data
    good_indices = [i for i in range(activation.shape[0]) if not np.isnan(activation[i, :]).all()]
    design_matrix_df = design_matrix_df.drop([design_matrix_df.index[i] for i in range(len(design_matrix_df)) if i not in good_indices])
    activation = activation[good_indices, :]
    good_indices = design_matrix_df.reset_index().index[~design_matrix_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]  # drop subjects with ANY nan values
    if "subid" in design_matrix_df.columns:
        design_matrix_df = design_matrix_df.drop(columns="subid")
    design_matrix_arr = (design_matrix_df.reset_index(drop=True).loc[good_indices].to_numpy())
    activation = activation[good_indices, :]
    print(f"Condition: {short_condition_name}")
    value_names = tuple(design_matrix_df.reset_index(drop=True).columns)
    perm_max_clus_sizes = np.empty((perms + 1, len(value_names) * 2))  # store largest cluster sizes from both hemis on each iteration
    for perm in range(perms + 1):
        pvals = np.empty((design_matrix_arr.shape[1], activation.shape[1]))
        tstats = np.empty((design_matrix_arr.shape[1], activation.shape[1]))
        betas = np.empty((design_matrix_arr.shape[1], activation.shape[1]))
        ses = np.empty((design_matrix_arr.shape[1], activation.shape[1]))
        if perm < perms:
            print(f"Perm {perm+1} for {short_condition_name}")
            permuted_design_matrix_arr = design_matrix_arr.copy()
            for col in range(permuted_design_matrix_arr.shape[1]):
                np.random.shuffle(permuted_design_matrix_arr[:, col])
        else:
            print(f"Uncorrected OLS for {short_condition_name}")
            permuted_design_matrix_arr = design_matrix_arr
        for vtx in progressbar.progressbar(range(activation.shape[1]), redirect_stdout=True):
            n, p = permuted_design_matrix_arr.shape
            beta, ssr, rank, s = np.linalg.lstsq(
                permuted_design_matrix_arr,
                activation[:, vtx],
                rcond=None
            )
            sigma_sq = ssr[0] / (n - p)
            v_cov = np.linalg.inv(permuted_design_matrix_arr.T @ permuted_design_matrix_arr) * sigma_sq
            se = np.sqrt(np.diag(v_cov))
            tstat = beta / se
            pval = np.array([2 * (1 - stats.t.cdf(np.abs(t), df=n - p)) for t in tstat])
            for value_arr, value_vec in ((betas, beta), (ses, se), (tstats, tstat), (pvals, pval)):
                value_arr[:, vtx] = value_vec
        # perm_pvals[:, perm * n_top_pvals:(perm * n_top_pvals) + n_top_pvals] = np.sort(pvals, axis=1)[:, n_top_pvals]
        l_pvals, r_pvals = extract_hemi_values(pvals, hdr, l_numverts, r_numverts)
        perm_max_clus_sizes[perm, :len(value_names)] = get_biggest_clusters_from_pmap(l_pvals, alpha, l_neigh, l_area)
        perm_max_clus_sizes[perm, len(value_names):] = get_biggest_clusters_from_pmap(r_pvals, alpha, r_neigh, r_area)
        if perm == perms:  # save out uncorrected results
            betas_cifti = nib.cifti2.cifti2.Cifti2Image(betas, (nib.cifti2.cifti2_axes.ScalarAxis(value_names), hdr.get_axis(1)))
            nib.save(betas_cifti, p := out_folder / f"{short_condition_name}_{model_desc}_betas.dscalar.nii")
            print(f"saved {p}")
            del betas_cifti
            ses_cifti = nib.cifti2.cifti2.Cifti2Image(ses, (nib.cifti2.cifti2_axes.ScalarAxis(value_names), hdr.get_axis(1)))
            nib.save(ses_cifti, p := out_folder / f"{short_condition_name}_{model_desc}_ses.dscalar.nii")
            print(f"saved {p}")
            del ses_cifti
            tstats_cifti = nib.cifti2.cifti2.Cifti2Image(tstats, (nib.cifti2.cifti2_axes.ScalarAxis(value_names), hdr.get_axis(1)))
            nib.save(tstats_cifti, p := out_folder / f"{short_condition_name}_{model_desc}_tstats.dscalar.nii")
            print(f"saved {p}")
            del tstats_cifti
            uncorr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(pvals, (nib.cifti2.cifti2_axes.ScalarAxis(value_names), hdr.get_axis(1)))
            nib.save(uncorr_pvals_cifti, p := out_folder / f"{short_condition_name}_{model_desc}_uncorr_pvals.dscalar.nii")
            print(f"saved {p}")
            del uncorr_pvals_cifti
            if perms > 0:
                # cluster correction
                l_clus_corr = np.ones((len(value_names), l_numverts), dtype=np.float32)
                r_clus_corr = np.ones((len(value_names), r_numverts), dtype=np.float32)
                for value_idx in range(len(value_names)):
                    l_mask = np.isfinite(l_pvals[value_idx, :]) & (l_pvals[value_idx, :] < alpha)
                    for cluster in get_cluster_index_groups(l_mask, l_neigh):
                        cluster_size = np.sum(l_area[cluster])
                        sizes_larger_than_this_cluster = np.float32(np.sum(perm_max_clus_sizes[:, value_idx] >= cluster_size))
                        l_clus_corr[value_idx, cluster] = sizes_larger_than_this_cluster / (perms + 1)
                    r_mask = np.isfinite(r_pvals[value_idx, :]) & (r_pvals[value_idx, :] < alpha)
                    for cluster in get_cluster_index_groups(r_mask, r_neigh):
                        cluster_size = np.sum(r_area[cluster])
                        sizes_larger_than_this_cluster = np.float32(np.sum(perm_max_clus_sizes[:, value_idx + len(value_names)] >= cluster_size))
                        r_clus_corr[value_idx, cluster] = sizes_larger_than_this_cluster / (perms + 1)
                full_clus_corr = np.full_like(pvals, np.nan)
                for (name, slc, bmodel) in hdr.get_axis(1).iter_structures():
                    if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
                        vidx = bmodel.vertex.astype(np.int64)
                        full_clus_corr[:, slc] = l_clus_corr[:, vidx]
                    elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
                        vidx = bmodel.vertex.astype(np.int64)
                        full_clus_corr[:, slc] = r_clus_corr[:, vidx]
                    else:
                        full_clus_corr[:, slc] = 1.0  # not clustered here
                clus_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(full_clus_corr, (nib.cifti2.cifti2_axes.ScalarAxis(value_names), hdr.get_axis(1)))
                nib.save(clus_corr_pvals_cifti, p := out_folder / f"{short_condition_name}_{model_desc}_clus_corr_pvals.dscalar.nii")
                print(f"saved {p}")
                del full_clus_corr
                del l_clus_corr
                del r_clus_corr
                del clus_corr_pvals_cifti
            # fdr correction with Benjamini-Hochberg (for independent or positively-correlated tests)
            fdr_corr_pvals = np.empty_like(pvals)
            for i in range(pvals.shape[0]):
                pval_vec = pvals[i, :].copy()
                pval_vec[np.isnan(pval_vec)] = 1
                _, fdr_corr_pvals[i, :] = fdrcorrection(pval_vec, alpha=0.05, method="indep")
            fdr_corr_pvals_cifti = nib.cifti2.cifti2.Cifti2Image(fdr_corr_pvals, (nib.cifti2.cifti2_axes.ScalarAxis(value_names), hdr.get_axis(1)))
            nib.save(fdr_corr_pvals_cifti, p := out_folder / f"{short_condition_name}_{model_desc}_fdr_corr_pvals.dscalar.nii")
            print(f"saved {p}")
            del fdr_corr_pvals
            del fdr_corr_pvals_cifti


@memory.cache
def get_ols_activation(fladir: str | Path,
                       subjects: list[str],
                       condition: str,
                       suffix: str = ".dscalar.nii") -> np.ndarray:
    fladir = Path(fladir)

    def _find_dscalar(sub):
        try:
            if not sub.startswith('sub-'):
                sub = f"sub-{sub}"
            sub_fla_dir = fladir / sub
            if not sub_fla_dir.is_dir():
                sub_fla_dir = fladir / sub.replace("sub-", "")
            sub_activation_map = next(sub_fla_dir.rglob(f"{sub}*{condition}*{suffix}"))
            print(f'Found a .dscalar map for sub {sub}, '
                  f'condition {condition} in {sub_fla_dir}')
            return nib.load(sub_activation_map)
        # if a map is not found
        except StopIteration:
            print(f'Could not find .dscalar map for sub {sub}, '
                  f'condition {condition} in {sub_fla_dir}')
            return None
        # if it couldn't load for some reason (this shouldn't happen)
        except nib.filebasedimages.ImageFileError:
            print(f'Error loading .dscalar map for sub {sub}, '
                  f'condition {condition} in {sub_fla_dir}')
            return None

    potential_subs = len(subjects)
    imgs = Parallel(n_jobs=10)(delayed(_find_dscalar)(sub) for sub in subjects)
    valid_indices = [i for i in range(len(imgs)) if imgs[i] is not None]
    imgs = list(filter(None, imgs))
    activation = np.concatenate([img.get_fdata() for img in imgs], axis=0)
    subs_found = int(activation.shape[0])
    print(f"Found {subs_found} out of {potential_subs} subjects mentioned in the "
          f"input CSV at: {CSV_PATH}")
    return activation, imgs[0].header, valid_indices


@memory.cache
def get_ols_design_matrix(csv_path: str | Path, ols_vars: list[str]) -> pd.DataFrame:
    csv_df = pd.read_csv(csv_path)
    design_matrix_dict = defaultdict(list)
    design_matrix_dict["subid"] = csv_df["subid"]
    csv_df = csv_df.set_index("subid")
    for sub in csv_df.index:
        design_matrix_dict["intercept"].append(1)
        for ols_var in ols_vars:
            design_matrix_dict[ols_var].append(csv_df.loc[sub, ols_var])
    design_matrix_df = pd.DataFrame(design_matrix_dict)
    design_matrix_df.dropna(inplace=True)
    return design_matrix_df


def main():
    args = get_parser().parse_args()
    out_folder = args.outdir
    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be stored in {out_folder.resolve()!s}")

    for condition in (
        "condition-block-start_stat-effect",
        "condition-block_stat-effect",
        "condition-correct_stat-effect",
        "condition-incorrect_stat-effect",
        "condition-no-response_stat-effect",
    ):
        for varlist, model_desc in (
            (["adhd_score"], "adhd_maineffect"),
            (["anx_score"], "anx_maineffect"),
            (["anx_score", "adhd_score", "adhd_score_x_anx_score"], "adhd_anx_interaction"),
        ):
            design_matrix_df = get_ols_design_matrix(CSV_PATH, varlist)
            activation, hdr, valid_indices = get_ols_activation(args.fladir, [str(s) for s in design_matrix_df["subid"]], condition)
            design_matrix_df = design_matrix_df.reset_index().loc[valid_indices].reset_index(drop=True).drop(columns=["index", "subid"])
            run_ols(design_matrix_df,
                    activation,
                    hdr,
                    out_folder,
                    model_desc,
                    perms=args.perms,
                    alpha=args.alpha,
                    condition=condition)

if __name__ == "__main__":
    main()
