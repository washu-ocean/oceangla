from pathlib import Path
from collections import defaultdict
from collections.abc import Callable
from itertools import product

import pandas as pd
import nibabel as nib
from nilearn.image import concat_imgs
import re

from .formula import ModelMetadata
from .model import ModelInputs
from .utils import listify


def build_rows_from_paths(paths: list[Path]):
    attributes = defaultdict(list)
    for p in paths:
        pos = 0
        word = ""
        attributes["path"].append(str(p.resolve()))
        while pos < len(p.name):
            word += p.name[pos]
            match word:
                case "sub-":
                    word = ""
                    attributes["subject"].append("")
                    pos += 1
                    while p.name[pos] != '_':
                        attributes["subject"][-1] += p.name[pos]
                        pos += 1
                    pos += 1
                case "ses-":
                    word = ""
                    attributes["session"].append("")
                    pos += 1
                    while p.name[pos] != '_':
                        attributes["session"][-1] += p.name[pos]
                        pos += 1
                    pos += 1
                case "task-":
                    word = ""
                    attributes["task"].append("")
                    pos += 1
                    while p.name[pos] != '_':
                        attributes["task"][-1] += p.name[pos]
                        pos += 1
                    pos += 1
                case "space-":
                    word = ""
                    attributes["space"].append("")
                    pos += 1
                    while p.name[pos] != '_':
                        attributes["space"][-1] += p.name[pos]
                        pos += 1
                    pos += 1
                case "condition-":
                    word = ""
                    attributes["condition"].append("")
                    pos += 1
                    while p.name[pos] != '_':
                        attributes["condition"][-1] += p.name[pos]
                        pos += 1
                    pos += 1
                case _:
                    if word.endswith("boldmap"):
                        attributes["suffix"].append("")
                        pos += 1
                        while pos < len(p.name):
                            attributes["suffix"][-1] += p.name[pos]
                            pos += 1
                    else:
                        pos += 1
    return attributes


def build_path_df(fladirs: list[Path]) -> pd.DataFrame:
    if len(fladirs) == 0:
        raise ValueError("list of fladir paths cannot be empty")
    paths = []
    for fladir in fladirs:
        paths.extend(fladir.glob("sub-*/ses-*/func/*condition*stat-effect_boldmap*"))
    df_dict = build_rows_from_paths(paths)
    return pd.DataFrame(df_dict)


def build_indepvar_df(csvs: list[Path]) -> pd.DataFrame:
    if len(csvs) == 0:
        raise ValueError("list of .csv/.tsv files cannot be empty")
    dfs = []
    for csv in csvs:
        if csv.suffix == ".csv":
            sep = ","
        elif csv.suffix == ".tsv":
            sep = "\t"
        dfs.append(pd.read_csv(csv, sep=sep, dtype={"subject": str}))
    if len(dfs) == 1:
        return dfs[0]
    master_df = dfs[0]
    for i in range(1, len(dfs)):
        if not set(master_df['subject']).isdisjoint(dfs[i]['subject']):
            raise ValueError("All .csv/.tsv files must have a unique set of subjects.")
        master_df = pd.concat([master_df, dfs[i]], join="inner", ignore_index=True)
    return master_df


def zscore(series: pd.Series):
    return (series - series.mean()) / series.std()


def fetch_effects_and_design_matrices(
    path_df: pd.DataFrame,
    var_df: pd.DataFrame,
    meta: ModelMetadata,
    session: str | list[str] = None,
    task: str | list[str] = None,
    space: str | list[str] = None,
    normalization_strategy: Callable[[pd.Series], pd.Series] = zscore
) -> nib.Nifti1Image | nib.cifti2.cifti2.Cifti2Image:
    valid_suffixes = (".nii", ".nii.gz", ".dscalar.nii")
    session = listify(session)
    task = listify(task)
    space = listify(space)
    entity_combos = product(session, task, space)
    model_inputs_list = []
    scalars, conditions = [c.scalar for c in meta["contrasts"]], [c.condition for c in meta["contrasts"]]
    if meta["mixed_effects"]:
        fir_rgx = re.compile(rf"(?:{'|'.join([c.replace('_', '[-_]') for c in conditions])})-\d+")
    for session_, task_, space_ in entity_combos:
        condition_df = path_df[
            (path_df["session"] == session_) & (path_df["task"] == task_) & (path_df["space"] == space_) & (path_df["suffix"].isin(valid_suffixes))
        ]
        if meta["mixed_effects"]:  # Pull unassumed response models
            condition_df = condition_df[path_df["condition"].str.contains(fir_rgx, regex=True)]
            condition_df["fir_frame"] = condition_df["condition"].str.split("-").str[-1]
            condition_df["condition"] = condition_df["condition"].str.split("-").str[0:-1].str.join("-")
            condition_df["subject_count"] = condition_df["subject"].map(condition_df["subject"].value_counts())
            condition_df = condition_df[condition_df["subject_count"] == len(conditions)]
            condition_df = condition_df.sort_values(by=["subject", "condition", "fir_frame"]).reset_index(drop=True)
            assert all([
                condition_df["subject"].value_counts().nunique() == 1,
                condition_df["condition"].value_counts().nunique() == 1,
                condition_df["fir_frame"].value_counts().nunique() == 1,
                len(condition_df["suffix"].unique()) == 1,
            ]), "The group of filtered paths for an FIR group model must have a balanced number of subjects, conditions, and fir_frames. They must also have the same suffix."
        else:  # Pull assumed response models
            condition_df = (
                path_df[path_df["condition"].isin(conditions)]
            )
            condition_df["subject_count"] = condition_df["subject"].map(condition_df["subject"].value_counts())
            condition_df = condition_df[condition_df["subject_count"] == len(conditions)]
            condition_df = condition_df.sort_values(by=["subject", "condition"]).reset_index(drop=True)
            assert all([
                condition_df["subject"].value_counts().nunique() == 1,
                condition_df["condition"].value_counts().nunique() == 1,
                len(condition_df["suffix"].unique()) == 1,
            ]), "The group of filtered paths for an assumed-HRF group model must have a balanced number of subjects and conditions"
        suffix = condition_df["suffix"].unique()[0]
        match suffix:
            case ".nii" | ".nii.gz":  # NIFTI
                condition_df["path"]
