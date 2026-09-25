import logging
import os
import sys
import re
import sqlite3
import time
from collections import defaultdict, namedtuple
from pathlib import Path
import traceback
from itertools import product
from copy import deepcopy

import nibabel as nib
import numpy as np
import pandas as pd
import joblib

from .error import (
    print_unique_conditions,
    print_unique_sessions,
    print_unique_spaces,
    print_unique_tasks,
)
from .formula import FormulaParser, Token, TokenType
from .config import config
from .model import ModelDesc


logger = logging.getLogger(__name__)

ROW_REGEX = re.compile(r'sub-([a-zA-Z0-9]+)_ses-([a-zA-Z0-9]+)_task-([a-zA-Z0-9]+)_space-([a-zA-Z0-9\-]+)_condition-([a-zA-Z0-9\-]+)_*stat-effect_boldmap(.*)')


def collect_models_and_dataframes() -> tuple[list[ModelDesc], pd.DataFrame, pd.DataFrame]:
    logger.info("Parsing model(s)...")
    models = [] # list of ModelDesc including space, task, session ids
    parsed_models = [] # list of ModelDesc after just parsing formulas
    for model_name, model_formula in zip(config.model_names, config.models):
        model_desc = FormulaParser(model_formula).model_desc
        model_desc["model_name"] = model_name
        parsed_models.append(model_desc)
    logger.info(f"{len(parsed_models)} found.")
    logger.info("Collecting subject variables...")
    config.subject_variables_path = populate_indepvar_tsv(
        config.outdir_path,
        config.var_paths,
        standardization=config.standardization_method,
        reindex=config.reindex
    )
    subject_variables_df = pd.read_csv(
        config.subject_variables_path,
        sep="\t",
        dtype={
            "subject": str
        }
    )
    logger.info("Collecting subject activation...")
    config.subject_activation_path = populate_subject_activation_tsv(
        config.outdir_path,
        config.fladir_paths,
        reindex=config.reindex
    )
    subject_activation_df = pd.read_csv(
        config.subject_activation_path,
        sep="\t",
        dtype={
            "subject": str,
            "session": str,
            "task": str,
            "space": str,
            "condition": str,
            "frame_no": int
        }
    )
    # Only include overlapping subjects in both dataframes
    subject_activation_df = subject_activation_df[subject_activation_df["subject"].isin(subject_variables_df["subject"])]
    subject_variables_df = subject_variables_df[subject_variables_df["subject"].isin(subject_activation_df["subject"])]
    unique_spaces = subject_activation_df["space"].unique()
    if len(config.space_ids) > 0:
        if (
            len((unique_spaces_ := 
                list(set(unique_spaces).intersection(config.space_ids))
            )) == 0
        ):
            raise ValueError(f"spaces {' '.join(config.space_ids)} not present in this dataset.")
        else:
            unique_spaces = unique_spaces_
    unique_tasks = subject_activation_df["task"].unique()
    if len(config.task_ids) > 0:
        if (
            len((unique_tasks_ := 
                list(set(unique_tasks).intersection(config.task_ids))
            )) == 0
        ):
            raise ValueError(f"tasks {' '.join(config.task_ids)} not present in this dataset.")
        else:
            unique_tasks = unique_tasks_
    unique_sessions = subject_activation_df["session"].unique()
    if len(config.session_ids) > 0:
        if (
            len((unique_sessions_ := 
                list(set(unique_sessions).intersection(config.session_ids))
            )) == 0
        ):
            raise ValueError(f"sessions {' '.join(config.session_ids)} not present in this dataset.")
        else:
            unique_sessions = unique_sessions_
    unique_combos = list(product(unique_spaces, unique_tasks, unique_sessions))
    for desc, combo in product(parsed_models, unique_combos):
        models.append(deepcopy(desc))
        (
            models[-1]["space"],
            models[-1]["task"],
            models[-1]["session"]
        ) = combo
    logger.info(f"Will run each model {len(unique_combos)} times for each (template space - task - session) combo:")
    for combo in unique_combos:
        logger.info(f"\t({' - '.join(combo)})")
    logger.info(f"Will run {len(models)} total group-level models.")
    del parsed_models, unique_combos, unique_spaces, unique_tasks, unique_sessions
    return (models, subject_activation_df, subject_variables_df)

   


def build_subject_activation_row_from_path(p: Path) -> dict | None:
    row = {}
    row["path"] = str(p)
    try:
        (
            row["subject"],
            row["session"],
            row["task"],
            row["space"],
            row["condition"],
            row["suffix"]
        ) = re.search(ROW_REGEX, p.name).group(1,2,3,4,5,6)
        if len(row["condition"].split("-")) > 1 and (frame_no_match := re.match(r'\d\d', row["condition"].split("-")[-1])):
            row["frame_no"] = int(frame_no_match.group())
            row["condition"] = row["condition"].removesuffix(f"-{frame_no_match.group()}")
        else:
            row["frame_no"] = -1  # not a frame in an FIR response
        logger.debug(f"Built row for {p.resolve()!s}")
        return row
    except AttributeError:
        logger.error(f"Could not build database row with path: {p}")
        logger.error("Attempted to use pattern: sub-([a-zA-Z0-9]+)_ses-([a-zA-Z0-9]+)_task-([a-zA-Z0-9]+)_space-([a-zA-Z0-9\\-]+)_condition-([a-zA-Z0-9\\-]+)_*stat-effect_boldmap(.*)")
        logger.error("Error found", exc_info=True)
        return None


def populate_subject_activation_tsv(
    output_dir: Path,
    fladirs: list[Path],
    reindex: bool = False
) -> Path:
    if (subject_activation_tsv_path := (output_dir / "subject_activation.tsv")).is_file():
        if reindex:
            logger.info(f"Removing {subject_activation_tsv_path} and reindexing paths.")
            subject_activation_tsv_path.unlink()
        else:
            logger.info(f"{subject_activation_tsv_path} already exists. Use the --reindex option to reindex FLA paths if they have changed.")
            return subject_activation_tsv_path
    files_of_interest = []
    for fladir in fladirs:
        files_of_interest.extend(
            fladir.glob("sub-*/ses-*/func/sub*condition*stat-effect_boldmap*")  # Include 'sub' at beginning of filename to avoid '._'-prefixed files
        )

    rows = [build_subject_activation_row_from_path(p) for p in files_of_interest]

    paths_with_no_row = []  # Print out paths that were not included
    for i in range(len(rows)):
        if rows[i] is None:
            paths_with_no_row.append(files_of_interest[i])
    if len(paths_with_no_row) > 0:
        logger.warning("Could not build rows for these paths:")
        logger.warning('\n'.join([str(p) for p in paths_with_no_row]))
    del paths_with_no_row

    rows = list(filter(None, rows))
    df = pd.DataFrame(rows)
    df.to_csv(subject_activation_tsv_path, sep="\t", index=False)
    return subject_activation_tsv_path


def populate_indepvar_tsv(
    output_dir: Path,
    var_paths: list[Path],
    categorical_columns: list[str] = [],
    standardization: str = "zscore",
    reindex: bool = False
) -> Path:
    standardization = standardization.lower()
    if standardization not in ("zscore", "meancenter", "none"):
        raise ValueError(f"Standardization method must be one of 'zscore', 'meancenter', or 'none'. Received {standardization}")

    if (indepvar_tsv_path := (output_dir / "indepvar.tsv")).is_file():
        if reindex:
            logger.info(f"Removing {indepvar_tsv_path} and reindexing paths.")
            indepvar_tsv_path.unlink()
        else:
            logger.info(f"{indepvar_tsv_path} already exists. Use the --reindex option to reindex subject-specific variables if they have changed.")
            return indepvar_tsv_path
        
    dfs = []
    for var_path in var_paths:
        if var_path.suffix not in (".csv", ".tsv"):
            raise ValueError(f"Subject variable files should either be a valid .csv or .tsv file. Received path: {var_path}")
        dfs.append(pd.read_csv(
            var_path,
            sep="," if var_path.suffix == ".csv" else "\t"
        ))
        df = dfs[-1]
        subj_col = df.columns[0] #always assume first column is subject column
        df.rename(columns={subj_col: "subject"}, inplace=True)
        df["subject"] = df["subject"].astype(str)
        if not len(df["subject"]) == len(df):
            raise ValueError(
                f"Variable file {var_path} has multiple rows for these subjects: \n"
                f"{df['subject'].value_counts.loc(lambda x : x > 1)}"
            )
        for col in df.columns[1:]:
            if df[col].dtype == "str" or col in categorical_columns:  # for N categories, break into N-1 columns
                unique_groups = df[col].unique()
                for i in range(len(unique_groups)-1):
                    df[f"{col}[{i}]"] = df[col] == unique_groups[i]
            else:  # continuous variable case
                if standardization == "zscore":
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif standardization == "meancenter":
                    df[col] = (df[col] - df[col].mean())
                elif standardization == "none":
                    continue
        
    df_ = pd.concat(dfs, ignore_index=True)
    df_.to_csv(indepvar_tsv_path, sep="\t", index=False)
    return indepvar_tsv_path