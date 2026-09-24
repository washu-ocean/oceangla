import logging
import os
import sys
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
import traceback

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
from .formula import FormulaParser, Token, TokenType, is_scaled_value_node
from .config import config

logger = logging.getLogger(__name__)


def __db_is_valid(db_path: Path) -> bool:
    query_table = "SELECT name FROM sqlite_master WHERE type='table' AND name='%s'"
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        # Check subject_activation table exists
        if cur.execute(query_table % "subject_activation").fetchone() is None:
            logger.warning("Table subject_activation not present in db, reindexing.")
            return False
        # Check indepvar table exists
        if cur.execute(query_table % "indepvar").fetchone() is None:
            logger.warning("Table indepvar not present in db, reindexing.")
            return False
    logger.info(
        f"Using database at {db_path.resolve()!s} (last modified {time.ctime(os.path.getmtime(str(db_path)))})"
    )
    logger.warning(
        "Run oceangla with the --reindex option if the contents of your FLA folder or "
        "variable .csv/.tsv files have changed."
    )
    return True

ROW_REGEX = re.compile(r'sub-([a-zA-Z0-9]+)_ses-([a-zA-Z0-9]+)_task-([a-zA-Z0-9]+)_space-([a-zA-Z0-9\-]+)_condition-([a-zA-Z0-9\-]+)_*stat-effect_boldmap(.*)')

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
            logger.info(f"{indepvar_tsv_path} already exists. Use the --reindex option to reindex FLA paths if they have changed.")
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
                f"{df["subject"].value_counts.loc(lambda x : x > 1)}"
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


def get_activation_and_design_matrix(
    formula: str,
    db_path: str,
    space: str = "fsLR",
    task: str = None,
    session: str = None,
    memory: joblib.Memory | None = None
) -> tuple[pd.DataFrame, dict]:
    deptree, indeptree = FormulaParser(formula).tree[0], FormulaParser(formula).tree[1]
    column_queries = []
    column_names = []

    def _eval_indep_node(node):
        if isinstance(node, Token) and node.type == TokenType.INTERCEPT:
            return
        elif is_scaled_value_node(node):
            (sign, scalar), varname = node
            sign, scalar, varname = sign.value, scalar.value, varname.value
            column_names.append(varname)
            column_queries.append(f"{sign}{scalar} * {varname}_ZSCORE AS {varname}")
        elif (
            isinstance(node, list) and node[0].type == TokenType.MUL
        ):  # full interaction
            for node2 in node[1:]:
                (sign, scalar), varname = node2
                sign, scalar, varname = sign.value, scalar.value, varname.value
                column_names.append(varname)
                if (
                    subquery := f"{sign}{scalar} * {varname}_ZSCORE AS {varname}"
                ) not in column_queries:
                    column_queries.append(subquery)
            column_queries.append(
                " * ".join(
                    [
                        f"({sign.value}{scalar.value} * {varname.value}_ZSCORE)"
                        for (sign, scalar), varname in node[1:]
                    ]
                )
            )
            column_queries[-1] += " AS interaction_" + "_".join(
                varname.value for (_, _), varname in node[1:]
            )
        elif (
            isinstance(node, list) and node[0].type == TokenType.INTERACTION
        ):  # just interaction term
            column_names.extend([varname.value for (_, _), varname in node[1:]])
            column_queries.append(
                " * ".join(
                    [
                        f"({sign.value}{scalar.value} * {varname.value}_ZSCORE)"
                        for (sign, scalar), varname in node[1:]
                    ]
                )
            )
            column_queries[-1] += " AS interaction_" + "_".join(
                varname.value for (_, _), varname in node[1:]
            )
        else:
            raise NotImplementedError(
                "Can only handle scaled nodes in depvar as of now"
            )

    for node in indeptree:
        _eval_indep_node(node)

    column_names = list(set(column_names))

    with sqlite3.connect(db_path) as con:
        query = (
            "SELECT "
            + ",".join(column_queries)
            + " FROM indepvar "
            + " INNER JOIN subjects ON subjects.subject = indepvar.subject WHERE "
            + " AND ".join([f" {col} IS NOT NULL " for col in column_names])
            + "ORDER BY indepvar.subject"
        )
        df = pd.read_sql_query(query, con)
    df["intercept"] = 1
    cols = ["intercept"] + [
        c for c in df.columns if c != "intercept"
    ]  # rearrange so intercept is first
    df = df[cols]
    activations = {}
    final_activation = {}

    if memory is None:
        _query_depvar = query_depvar
    else:
        _query_depvar = memory.cache(query_depvar)

    def _query_activation(condition, scalar=1) -> dict:
        activation = _query_depvar(
            condition, db_path, column_names, space, task, session
        )
        activation["activation"] *= scalar
        return activation

    def _eval_depvar_node(node):
        if is_scaled_value_node(node):
            (sign, scalar), condition = node
            sign, scalar, condition = sign.value, scalar.value, condition.value
            scalar_int = int(f"{sign}{scalar}")
            activations[condition] = _query_activation(condition, scalar=scalar_int)
            if not final_activation:
                for key in activations[condition].keys():
                    if key != "activation":
                        final_activation[key] = activations[condition][key]
            return condition
        else:
            raise NotImplementedError(
                "Can only handle scaled nodes in depvar as of now"
            )

    for node in deptree:
        _eval_depvar_node(node)

    final_activation["activation"] = np.squeeze(
        np.sum(
            np.concatenate(
                [
                    activation["activation"][np.newaxis, ...]
                    for activation in activations.values()
                ]
            ),
            axis=0,
        )
    )
    return df, final_activation


def query_depvar(
    condition,
    db_path: str,
    column_names: list[str],
    space: str = "fsLR",
    task: str = None,
    session: str = None,
) -> dict:
    activation = {"space": space}
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        query = f"""
        SELECT path FROM subject_activation
        INNER JOIN indepvar ON subject_activation.subject = indepvar.subject
        WHERE (subject_activation.condition='{condition}' OR subject_activation.condition='{condition.replace("_", "-")}')
        AND subject_activation.space='{space}'
        """
        for col in column_names:
            query += f"AND indepvar.{col} IS NOT NULL "
        if task is not None:
            query += f"AND subject_activation.task='{task}' "
        if session is not None:
            query += f"AND subject_activation.session='{session}' "
        else:  # Try and get the most common session
            session, _ = cur.execute(
                """ SELECT session, COUNT(session) as frequency FROM subject_activation GROUP BY session ORDER BY frequency DESC LIMIT 1 """
            ).fetchone()
            query += f"AND subject_activation.session='{session}'"
        query += " ORDER BY subject_activation.subject"
        logger.debug(f"Running query:\n{query}")
        paths = [row[0] for row in cur.execute(query)]
        try:
            first_img = nib.load(paths[0])
        except IndexError:
            print("Query failed.")
            print_unique_conditions(cur)
            print_unique_sessions(cur)
            print_unique_tasks(cur)
            print_unique_spaces(cur)
            exit()
        print("Loading activation...")
        if len(first_img.dataobj.shape) == 2:  # CIFTI
            activation["type"] = "CIFTI"
            activation["header"] = first_img.header
            activation["nifti_header"] = first_img.nifti_header
            activation["activation"] = np.concatenate(
                [nib.load(path).get_fdata() for path in paths], axis=0
            )
        elif len(first_img.dataobj.shape) == 3:  # NIFTI
            activation["type"] = "NIFTI"
            activation["affine"] = first_img.affine
            activation["header"] = first_img.header
            activation["activation"] = np.concatenate(
                [nib.load(path).get_fdata()[..., np.newaxis] for path in paths], axis=3
            )
        elif len(first_img.dataobj.shape) == 4:  # NIFTI
            activation["type"] = "NIFTI"
            activation["affine"] = first_img.affine
            activation["header"] = first_img.header
            activation["activation"] = np.concatenate(
                [nib.load(path).get_fdata() for path in paths], axis=3
            )
        else:
            raise ValueError(
                f"Number of axes for image at path {paths[0]} must be 2 (for CIFTI) 3, or 4 (for NIFTI), but contains {len(first_img.dataobj.shape)}"
            )
        return activation
