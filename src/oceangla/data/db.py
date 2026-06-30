import logging
import os
import re
import sqlite3
import time
from pathlib import Path

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
from ..formula import FormulaParser, Token, TokenType, is_scaled_value_node

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


def populate_db(db_path: Path,
                fladirs: list[Path],
                var_paths: list[Path],
                reindex: bool = False) -> Path:
    if db_path.is_file():
        if reindex or not __db_is_valid(db_path):
            db_path.unlink()
        else:
            return db_path
    logger.debug(
        f"{'Reindexing' if reindex else 'Creating'} sqlite db file at {db_path}"
    )
    with sqlite3.connect(db_path) as con:
        logger.debug("creating table")
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS subject_activation")
        cur.execute("""
        CREATE TABLE subject_activation(
            subject TEXT,
            session TEXT,
            task TEXT,
            path TEXT,
            condition TEXT,
            suffix TEXT,
            space TEXT,
            fladir TEXT
        );""")

        files_of_interest = []

        s = time.time()
        logger.debug("looking for files")
        for fladir in fladirs:
            files_of_interest.extend(
                fladir.glob("sub-*/ses-*/func/*condition*stat-effect_boldmap*")
            )
        e = time.time()
        logger.debug(f"took {e - s}s to find all files")
        del s
        del e

        row_regex = re.compile(r'sub-([a-zA-Z0-9]+)_ses-([a-zA-Z0-9]+)_task-([a-zA-Z0-9]+)_space-([a-zA-Z0-9\-]+)_condition-([a-zA-Z0-9\-]+)_*stat-effect_boldmap(.*)')

        def __build_path_row(p: Path) -> dict:
            row = {}
            row["path"] = str(p)
            row["fladir"] = str(p.parent.parent.parent.parent.resolve())
            (
                row["subject"],
                row["session"],
                row["task"],
                row["space"],
                row["condition"],
                row["suffix"]
            ) = re.search(row_regex, p.name).group(1,2,3,4,5,6)
            logger.debug(f"Built row for {p.resolve()!s}")
            return row

        logger.debug("building path rows")
        start_time = time.time()
        db_data = [__build_path_row(p) for p in files_of_interest]
        end_time = time.time()
        logger.debug(f"took {end_time - start_time}s to build rows for db")
        del start_time
        del end_time
        cur.executemany(
            """
            INSERT INTO subject_activation VALUES(:subject, :session, :task, :path, :condition, :suffix, :space, :fladir)
            """,
            db_data,
        )
        cur.execute("""
        CREATE TABLE subjects
        AS SELECT DISTINCT subject
        FROM subject_activation
        ORDER BY subject
        """)
        indepvar_dfs = [
            pd.read_csv(
                p, sep="," if p.suffix == ".csv" else "\t", dtype={"subject": str}
            )
            for p in var_paths
        ]
        all_column_types = {"subject": "TEXT UNIQUE NOT NULL"}
        for idx in range(len(indepvar_dfs)):
            if "subject" not in indepvar_dfs[idx].columns:
                raise ValueError(
                    f"Missing required column 'subject' from {var_paths[idx].resolve()!s}"
                )
            indepvar_dfs[idx] = indepvar_dfs[idx].dropna(subset=["subject"])
            indepvar_dfs[idx]["subject"].str.replace("sub-", "")
            for col in indepvar_dfs[idx].columns:
                if col == "subject":
                    continue
                elif indepvar_dfs[idx][col].dtype in (np.float64, np.int64, np.float32, np.int32):
                    indepvar_dfs[idx][f"{col}_ZSCORE"] = (indepvar_dfs[idx][col] - indepvar_dfs[idx][col].mean()) / indepvar_dfs[idx][col].std()
                    all_column_types[col] = "NUM"
                    all_column_types[f"{col}_ZSCORE"] = "REAL"
                else:
                    all_column_types[col] = "TEXT"
            indepvar_dfs[idx] = (
                indepvar_dfs[idx]
                .sort_values(by="subject")
                .reset_index(drop=True)
            )
        indepvar_df = pd.concat(indepvar_dfs, axis=0, join='outer')
        indepvar_df.to_sql(
            name="indepvar",
            con=con,
            if_exists="append",
            index=False,
        )
        con.commit()

    logger.debug("DB created successfully!")
    return db_path


def get_unique_conditions_as_list(db_path: str | Path) -> list[str]:
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        unique_conditions = [
            row[0]
            for row in cur.execute(
                "SELECT DISTINCT condition FROM subject_activation"
            ).fetchall()
        ]
        return unique_conditions


def get_activation_and_design_matrix(
    formula: str,
    db_path: str,
    space: str = "fsLR",
    task: str = None,
    session: str = None,
    memory: joblib.Memory | None = None
) -> tuple[pd.DataFrame, dict]:
    deptree, indeptree = FormulaParser(formula).tree[0], FormulaParser(formula).tree[1]
    all_conditions = []
    for node in deptree:
        if is_scaled_value_node(node):
            (_, _), condition = node
            all_conditions.append(condition.value)
    all_conditions = list(set(all_conditions))
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
        subject_subquery = """
        SELECT subject FROM subject_activation
        WHERE condition IN (
        """ + ','.join(["'%s'" % cond for cond in all_conditions]) + f"""
        )
        GROUP BY subject
        HAVING COUNT(DISTINCT condition) = {len(all_conditions)}
        """
        query = (
            f"""
            SELECT {','.join(column_queries)} FROM indepvar
            WHERE subject IN ({subject_subquery})
            AND {f' AND '.join([' %s IS NOT NULL ' % col for col in column_names])}
            ORDER BY indepvar.subject
            """
        )
        print(f"Running query:\n{query}")
        cur = con.cursor()
        print(cur.execute(query).fetchall())
        df = pd.read_sql_query(query, con)
    df["intercept"] = 1
    cols = ["intercept"] + [
        c for c in df.columns if c != "intercept"
    ]  # rearrange so intercept is first
    df = df[cols]
    logger.debug(f"queried indepvar rows: {len(df)}")
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
        WHERE subject IN (
            SELECT DISTINCT subject FROM indepvar
            WHERE {' AND '.join([' %s IS NOT NULL ' % col for col in column_names])}
        )
        AND (subject_activation.condition='{condition}' OR subject_activation.condition='{condition.replace("_", "-")}')
        AND subject_activation.space='{space}'
        """
        if task is not None:
            query += f" AND subject_activation.task='{task}' "
        if session is not None:
            query += f" AND subject_activation.session='{session}' "
        else:  # Try and get the most common session
            session, _ = cur.execute(
                """ SELECT session, COUNT(session) as frequency FROM subject_activation GROUP BY session ORDER BY frequency DESC LIMIT 1 """
            ).fetchone()
            query += f"AND subject_activation.session='{session}'"
        query += " ORDER BY subject_activation.subject"
        logger.debug(f"Running query:\n{query}")
        paths = [row[0] for row in cur.execute(query)]
        print(paths)
        logger.debug(f"queried activations: {len(paths)}")
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
