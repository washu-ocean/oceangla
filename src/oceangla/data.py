import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import nibabel as nib

from .config import config
from .error import print_unique_conditions, print_unique_tasks, print_unique_sessions, print_unique_spaces
from .formula import Token, TokenType, FormulaParser, is_scaled_value_node

logger = logging.getLogger(__name__)


# TODO: call this validate_db() function if we don't want to reindex,
# just to make sure all required tables are present
#
# Maybe we should have a table containing the fla directories that were
# indexed, and check against that before deciding not to reindex?
def validate_db():
    pass


def __build_path_row(p: Path) -> dict:
    keys_to_check = (
        "path",
        "suffixes",
        "fladir",
        "subject",
        "session",
        "task",
        "condition",
        "space",
    )
    row = defaultdict(str)
    row["path"] = str(p)
    row["suffixes"] = "".join(p.suffixes)
    row["fladir"] = str(p.parent.parent.parent.parent.resolve())
    if match := re.search(r"sub-([a-zA-Z0-9]+)_", p.name):
        row["subject"] = match.group(1)
    if match := re.search(r"ses-([a-zA-Z0-9]+)_", p.name):
        row["session"] = match.group(1)
    if match := re.search(r"task-([a-zA-Z0-9]+)_", p.name):
        row["task"] = match.group(1)
    if match := re.search(r"condition-([a-zA-Z0-9\-]+)_", p.name):
        row["condition"] = match.group(1)
    if match := re.search(r"space-([a-zA-Z0-9\-]+)_", p.name):
        row["space"] = match.group(1)
    if not all((k in row.keys() for k in keys_to_check)):
        raise ValueError(
            f"Could not index path {p.resolve()} "
            f"in database -- missing keys {','.join([k for k in keys_to_check if k not in row.keys()])}"
        )
    logger.debug(f"Inserted {p.resolve()!s}")
    return row


def populate_db(fladirs: list[Path], reindex: bool = False) -> Path:
    db_path = config.outdir_path / ".oceangla.db"
    if db_path.is_file():
        if reindex:
            db_path.unlink()
        else:
            return db_path
    logger.debug(
        f"{'Reindexing' if reindex else 'Creating'} sqlite db file at {db_path}"
    )
    with sqlite3.connect(db_path) as con:
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

        for fladir in fladirs:
            files_of_interest.extend(
                fladir.glob("sub-*/ses-*/func/*condition*stat-effect_boldmap*")
            )

        db_data = (__build_path_row(p) for p in files_of_interest)
        cur.executemany(
            "INSERT INTO subject_activation VALUES(:subject, :session, :task, :path, :condition, :suffix, :space, :fladir);",
            db_data,
        )
        indepvar_dfs = [
            pd.read_csv(p, sep="," if p.suffix == ".csv" else "\t")
            for p in config.var_paths
        ]
        columns_to_keep = set.intersection(*[set(df.columns) for df in indepvar_dfs])
        for idx in range(len(indepvar_dfs)):
            if "subject" not in indepvar_dfs[idx].columns:
                raise ValueError(
                    f"Missing required column 'subject' from {config.var_paths[idx].resolve()!s}"
                )
            indepvar_dfs[idx]["subject"] = indepvar_dfs[idx]["subject"].astype(str)
            indepvar_dfs[idx]["subject"].str.replace("sub-", "")
            indepvar_dfs[idx] = (
                indepvar_dfs[idx]
                .drop(
                    columns=[
                        column
                        for column in indepvar_dfs[idx].columns
                        if column not in columns_to_keep
                    ]
                )
                .sort_values(by="subject")
                .reset_index(drop=True)
            )
        for df in indepvar_dfs:
            df.to_sql(
                name="indepvar",
                con=con,
                if_exists="append",
                index=False,
            )
        con.commit()

    logger.debug("DB created successfully!")
    return db_path


def get_activation_and_design_matrix(
    formula: str,
    db_path: str,
    space: str = "fsLR",
    task: str = None,
    session: str = None,
) -> tuple[pd.DataFrame, dict]:
    deptree, indeptree = FormulaParser(formula).tree[0], FormulaParser(formula).tree[1]
    columns_to_query = []

    def _eval_indep_node(node):
        if isinstance(node, Token) and node.type == TokenType.INTERCEPT:
            return
        elif is_scaled_value_node(node):
            (sign, scalar), varname = node
            sign, scalar, varname = sign.value, scalar.value, varname.value
            columns_to_query.append(f"{sign}{scalar} * {varname} AS {varname}")
        elif (
            isinstance(node, list) and node[0].type == TokenType.MUL
        ):  # full interaction
            for node2 in node[1:]:
                (sign, scalar), varname = node2
                sign, scalar, varname = sign.value, scalar.value, varname.value
                if (
                    subquery := f"{sign}{scalar} * {varname} AS {varname}"
                ) not in columns_to_query:
                    columns_to_query.append(subquery)
            columns_to_query.append(
                " * ".join(
                    [
                        f"({sign.value}{scalar.value} * {varname.value})"
                        for (sign, scalar), varname in node[1:]
                    ]
                )
            )
            columns_to_query[-1] += " AS interaction_" + "_".join(
                varname.value for (_, _), varname in node[1:]
            )
        elif (
            isinstance(node, list) and node[0].type == TokenType.INTERACTION
        ):  # just interaction term
            columns_to_query.append(
                " * ".join(
                    [
                        f"({sign.value}{scalar.value} * {varname.value})"
                        for (sign, scalar), varname in node[1:]
                    ]
                )
            )
            columns_to_query[-1] += " AS interaction_" + "_".join(
                varname.value for (_, _), varname in node[1:]
            )
        else:
            raise NotImplementedError(
                "Can only handle scaled nodes in depvar as of now"
            )

    for node in indeptree:
        _eval_indep_node(node)

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        is_not_null_condition = " AND ".join(
            [f"indepvar.{v.split()[-1].strip()} IS NOT NULL" for v in columns_to_query]
        )
        is_null_condition = is_not_null_condition.replace(
            "IS NOT NULL", "IS NULL"
        ).replace("AND", "OR")
        if len(cur.execute('SELECT name FROM sqlite_master WHERE type = "view" AND name = "subs_with_all_variables"').fetchall()) == 0:
            cur.execute(f"""
            CREATE VIEW subs_with_all_variables AS
            SELECT DISTINCT indepvar.subject FROM indepvar INNER JOIN subject_activation ON subject_activation.subject = indepvar.subject
            WHERE {is_not_null_condition}
            """)
        if len(cur.execute('SELECT name FROM sqlite_master WHERE type = "view" AND name = "subs_without_all_variables"').fetchall()) == 0:
            cur.execute(f"""
            CREATE VIEW subs_without_all_variables AS
            SELECT DISTINCT indepvar.subject FROM indepvar INNER JOIN subject_activation ON subject_activation.subject = indepvar.subject
            WHERE {is_null_condition}
            """)
        if (
            len(
                subs_without_variables := [
                    row[0]
                    for row in cur.execute(
                        "SELECT subject FROM subs_without_all_variables"
                    ).fetchall()
                ]
            )
            > 0
        ):
            logger.warning(
                "Subjects who have missing data for one or more independent variables: "
                + ",".join(subs_without_variables)
            )
        query = (
            "SELECT "
            + ",".join(columns_to_query)
            + " FROM indepvar WHERE subject IN subs_with_all_variables ORDER BY subject"
        )
        df = pd.read_sql_query(query, con)
    df["intercept"] = 1
    cols = ["intercept"] + [
        c for c in df.columns if c != "intercept"
    ]  # rearrange so intercept is first
    df = df[cols]
    activations = {}
    final_activation = {}

    def _query_activation(condition, scalar=1) -> dict:
        activation = query_depvar(condition, db_path, space, task, session)
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


@config.joblib_memory.cache
def query_depvar(
    condition, db_path: str, space: str = "fsLR", task: str = None, session: str = None
) -> dict:
    activation = {"space": space}
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        query = f"""
        SELECT path FROM subject_activation
        WHERE subject IN subs_with_all_variables AND (condition='{condition}' OR condition='{condition.replace("_", "-")}')
        AND space='{space}'
        """
        if task is not None:
            query += f"AND task='{task}' "
        if session is not None:
            query += f"AND session='{session}' "
        else:  # Try and get the most common session
            session, _ = cur.execute(
                """ SELECT session, COUNT(session) as frequency FROM subject_activation GROUP BY session ORDER BY frequency DESC LIMIT 1 """
            ).fetchone()
            query += f"AND session='{session}'"
        query += " ORDER BY subject_activation.subject"
        print(f"Running query:\n{query}")
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
