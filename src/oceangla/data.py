import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import config

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
