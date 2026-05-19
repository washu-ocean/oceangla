from pathlib import Path
import logging
import sqlite3
import re

import numpy as np
import nibabel as nib
from joblib import Memory, Parallel, delayed
import pandas as pd

from .config import config

logger = logging.getLogger(__name__)


def populate_db(fladirs: list[str] | list[Path] | str | Path) -> Path:
    if isinstance(fladirs, (str, Path)):
        fladirs = [fladirs]
    fladirs = [Path(d) for d in fladirs]

    db_path = config.outdir_path / ".oceangla.db"
    if db_path.is_file():
        db_path.unlink()
    logger.debug(f"Creating sqlite db file at {db_path}")
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
            files_of_interest.extend(fladir.glob("sub-*/ses-*/func/*condition*stat-effect_boldmap*"))

        db_data = (
            {
                "subject": re.search(r'sub-([a-zA-Z0-9]+)_', p.name).group(1),
                "session": re.search(r'ses-([a-zA-Z0-9]+)_', p.name).group(1),
                "task": re.search(r'task-([a-zA-Z0-9]+)_', p.name).group(1),
                "path": str(p),
                "condition": re.search(r'condition-([a-zA-Z0-9\-]+)_', p.name).group(1),
                "suffix": ''.join(p.suffixes),
                "space": re.search(r'space-([a-zA-Z0-9\-]+)_', p.name).group(1),
                "fladir": str(p.parent.parent.parent.parent.resolve())
            } for p in files_of_interest
        )
        cur.executemany("INSERT INTO subject_activation VALUES(:subject, :session, :task, :path, :condition, :suffix, :space, :fladir);", db_data)
        df = (
            pd.read_csv(config.var_path, sep="," if config.var_path.suffix == ".csv" else "\t")
            .sort_values(by="subject")
            .reset_index(drop=True)
        )
        if "subject" not in df.columns:
            raise ValueError(f"Missing required column 'subject' from {config.var_path.resolve()!s}")
        df.to_sql(name="indepvar", con=con)
        con.commit()

    logger.debug("DB created successfully!")
    return db_path
