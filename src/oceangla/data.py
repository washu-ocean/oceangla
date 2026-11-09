import logging
import re
from pathlib import Path
from itertools import product
from copy import deepcopy

import pandas as pd

from .formula import FormulaParser
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
    logger.info("Collecting subject contrasts...")
    config.firstlevel_contrasts_path = populate_firstlevel_contrasts_tsv(
        config.outdir_path,
        config.fladir_paths,
        reindex=config.reindex
    )
    firstlevel_contrasts_df = pd.read_csv(
        config.firstlevel_contrasts_path,
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
    if config.var_paths:
        logger.info("Collecting subject variables...")
        config.subject_variables_path = populate_subject_variables_tsv(
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
    else:   # Empty dataframe if no .csv/.tsv passed in
        subject_variables_df = (
            pd.DataFrame(
                {"subject": firstlevel_contrasts_df["subject"]},
                dtype=str
            )
            .drop_duplicates(ignore_index=True)
        )
    # Only include overlapping subjects in both dataframes
    firstlevel_contrasts_df = pd.merge(firstlevel_contrasts_df, subject_variables_df["subject"], how="inner", on="subject", sort=True).drop_duplicates()
    subject_variables_df = pd.merge(subject_variables_df, firstlevel_contrasts_df["subject"], how="inner", on="subject", sort=True).drop_duplicates()
    unique_spaces = firstlevel_contrasts_df["space"].unique()
    if len(config.space_ids) > 0:
        if (
            len((unique_spaces_ := 
                list(set(unique_spaces).intersection(config.space_ids))
            )) == 0
        ):
            raise ValueError(f"spaces {' '.join(config.space_ids)} not present in this dataset.")
        else:
            unique_spaces = unique_spaces_
    unique_tasks = firstlevel_contrasts_df["task"].unique()
    if len(config.task_ids) > 0:
        if (
            len((unique_tasks_ := 
                list(set(unique_tasks).intersection(config.task_ids))
            )) == 0
        ):
            raise ValueError(f"tasks {' '.join(config.task_ids)} not present in this dataset.")
        else:
            unique_tasks = unique_tasks_
    unique_sessions = firstlevel_contrasts_df["session"].unique()
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
    print(models)
    return (models, firstlevel_contrasts_df, subject_variables_df)

   


def build_firstlevel_contrasts_row_from_path(p: Path) -> dict | None:
    row = {}
    row["path"] = str(p)
    if p.suffix == ".json":
        return None
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


def populate_firstlevel_contrasts_tsv(
    output_dir: Path,
    fladirs: list[Path],
    reindex: bool = False
) -> Path:
    if (firstlevel_contrasts_tsv_path := (output_dir / "firstlevel_contrasts.tsv")).is_file():
        if reindex:
            logger.info(f"Removing {firstlevel_contrasts_tsv_path} and reindexing paths.")
            firstlevel_contrasts_tsv_path.unlink()
        else:
            logger.info(f"{firstlevel_contrasts_tsv_path} already exists. Use the --reindex option to reindex FLA paths if they have changed.")
            return firstlevel_contrasts_tsv_path
    files_of_interest = []
    for fladir in fladirs:
        files_of_interest.extend(
            fladir.glob("sub-*/ses-*/func/sub*condition*stat-effect_boldmap*")  # Include 'sub' at beginning of filename to avoid '._'-prefixed files
        )

    rows = [build_firstlevel_contrasts_row_from_path(p) for p in files_of_interest]

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
    df.to_csv(firstlevel_contrasts_tsv_path, sep="\t", index=False)
    return firstlevel_contrasts_tsv_path


def populate_subject_variables_tsv(
    output_dir: Path,
    var_paths: list[Path],
    categorical_columns: list[str] = [],
    standardization: str = "zscore",
    reindex: bool = False
) -> Path:
    standardization = standardization.lower()
    if standardization not in ("zscore", "meancenter", "none"):
        raise ValueError(f"Standardization method must be one of 'zscore', 'meancenter', or 'none'. Received {standardization}")

    if (subject_variables_tsv_path := (output_dir / "subject_variables.tsv")).is_file():
        if reindex:
            logger.info(f"Removing {subject_variables_tsv_path} and reindexing paths.")
            subject_variables_tsv_path.unlink()
        else:
            logger.info(f"{subject_variables_tsv_path} already exists. Use the --reindex option to reindex subject-specific variables if they have changed.")
            return subject_variables_tsv_path
        
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
    df_.to_csv(subject_variables_tsv_path, sep="\t", index=False)
    return subject_variables_tsv_path