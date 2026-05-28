import json
import logging
import tomllib
from argparse import ArgumentParser, ArgumentTypeError, RawTextHelpFormatter
from importlib import metadata
from pathlib import Path
from textwrap import dedent

from joblib import Memory
from tomlkit import aot, document, dumps, item, loads, table

from .formula import parse_model_file
from ._version import __version__

MODEL_CHOICES = ("ols",)

logger = logging.getLogger(__name__)


def _pos_int_or_zero(value):
    try:
        value = int(value)
        if value < 0:
            raise ValueError()
        return value
    except ValueError as err:
        raise ArgumentTypeError(
            f"Expected either 0 or a positive integer, received {value!s}."
        ) from err


def _indep_var_csv_or_tsv(value):
    value = Path(value)
    if value.suffix not in (".csv", ".tsv"):
        raise ArgumentTypeError(
            f"--csv or --tsv file must end in .csv or .tsv, received file {value.resolve()}"
        )
    return value


def _pos_float(value):
    try:
        value = float(value)
        if value <= 0:
            raise ValueError()
        return value
    except ValueError as err:
        raise ArgumentTypeError(
            f"Expected a positive float, received {value!s}."
        ) from err


def _path_exists_as_dir(value):
    value = Path(value)
    if not value.is_dir():
        raise ArgumentTypeError(f"Directory {value.resolve()!s} should exist.")
    return value


def _path_exists_as_file(value):
    value = Path(value)
    if not value.is_file():
        raise ArgumentTypeError(f"File {value.resolve()!s} should exist.")
    return value


def _get_parser():
    parser = ArgumentParser(
        prog="oceangla",
        description="Tool for group-level analysis of task-based fMRI",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "-a",
        "--alpha",
        nargs="+",
        type=_pos_float,
        dest="alphas",
        default=[0.05],
        help="Alpha level(s) that determine significance in statistical tests.",
    )
    parser.add_argument("--model",
                        action="append",
                        metavar=("FORMULA"),
                        dest="models",
                        default=[])
    parser.add_argument("--model-file",
                        "--model_file",
                        dest="model_file",
                        type=Path,
                        help=dedent("""\
                        Path to a .txt file containing one model specifier and one formula on each line.
                        The model name should come first, enclosed in <> brackets, then the formula should appear
                        after. Example file contents:

                        correct_main_effect_anxiety -> correct ~ anx_score
                        incorrect_main_effect_anxiety -> incorrect ~ anx_score
                        correct_minus_incorrect_main_effect_anxiety -> correct - incorrect ~ anx_score
                        """))
    parser.add_argument(
        "--model_name",
        "--model-name",
        action="append",
        metavar=("MODEL_NAME"),
        dest="model_names",
        default=[],
        help="A short identifier for each model you specify. There must be as many names as there are models defined, "
        "and they will be assigned in the same order as models are specified. The outputs for each model will be stored "
        "in a folder named after this, under the folder specified by `-o`.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Recreate the sqlite database of first-level outputs if one "
        "already exists.",
    )
    parser.add_argument(
        "-c","--config",
        type=Path,
        help="Path to a config .toml file."
    )
    parser.add_argument(
        "--parcellation_dlabel",
        "--parcellation-dlabel",
        nargs="+",
        type=Path,
        dest="dlabel_paths",
        help="Path(s) to .dlabel.nii file(s) containing parcellation schemes to run models on, in addition to dense models.",
    )
    parser.add_argument(
        "--run-models-on-parcels-only",
        action="store_true",
        dest="run_models_on_parcels_only",
        help="If --parcellation-dlabel is specified, only run models on parcellated first-level outputs instead of vertex-wise.",
    )
    parser.add_argument(
        "-f",
        "--fladir",
        "--fla_dir",
        "--fla-dir",
        nargs="+",
        type=_path_exists_as_dir,
        dest="fladir_paths",
        help="Path to first-level analysis directory. Can specify multiple for different subject sets (provided the analyses are the same)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        "--out_dir",
        "--out-dir",
        type=Path,
        dest="outdir_path",
        help="Path to group-level model outputs.",
    )
    parser.add_argument(
        "-p",
        "--perms",
        type=_pos_int_or_zero,
        dest="perms",
        default=0,
        help="Number of permutations to use for FWER and/or cluster correction",
    )
    areamap_group = parser.add_mutually_exclusive_group()
    areamap_group.add_argument(
        "--preprocdir",
        "--preproc_dir",
        "--preproc-dir",
        nargs="+",
        type=Path,
        dest="preprocdir_paths",
        help="Path to folder containing minimally preprocessed outputs (including this generates an average area-per-vertex map for cluster correction. These can also be set with the --vertex-area-maps flag.).",
    )
    areamap_group.add_argument(
        "--vertex-area-maps",
        "--vertex_area_maps",
        nargs=2,
        metavar="LEFT_AREA_GII RIGHT_AREA_GII",
        type=Path,
        dest="vertex_area_map_paths",
        help="Path to a left and right hemisphere (in that order!) denoting a vertex area map, which will scale cluster size on a per-vertex basis instead of denoting cluster sizes as just the number of vertices in a cluster.",
    )
    parser.add_argument(
        "--csv",
        "--tsv",
        type=_indep_var_csv_or_tsv,
        nargs="+",
        dest="var_paths",
        help="Path to .csv or .tsv file containing independent variables to include "
        "in the model. IMPORTANT: column containing subject IDs must be titled "
        "'subject' (program will throw an error if this is not true).",
    )
    parser.add_argument(
        "--session-name",
        "--session_name",
        "--sessionname",
        nargs="+",
        dest="session_name",
        help="""\
                        One or more BIDS session names to use, in case there are multiple. If multiple
                        are specified, separate models will be ran for each depvar/indepvar pair for each session subset.
                        Here's an example where each BIDS session name is 'ses-01': ``--session-name 01``. Here's another
                        with different pre- and post- sessions: ``--session-name pre post``. By default, will run
                        for each unique session name.""",
    )
    return parser


def parse_args():
    parser = _get_parser()
    args = parser.parse_args()
    from .config import config

    if args.config is not None:
        toml_data = loads(args.config.read_text())
        for k, v in toml_data["config"].items():
            if k in config._paths:
                if isinstance(v, list):
                    setattr(config, k, [Path(p) for p in v])
                elif isinstance(v, str):
                    setattr(config, k, Path(v))
            else:
                setattr(config, k, v)
        for model_spec in toml_data["model_spec"]:
            if model_spec["name"] not in config.model_names and model_spec["formula"] not in config.models:
                config.model_names.append(model_spec["name"])
                config.models.append(model_spec["formula"])
    for k, v in args.__dict__.items():
        if v not in (None, []):
            setattr(config, k, v)
    if config.outdir_path is None:
        raise ValueError("Must specify an -o/--outdir path.")
    elif config.fladir_paths in (None, []):
        raise ValueError("Must specify at least one -f/--fladir path.")
    if args.model_file is not None:
        file_model_names, file_models = parse_model_file(config.model_file)
        for file_model_name, file_model in zip(file_model_names, file_models):
            if file_model_name not in config.model_names and file_model not in config.models:
                config.model_names.append(file_model_name)
                config.models.append(file_model)
            else:
                logger.warning(f"Duplicate model found between config and cmdline: {file_model_name} -> {file_model}")
    if not config.outdir_path.is_dir():
        logger.info(
            f"Outdir not found, creating new outdir at: {config.outdir_path.resolve()!s}"
        )
        config.outdir_path.mkdir(parents=True, exist_ok=False)
    config.joblib_memory_path = config.outdir_path / ".oceangla_memory"
    config.joblib_memory = Memory(config.joblib_memory_path)
    serializable = {}
    for k, v in config.__dict__.items():
        try:
            if k.startswith("_") or k in ("model_names", "models"):
                continue
            if k in config._paths:
                if isinstance(v, list):
                    serializable[k] = [str(p.resolve()) for p in v]
                elif isinstance(v, Path):
                    serializable[k] = str(v.resolve())
            else:
                json.dumps(v)  # check if object is serializable (i.e. joblib.Memory is not)
                serializable[k] = v
        except TypeError:
            continue
    doc = document()
    config_tab = table()
    for k, v in serializable.items():
        if v is not None:
            config_tab.add(k, v)
    # out_toml_dict = {k: v for k, v in serializable.items() if v is not None}
    doc["config"] = config_tab
    model_spec_aot = aot()
    for mn, m in zip(config.model_names, config.models):
        model_spec_aot.append(item({"name": mn, "formula": m}))
    doc.append("model_spec", model_spec_aot)
    with open(out_config_path := (config.outdir_path / "config.toml"), "w") as f:
        f.write(dumps(doc))
        logger.info(f"Wrote {out_config_path.resolve()!s}")
