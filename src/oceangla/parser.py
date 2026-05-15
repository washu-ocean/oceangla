from argparse import ArgumentParser, ArgumentTypeError
from importlib import metadata
from pathlib import Path

MODEL_CHOICES = ("ols",)

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


def _get_parser():
    parser = ArgumentParser(
        prog="oceangla",
        description="Tool for group-level analysis of task-based fMRI"
    )
    parser.add_argument("--version", action="version", version=metadata.version('oceangla'))
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("-a", "--alpha",
                        nargs="+",
                        type=_pos_float,
                        dest="alphas",
                        default=[0.05],
                        help="Alpha level(s) that determine significance in statistical tests.")
    parser.add_argument("--model",
                        nargs="+",
                        dest="models")
    parser.add_argument("--parcellation_dlabel", "--parcellation-dlabel",
                        nargs="+",
                        type=Path,
                        dest="dlabel_paths",
                        help="Path(s) to .dlabel.nii file(s) containing parcellation schemes to run models on, in addition to dense models.")
    parser.add_argument("-f", "--fladir", "--fla_dir", "--fla-dir",
                        nargs="+",
                        type=_path_exists_as_dir,
                        dest="fladir_paths",
                        required=True,
                        help="Path to first-level analysis directory. Can specify multiple for different subject sets (provided the analyses are the same)")

    parser.add_argument("-o", "--outdir", "--out_dir", "--out-dir",
                        type=Path,
                        dest="outdir_path",
                        required=True,
                        help="Path to group-level model outputs.")
    parser.add_argument("-p", "--perms",
                        type=_pos_int_or_zero,
                        dest="perms",
                        default=0,
                        help="Number of permutations to use for FWER and/or cluster correction")
    areamap_group = parser.add_mutually_exclusive_group()
    areamap_group.add_argument("--preprocdir", "--preproc_dir", "--preproc-dir",
                               nargs="+",
                               type=Path,
                               dest="preprocdir_paths",
                               help="Path to folder containing minimally preprocessed outputs (including this generates an average area-per-vertex map for cluster correction. These can also be set with the --vertex-area-maps flag.).")
    areamap_group.add_argument("--vertex-area-maps", "--vertex_area_maps",
                               nargs=2,
                               metavar="LEFT_AREA_GII RIGHT_AREA_GII",
                               type=Path,
                               dest="vertex_area_map_paths",
                               help="Path to a left and right hemisphere (in that order!) denoting a vertex area map, which will scale cluster size on a per-vertex basis instead of denoting cluster sizes as just the number of vertices in a cluster.")
    parser.add_argument("--csv", "--tsv",
                        type=_indep_var_csv_or_tsv,
                        dest="var_path",
                        help="Path to .csv or .tsv file containing independent variables to include "
                        "in the model. IMPORTANT: column containing subject IDs must be titled "
                        "'subject' (program will throw an error if this is not true).")
    parser.add_argument("--session-name", "--session_name", "--sessionname",
                        nargs="+",
                        dest="session_name",
                        help="""\
                        One or more BIDS session names to use, in case there are multiple. If multiple
                        are specified, separate models will be ran for each depvar/indepvar pair for each session subset.
                        Here's an example where each BIDS session name is 'ses-01': ``--session-name 01``. Here's another
                        with different pre- and post- sessions: ``--session-name pre post``. By default, will run
                        for each unique session name.""")
    parser.add_argument("--save_db", "--save-db",
                        action="store_true",
                        dest="save_db",
                        help="Save database file in output directory specified by `-o`.")
    return parser


def parse_args():
    parser = _get_parser()
    args = parser.parse_args()
    from .config import config
    for k, v in args.__dict__.items():
        setattr(config, k, v)
