from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path


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


def _csv_or_tsv(value):
    try:
        value = Path(value)
        if not any((
            value.name.endswith(".csv"),
            value.name.endswith(".tsv")
        )):
            raise ValueError()
        return value
    except ValueError as err:
        raise ArgumentTypeError(
            f"Expected a path to a .csv or .tsv file, received {value.resolve()!s}."
        ) from err


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


def _get_parser():
    parser = ArgumentParser(
        prog="oceangla",
        description="Tool for group-level analysis of task-based fMRI"
    )
    parser.add_argument("-a", "--alpha",
                        nargs="+",
                        type=_pos_float,
                        dest="alphas",
                        help="Alpha level(s) that determine significance in statistical tests.")
    parser.add_argument("--parcellation_dlabel", "--parcellation-dlabel",
                        nargs="+",
                        type=Path,
                        dest="dlabel_paths",
                        help="Path(s) to .dlabel.nii file(s) containing parcellation schemes to run models on, in addition to dense models.")
    parser.add_argument("-f", "--fladir", "--fla_dir", "--fla-dir",
                        nargs="+",
                        type=Path,
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
                        required=True,
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
                        nargs="+",
                        type=_csv_or_tsv,
                        dest="varcsv_paths",
                        help="Path to group-level model outputs.")
    return parser


def parse_args():
    parser = _get_parser()
    args = parser.parse_args()
    from .config import config
    for k, v in args.__dict__.items():
        setattr(config, k, v)
