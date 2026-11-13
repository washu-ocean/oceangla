import itertools
import tempfile
from pathlib import Path
import pytest

from oceangla.config import config
from oceangla.parser import _get_parser, get_parser, parse_args


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dirs():
    """Create temporary directories for fladir and outdir."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        fladir = base / "fla"
        fladir.mkdir()
        outdir = base / "out"
        outdir.mkdir()
        yield {"base": base, "fladir": fladir, "outdir": outdir}


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config state before and after each test."""
    config.fladir_paths = None
    config.outdir_path = None
    config.model_names = []
    config.models = []
    config.perms = 0
    config.alphas = [0.05]
    config.tfce = False
    config.fdr = True
    config.cluster_correct = True
    yield
    config.fladir_paths = None
    config.outdir_path = None
    config.model_names = []
    config.models = []


@pytest.fixture
def parser():
    return _get_parser()


# ===========================================================================
# 1. Basic Defaults and Required Arguments
# ===========================================================================

def test_parser_defaults(parser, tmp_dirs):
    """Verify default values when only required flags are supplied."""
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
    ])
    assert args.fladir_paths == [tmp_dirs["fladir"]]
    assert args.outdir_path == tmp_dirs["outdir"]
    assert args.alphas == [0.05]
    assert args.perms == 0
    assert args.fwhm == 0
    assert args.volume_cluster_strategy == "NN1"
    assert args.standardization_method == "zscore"
    assert args.models == []
    assert args.model_names == []
    assert args.space_ids == []
    assert args.task_ids == []
    assert args.session_ids == []
    assert args.var_paths == []
    assert not args.verbose
    assert not args.just_build_db


def test_missing_outdir_in_parse_args(tmp_dirs):
    """parse_args() should raise ValueError if -o is missing."""
    config.fladir_paths = None
    config.outdir_path = None
    with pytest.raises(ValueError, match="Must specify an -o/--outdir path"):
        parser = _get_parser()
        ns = parser.parse_args(["-f", str(tmp_dirs["fladir"])])
        parse_args(parsed_args=ns)


def test_missing_fladir_in_parse_args(tmp_dirs):
    """parse_args() should raise ValueError if -f is missing."""
    config.fladir_paths = None
    config.outdir_path = None
    with pytest.raises(ValueError, match="Must specify at least one -f/--fladir path"):
        parser = _get_parser()
        ns = parser.parse_args(["-o", str(tmp_dirs["outdir"])])
        parse_args(parsed_args=ns)


# ===========================================================================
# 2. TFCE Flag Permutations (User Requested)
# ===========================================================================

def test_tfce_defaults(parser, tmp_dirs):
    """TFCE should be disabled by default with standard default parameters."""
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
    ])
    assert args.tfce is False
    assert args.tfce_h == 2.0
    assert args.tfce_e == 0.5
    assert args.tfce_dh is None


@pytest.mark.parametrize("tfce_flag", ["--tfce"])
def test_tfce_flag_activation(parser, tmp_dirs, tfce_flag):
    """--tfce should activate TFCE correction."""
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        tfce_flag,
    ])
    assert args.tfce is True


@pytest.mark.parametrize("flag,val,attr,expected", [
    ("--tfce-h", "1.5", "tfce_h", 1.5),
    ("--tfce_h", "3.0", "tfce_h", 3.0),
    ("--tfce-e", "0.6", "tfce_e", 0.6),
    ("--tfce_e", "1.0", "tfce_e", 1.0),
    ("--tfce-dh", "0.01", "tfce_dh", 0.01),
    ("--tfce_dh", "0.1", "tfce_dh", 0.1),
])
def test_tfce_parameter_flags(parser, tmp_dirs, flag, val, attr, expected):
    """Verify --tfce-h, --tfce-e, --tfce-dh set the correct attributes."""
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        "--tfce",
        flag, val,
    ])
    assert getattr(args, attr) == expected


@pytest.mark.parametrize("flag,bad_val", [
    ("--tfce-h", "0"),
    ("--tfce-h", "-1.0"),
    ("--tfce-h", "not_a_num"),
    ("--tfce-e", "-0.5"),
    ("--tfce-dh", "0"),
])
def test_tfce_invalid_parameters(parser, tmp_dirs, flag, bad_val):
    """Negative, zero, or non-numeric TFCE exponents should fail."""
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            flag, bad_val,
        ])


# ===========================================================================
# 3. FDR & Cluster Correction Toggle Flags
# ===========================================================================

@pytest.mark.parametrize("fdr_flag,expected", [
    ("--fdr", True),
    ("--no-fdr", False),
    ("--no_fdr", False),
])
def test_fdr_flags(parser, tmp_dirs, fdr_flag, expected):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        fdr_flag,
    ])
    assert args.fdr is expected


def test_fdr_mutually_exclusive(parser, tmp_dirs):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "--fdr", "--no-fdr",
        ])


@pytest.mark.parametrize("clus_flag,expected", [
    ("--cluster-correct", True),
    ("--cluster_correct", True),
    ("--no-cluster-correct", False),
    ("--no_cluster_correct", False),
])
def test_cluster_correct_toggle(parser, tmp_dirs, clus_flag, expected):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        clus_flag,
    ])
    assert args.cluster_correct is expected


def test_cluster_correct_mutually_exclusive(parser, tmp_dirs):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "--cluster-correct", "--no-cluster-correct",
        ])


# ===========================================================================
# 4. Alpha Level Permutations
# ===========================================================================

@pytest.mark.parametrize("alpha_args,expected", [
    (["-a", "0.01"], [0.01]),
    (["--alpha", "0.05"], [0.05]),
    (["-a", "0.01", "0.05", "0.001"], [0.01, 0.05, 0.001]),
    (["--alpha", "1e-4", "0.1"], [0.0001, 0.1]),
])
def test_alpha_flag_variations(parser, tmp_dirs, alpha_args, expected):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        *alpha_args,
    ])
    assert args.alphas == expected


@pytest.mark.parametrize("bad_alpha", ["0", "-0.05", "-1", "abc", "0.0"])
def test_alpha_invalid(parser, tmp_dirs, bad_alpha):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "-a", bad_alpha,
        ])


# ===========================================================================
# 5. Permutation Count (-p / --perms)
# ===========================================================================

@pytest.mark.parametrize("flag,val,expected", [
    ("-p", "0", 0),
    ("-p", "100", 100),
    ("--perms", "5000", 5000),
    ("-p", "1", 1),
])
def test_perms_flag(parser, tmp_dirs, flag, val, expected):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        flag, val,
    ])
    assert args.perms == expected


@pytest.mark.parametrize("bad_perms", ["-1", "-10", "abc", "3.14"])
def test_perms_invalid(parser, tmp_dirs, bad_perms):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "-p", bad_perms,
        ])


# ===========================================================================
# 6. Cluster Strategy & Volume Neighbors
# ===========================================================================

@pytest.mark.parametrize("flag,strat", [
    ("--cluster_strategy", "NN1"),
    ("--cluster_strategy", "NN2"),
    ("--cluster_strategy", "NN3"),
    ("--cluster-strategy", "NN1"),
    ("--cluster-strategy", "NN2"),
    ("--cluster-strategy", "NN3"),
])
def test_cluster_strategy_valid(parser, tmp_dirs, flag, strat):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        flag, strat,
    ])
    assert args.volume_cluster_strategy == strat


@pytest.mark.parametrize("invalid_strat", ["NN4", "nn1", "face", "corner", "26"])
def test_cluster_strategy_invalid(parser, tmp_dirs, invalid_strat):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "--cluster-strategy", invalid_strat,
        ])


# ===========================================================================
# 7. Standardization Method Permutations
# ===========================================================================

@pytest.mark.parametrize("flag,method", [
    ("--standardization-method", "zscore"),
    ("--standardization-method", "meancenter"),
    ("--standardization-method", "none"),
    ("--standardization_method", "zscore"),
    ("--standardization_method", "meancenter"),
    ("--standardization_method", "none"),
])
def test_standardization_methods(parser, tmp_dirs, flag, method):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        flag, method,
    ])
    assert args.standardization_method == method


def test_standardization_method_invalid(parser, tmp_dirs):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "--standardization-method", "minmax",
        ])


# ===========================================================================
# 8. Mutually Exclusive Options
# ===========================================================================

def test_reindex_flags(parser, tmp_dirs):
    args1 = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        "--reindex",
    ])
    assert args1.reindex is True

    args2 = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        "--no-reindex",
    ])
    assert args2.reindex is False


def test_reindex_mutually_exclusive(parser, tmp_dirs):
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "--reindex", "--no-reindex",
        ])


def test_preprocdir_vs_vertex_area_maps(parser, tmp_dirs):
    area_l = tmp_dirs["base"] / "lh.area.gii"
    area_r = tmp_dirs["base"] / "rh.area.gii"
    area_l.touch()
    area_r.touch()

    # Alone: preprocdir works
    args1 = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        "--preprocdir", str(tmp_dirs["base"]),
    ])
    assert args1.preprocdir_paths == [tmp_dirs["base"]]

    # Alone: vertex-area-maps works
    args2 = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        "--vertex-area-maps", str(area_l), str(area_r),
    ])
    assert args2.vertex_area_map_paths == [area_l, area_r]

    # Together: should fail mutual exclusivity
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "--preprocdir", str(tmp_dirs["base"]),
            "--vertex-area-maps", str(area_l), str(area_r),
        ])


# ===========================================================================
# 9. CSV / TSV Validation
# ===========================================================================

@pytest.mark.parametrize("flag,ext", [
    ("--csv", ".csv"),
    ("--tsv", ".tsv"),
])
def test_csv_tsv_valid(parser, tmp_dirs, flag, ext):
    f = tmp_dirs["base"] / f"pheno{ext}"
    f.touch()
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        flag, str(f),
    ])
    assert args.var_paths == [f]


def test_csv_tsv_invalid_extension(parser, tmp_dirs):
    f = tmp_dirs["base"] / "pheno.txt"
    f.touch()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "-f", str(tmp_dirs["fladir"]),
            "-o", str(tmp_dirs["outdir"]),
            "--csv", str(f),
        ])


# ===========================================================================
# 10. Filter Identifiers (space, task, session)
# ===========================================================================

@pytest.mark.parametrize("flag,values,attr", [
    ("--space", ["MNI152NLin2009cAsym", "fsLR"], "space_ids"),
    ("--template-space", ["MNI152NLin2009cAsym"], "space_ids"),
    ("--task", ["rest", "nback"], "task_ids"),
    ("--task-id", ["gambling"], "task_ids"),
    ("--session", ["ses-1", "ses-2"], "session_ids"),
    ("--session-name", ["ses-baseline"], "session_ids"),
])
def test_filter_identifiers(parser, tmp_dirs, flag, values, attr):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        flag, *values,
    ])
    assert getattr(args, attr) == values


# ===========================================================================
# 11. Null Distribution Grouping Flags
# ===========================================================================

def test_null_grouping_flags(parser, tmp_dirs):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        "--join-null-across-parameters",
        "--join-null-across-hemispheres",
    ])
    assert args.separate_null_by_parameter is False
    assert args.separate_null_by_hemisphere is False


@pytest.mark.parametrize("grouping", [
    "across_conditions_and_structures",
    "across_conditions_within_structures",
    "within_conditions_across_structures",
    "within_conditions_within_structures",
])
def test_cluster_size_grouping(parser, tmp_dirs, grouping):
    args = parser.parse_args([
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        "--cluster_size_grouping", grouping,
    ])
    assert args.cluster_grouping_strategy == grouping


# ===========================================================================
# 12. Full Permutation Matrix of Combined Flags
# ===========================================================================

# Combinations of flags across multiple orthogonal dimensions
PERM_ALPHA = [["-a", "0.05"], ["-a", "0.01", "0.05"]]
PERM_PERMS = [["-p", "0"], ["-p", "100"]]
PERM_STRAT = [["--cluster-strategy", "NN1"], ["--cluster-strategy", "NN3"]]
PERM_TFCE = [[], ["--tfce"], ["--tfce", "--tfce-h", "2.5", "--tfce-e", "0.6"]]
PERM_FDR = [[], ["--no-fdr"]]
PERM_CLUS = [[], ["--no-cluster-correct"]]
PERM_STD = [["--standardization-method", "zscore"], ["--standardization-method", "meancenter"]]


@pytest.mark.parametrize("alpha_opt,perms_opt,strat_opt,tfce_opt,fdr_opt,clus_opt,std_opt",
    list(itertools.product(
        PERM_ALPHA,
        PERM_PERMS,
        PERM_STRAT,
        PERM_TFCE,
        PERM_FDR,
        PERM_CLUS,
        PERM_STD,
    ))[:30]  # Sample 30 rich orthogonal combinations
)
def test_argparse_orthogonal_permutations(
    parser, tmp_dirs,
    alpha_opt, perms_opt, strat_opt, tfce_opt, fdr_opt, clus_opt, std_opt
):
    """Test rich orthogonal combinations of flags parse without error and set correct attributes."""
    cmd = [
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(tmp_dirs["outdir"]),
        *alpha_opt,
        *perms_opt,
        *strat_opt,
        *tfce_opt,
        *fdr_opt,
        *clus_opt,
        *std_opt,
    ]
    args = parser.parse_args(cmd)

    # Check alpha
    assert len(args.alphas) == (len(alpha_opt) - 1)

    # Check perms
    assert args.perms == int(perms_opt[1])

    # Check strategy
    assert args.volume_cluster_strategy == strat_opt[1]

    # Check TFCE
    if "--tfce" in tfce_opt:
        assert args.tfce is True
        if "--tfce-h" in tfce_opt:
            assert args.tfce_h == 2.5
            assert args.tfce_e == 0.6
    else:
        assert args.tfce is False

    # Check FDR
    if "--no-fdr" in fdr_opt:
        assert args.fdr is False
    else:
        assert args.fdr is True

    # Check cluster correction
    if "--no-cluster-correct" in clus_opt:
        assert args.cluster_correct is False
    else:
        assert args.cluster_correct is True

    # Check standardization
    assert args.standardization_method == std_opt[1]


# ===========================================================================
# 13. Integration with parse_args() and config Singleton
# ===========================================================================

def test_parse_args_updates_config(tmp_dirs):
    """parse_args(args=...) should update the global config singleton and create config.toml."""
    outdir = tmp_dirs["outdir"] / "run1"
    args_list = [
        "-f", str(tmp_dirs["fladir"]),
        "-o", str(outdir),
        "-p", "50",
        "-a", "0.01",
        "--tfce",
        "--tfce-h", "3.0",
        "--cluster-strategy", "NN2",
        "--standardization-method", "none",
        "--no-fdr",
        "--no-cluster-correct",
    ]
    parse_args(args=args_list)

    assert config.outdir_path == outdir
    assert config.perms == 50
    assert config.alphas == [0.01]
    assert config.tfce is True
    assert config.tfce_h == 3.0
    assert config.volume_cluster_strategy == "NN2"
    assert config.standardization_method == "none"
    assert config.fdr is False
    assert config.cluster_correct is False

    # Output directory and config.toml should be written
    assert outdir.is_dir()
    assert (outdir / "config.toml").is_file()
