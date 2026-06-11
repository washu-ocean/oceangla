from pathlib import Path

from joblib import Memory


class config:  # All attributes are set in parser.py parse_args()
    alphas: list[float] = [0.05]
    db_path: str | None = None
    depvar: list[str] | None = None
    dlabel_paths: list[Path] | None = None
    fladir_paths: list[Path] | None = None
    indepvar: list[str] | None = None
    models: list[str] = []
    model_file: Path | None = None
    model_names: list[str] = []
    outdir_path: Path = None
    reindex: bool = False
    perms: int = 0
    preprocdir_paths: list[Path] | None = None
    session_name: list[str] | None = None
    var_paths: list[Path] | None = None
    vertex_area_map_paths: tuple[Path, Path] | None = None
    version: str | None = None
    verbose: int = 0
    volume_cluster_strategy: str = "NN1"

    # Store joblib memory cache
    joblib_memory_path: Path | str | None = None
    joblib_memory: Memory | None = None

    _paths = (
        "db_path",
        "dlabel_paths",
        "fladir_paths",
        "model_file",
        "outdir_path",
        "preprocdir_paths",
        "var_paths",
        "vertex_area_map_paths",
    )
