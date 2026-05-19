from pathlib import Path


class config:  # All attributes are set in parser.py parse_args()
    alphas: list[float] = [0.05]
    db_path: str = None
    depvar: list[str] = None
    dlabel_paths: list[Path] = None
    fladir_paths: list[Path] = None
    indepvar: list[str] = None
    models: list[str] = None
    outdir_path: Path = None
    perms: int = 0
    preprocdir_paths: list[Path] = None
    session_name: list[str] = None
    var_path: Path = None
    vertex_area_map_paths: tuple[Path, Path] = None
    version: str = None
    verbose: int = 0

    # Store joblib memory cache
    joblib_memory_path = None
    joblib_memory = None
