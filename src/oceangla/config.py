from pathlib import Path
from joblib import Memory


class config:  # All attributes are set in parser.py parse_args()
    alphas: list[float] = [0.05]
    db_path: str | None = None
    depvar: list[str] | None = None
    dlabel_paths: list[Path] | None = None
    fladir_paths: list[Path] | None = None
    indepvar: list[str] | None = None
    models: list[str] | None = None
    model_names: list[str] | None = None
    outdir_path: Path = Path("out/")
    reindex: bool = False
    perms: int = 0
    preprocdir_paths: list[Path] | None = None
    session_name: list[str] | None = None
    var_path: Path | None = None
    vertex_area_map_paths: tuple[Path, Path] | None = None
    version: str | None = None
    verbose: int = 0

    # Store joblib memory cache
    joblib_memory_path: Path | str | None = None
    joblib_memory: Memory | None = None
