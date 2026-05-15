from joblib import Memory
import pandas as pd

from .config import config
from .data import populate_db, get_activation
from .utils import gen_dataset_description


def setup():
    # basic output directory structure
    config.outdir_path.mkdir(parents=True, exist_ok=True)
    config._cache_path = config.outdir_path / ".oceangla_cache"
    config._joblib_memory = Memory(config._cache_path)
    if not (ds_desc := (config.outdir_path / "dataset_description.json")).is_file():
        gen_dataset_description(ds_desc)
    del ds_desc

    # get models
    config._db_path = populate_db(config.fladir_paths)
    activation = get_activation(config._db_path, config.depvar)
    return activation
