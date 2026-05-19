from importlib import metadata
import warnings
import logging

from joblib import Memory
import ipdb

from .config import config
from .data import populate_db
from .parser import parse_args, MODEL_CHOICES
from .utils import gen_dataset_description
from .model import *

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    logger.info(f"oceangla {metadata.version('oceangla')}")
    parse_args()
    if config.verbose:
        logger.setLevel(warnings.DEBUG)

    config.db_path = populate_db(config.fladir_paths)
    from .formula import get_activation, get_design_matrix
    for model in config.models:
        activation = get_activation(model, config.db_path)
        design_matrix = get_design_matrix(model, config.db_path)
        OLSModel(activation, design_matrix).fit()



if __name__ == "__main__":
    main()
