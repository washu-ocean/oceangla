from importlib import metadata
import warnings
import logging

from joblib import Memory
import ipdb

from .config import config
from .data import populate_db 
from .parser import parse_args
from .utils import gen_dataset_description
from .formula import get_activation

logger = logging.getLogger(__name__)


def main():
    logger.info(f"oceangla {metadata.version('oceangla')}")
    parse_args()
    if config.verbose:
        logger.setLevel(warnings.DEBUG)

    config.db_path = populate_db(config.fladir_paths)
    for model in config.models:
        activation = get_activation(model, config.db_path)
        print(activation)
        print(activation.shape)
    # ipdb.set_trace()


# def run():
#     pass


if __name__ == "__main__":
    main()
