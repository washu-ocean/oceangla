import logging
import sys
from importlib import metadata

from .config import config
from .data import collect_models_and_dataframes
from .model import run_models
from .parser import parse_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    logger.info(f"oceangla {metadata.version('oceangla')}")
    parse_args()

    if config.verbose:
        logger.setLevel(level=logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    run_models(*collect_models_and_dataframes())


if __name__ == "__main__":
    main()
