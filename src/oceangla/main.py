import logging
import sys
import warnings
from importlib import metadata


from .config import config
from .data import populate_db
from .model import OLSModel
from .parser import parse_args
from .prompt import prompt_space, prompt_task

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    logger.info(f"oceangla {metadata.version('oceangla')}")
    parse_args()
    if config.verbose:
        logger.setLevel(warnings.DEBUG)

    config.db_path = populate_db(config.fladir_paths, reindex=config.reindex)
    from .formula import (
        get_activation,
        get_design_matrix,
    )  # importing now to leverage caching activation data

    for model_name, model in zip(config.model_names, config.models):
        space = prompt_space(config.db_path)
        task = prompt_task(config.db_path)
        activation = get_activation(model, config.db_path, space=space, task=task)
        design_matrix = get_design_matrix(model, config.db_path)
        OLSModel(
            activation,
            design_matrix,
            model_desc=model_name,
            perms=config.perms,
            alpha=config.alphas,
        ).fit()


if __name__ == "__main__":
    main()
