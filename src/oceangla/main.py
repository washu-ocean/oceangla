import logging
import sys
from importlib import metadata

from .config import config
from .model import OLSModel
from .parser import parse_args
from .prompt import prompt_space, prompt_task
from .formula import parse_model_file

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
    from .data import (  # importing now to leverage caching activation data
        get_activation_and_design_matrix,
        populate_db,
    )

    if config.verbose:
        logger.setLevel(logging.DEBUG)

    config.db_path = populate_db(config.fladir_paths, reindex=config.reindex)
    if config.model_file is not None:
        file_model_names, file_models = parse_model_file(config.model_file)
        config.model_names.extend(file_model_names)
        config.models.extend(file_models)

    for model_name, model in zip(config.model_names, config.models):
        space = prompt_space(config.db_path)
        task = prompt_task(config.db_path)
        design_matrix, activation = get_activation_and_design_matrix(
            model, config.db_path, space=space, task=task
        )
        OLSModel(
            activation,
            design_matrix,
            model_desc=model_name,
            perms=config.perms,
            alpha=config.alphas,
        ).fit()


if __name__ == "__main__":
    main()
