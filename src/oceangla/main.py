import logging
import sys
from importlib import metadata
from collections import defaultdict

from .config import config
from .data import populate_db,get_activation_and_design_matrix
from .model import OLSModel
from .parser import parse_args
from .prompt import prompt_space, prompt_task
from .formula import FormulaParser, TokenType

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

    config.db_path = populate_db(
        config.outdir_path / ".oceangla.db",
        config.fladir_paths,
        config.var_paths,
        reindex=config.reindex
    )
    model_params = defaultdict(dict)
    for model_name, model in zip(config.model_names, config.models):
        model_params[f"{model_name}_{model}"]["space"] = prompt_space(config.db_path)
        model_params[f"{model_name}_{model}"]["task"] = prompt_task(config.db_path)
    for model_name, model in zip(config.model_names, config.models):
        depvar = FormulaParser(model).tree[0]
        if (
            hasattr(depvar[0], "type")
            and depvar[0].type == TokenType.ALL_INDIVIDUAL_CONDITIONS
        ):
            from .data import get_unique_conditions_as_list

            all_conditions = get_unique_conditions_as_list(config.db_path)
            for condition in all_conditions:
                run_ols_model(
                    condition.replace("-", "_") + "~" + model.split("~")[-1].strip(),
                    model_name + f"_condition-{condition}",
                    model_params[f"{model_name}_{model}"]["space"],
                    model_params[f"{model_name}_{model}"]["task"]
                )
        else:
            run_ols_model(model, model_name, model_params[f"{model_name}_{model}"]["space"], model_params[f"{model_name}_{model}"]["task"])


def run_ols_model(model: str, model_name: str, space: str, task: str):
    design_matrix, activation = get_activation_and_design_matrix(
        model, config.db_path, space=space, task=task, memory=config.joblib_memory
    )
    OLSModel(
        activation,
        design_matrix,
        model_desc=model_name,
        perms=config.perms,
        alpha=config.alphas,
        volume_cluster_strategy=config.volume_cluster_strategy
    ).fit()


if __name__ == "__main__":
    main()
