import logging
import sqlite3


logger = logging.getLogger(__name__)


def prompt_space(db_path: str) -> str:
    unique_spaces = []
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        unique_spaces = [
            row[0]
            for row in cur.execute(
                "SELECT DISTINCT space FROM subject_activation"
            ).fetchall()
        ]
    assert len(unique_spaces) > 0, "No spaces found"
    if len(unique_spaces) == 1:
        logger.info(f"Running on template space {unique_spaces[0]}")
        return unique_spaces[0]
    idx_to_space = {
        str(i + 1): space
        for (i, space) in zip(range(len(unique_spaces)), unique_spaces)
    }

    def _get_space_input():
        for idx, space in unique_spaces.items:
            print(f"{idx}\t{space}")
        return input(
            "Please enter the number associated with the template space these analyses should run in: "
        )

    chosen_opt = None
    while chosen_opt is None:
        res = _get_space_input()
        if res.strip() not in idx_to_space.keys():
            print(f"Invalid option {res.strip()}, please choose a valid option.")
        else:
            chosen_opt = res
    return idx_to_space[chosen_opt]


def prompt_task(db_path: str) -> str:
    unique_tasks = []
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        unique_tasks = [
            row[0]
            for row in cur.execute(
                "SELECT DISTINCT task FROM subject_activation"
            ).fetchall()
        ]
    assert len(unique_tasks) > 0, "No tasks found"
    if len(unique_tasks) == 1:
        logger.info(f"Running on task {unique_tasks[0]}")
        return unique_tasks[0]
    idx_to_task = {
        str(i + 1): task for (i, task) in zip(range(len(unique_tasks)), unique_tasks)
    }

    def _get_task_input():
        for idx, task in unique_tasks.items:
            print(f"{idx}\t{task}")
        return input("Please enter the number associated with the appropriate task: ")

    chosen_opt = None
    while chosen_opt is None:
        res = _get_task_input()
        if res.strip() not in idx_to_task.keys():
            print(f"Invalid option {res.strip()}, please choose a valid option.")
        else:
            chosen_opt = res
    return idx_to_task[chosen_opt]
