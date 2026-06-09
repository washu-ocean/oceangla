from pathlib import Path
import pytest
from ..data import populate_db


@pytest.fixture(scope="session")
def fla_dir_one_task_one_space(tmp_path_factory):
    fla_dir = tmp_path_factory.mktemp("fla_dir_one_task_one_space")
    for i in range(10):
        (sub_dir := (fla_dir / f"sub-{i}" / "ses-01" / "func")).mkdir(parents=True)
        (sub_dir / f"sub-{i}_ses-01_task-TASK1_space-fsLR_condition-TESTCONDITION_stat-effect_boldmap.dscalar.nii").touch()
    return fla_dir


@pytest.fixture(scope="session")
def fla_dir_one_task_two_spaces(tmp_path_factory):
    fla_dir = tmp_path_factory.mktemp("fla_dir_one_task_two_spaces")
    for i in range(10):
        (sub_dir := (fla_dir / f"sub-{i}" / "ses-01" / "func")).mkdir(parents=True)
        (sub_dir / f"sub-{i}_ses-01_task-TASK1_space-fsLR_condition-TESTCONDITION_stat-effect_boldmap.dscalar.nii").touch()
        (sub_dir / f"sub-{i}_ses-01_task-TASK1_space-MNI152NLin6Asym_condition-TESTCONDITION_stat-effect_boldmap.nii.gz").touch()
    return fla_dir


@pytest.fixture(scope="session")
def fla_dir_two_tasks_one_space(tmp_path_factory):
    fla_dir = tmp_path_factory.mktemp("fla_dir_two_tasks_one_space")
    for i in range(10):
        (sub_dir := (fla_dir / f"sub-{i}" / "ses-01" / "func")).mkdir(parents=True)
        (sub_dir / f"sub-{i}_ses-01_task-TASK1_space-fsLR_condition-TESTCONDITION_stat-effect_boldmap.dscalar.nii").touch()
        (sub_dir / f"sub-{i}_ses-01_task-TASK2_space-fsLR_condition-TESTCONDITION_stat-effect_boldmap.dscalar.nii").touch()
    return fla_dir


@pytest.fixture(scope="session")
def fla_dir_two_tasks_two_spaces(tmp_path_factory):
    fla_dir = tmp_path_factory.mktemp("fla_dir_two_tasks_two_spaces")
    for i in range(10):
        (sub_dir := (fla_dir / f"sub-{i}" / "ses-01" / "func")).mkdir(parents=True)
        (sub_dir / f"sub-{i}_ses-01_task-TASK1_space-fsLR_condition-TESTCONDITION_stat-effect_boldmap.dscalar.nii").touch()
        (sub_dir / f"sub-{i}_ses-01_task-TASK2_space-fsLR_condition-TESTCONDITION_stat-effect_boldmap.dscalar.nii").touch()
        (sub_dir / f"sub-{i}_ses-01_task-TASK1_space-MNI152NLin6Asym_condition-TESTCONDITION_stat-effect_boldmap.nii.gz").touch()
        (sub_dir / f"sub-{i}_ses-01_task-TASK2_space-MNI152NLin6Asym_condition-TESTCONDITION_stat-effect_boldmap.nii.gz").touch()
    return fla_dir


@pytest.fixture(scope="session")
def var_csv(tmp_path_factory):
    var_csv = tmp_path_factory.mktemp("csvs") / "variables.csv"
    with open(var_csv, "a") as f:
        f.write("subject,continuous_variable,2_groups_variable\n")
        for i in range(10):
            f.write(f"{i},{i * 2},{i % 2}")
    return var_csv


