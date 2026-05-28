import pytest

from ..formula import parse_model_file


@pytest.mark.parametrize(
    "line,expected_model_name,expected_model",
    [
        ("model1 -> depvar ~ indepvar", "model1", "depvar ~ indepvar"),
        (
            "model2 -> depvar1 - depvar2 ~ indepvar",
            "model2",
            "depvar1 - depvar2 ~ indepvar",
        ),
        pytest.param("-> depvar ~ indepvar", None, None, marks=pytest.mark.xfail),
        pytest.param("modelname -> ", None, None, marks=pytest.mark.xfail),
        pytest.param("", None, None, marks=pytest.mark.xfail),
    ],
)
def test_parse_model_file(line, expected_model_name, expected_model, tmp_path):
    with (modelfile := (tmp_path / "models.txt")).open("w") as f:
        f.write(line)
    model_names, models = parse_model_file(modelfile)
    assert model_names[0] == expected_model_name
    assert models[0] == expected_model
