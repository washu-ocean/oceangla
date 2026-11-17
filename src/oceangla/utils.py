import json
from collections import Counter
from importlib import metadata
from pathlib import Path
import re


def gen_dataset_description(dataset_description_json: Path | str):
    d = {
        "Name": f"oceangla {metadata.version('oceangla')}",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
    }
    with Path(dataset_description_json).open("w") as f:
        json.dump(d, f, indent=4)


def get_unique_session_names_counter(fladir: str | Path) -> Counter:
    fladir = Path(fladir)
    return Counter((p.name for p in fladir.glob("sub-*/ses-*")))


def camel_to_kebab(text: str) -> str:
    """
    Detects if a string is camelCase (starts with a lowercase letter, contains
    no spaces/underscores/hyphens, and has at least one uppercase letter).
    If so, converts it to kebab-case; otherwise returns the original string.
    """
    is_camel_case = bool(re.fullmatch(r"[a-z]+(?:[A-Z][a-z0-9]*)+", text))

    if not is_camel_case:
        return text

    return re.sub(r"([A-Z])", r"-\1", text).lower()