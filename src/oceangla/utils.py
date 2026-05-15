import json
from pathlib import Path
from importlib import metadata
from collections import Counter


def gen_dataset_description(dataset_description_json: Path | str):
    d = {
        "Name": f"oceangla {metadata.version('oceangla')}",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative"
    }
    with Path(dataset_description_json).open("w") as f:
        json.dump(d, f, indent=4)


def get_unique_session_names_counter(fladir: str | Path) -> Counter:
    fladir = Path(fladir)
    return Counter((
        p.name for p in fladir.glob("sub-*/ses-*")
    ))
