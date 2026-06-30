import shutil
import atexit
import tarfile
import logging

import requests
import pandas as pd
import nibabel as nib

from ..config import config

logger = logging.getLogger(__name__)

VALID_ATLASES = (
    "Glasser",
    "Gordon",
    "HCP",
    "MIDB",
    "MyersLabonte",
    "Tian"
)


def download_atlases() -> bool:
    """
    Download template atlases into a temporary directory,
    tracked by the global config. Returns False if there
    was a problem, returns True if everything was pulled successfully.

    Returns
    =======
    bool
        True if successful, False if not
    """

    # Links to resources pulled from: https://github.com/PennLINC/xcp_d/blob/f4cb65204e19110133e2e7778cfeba84636f098d/Dockerfile.base
    # Just doing XCPDAtlases now
    try:
        config.atlas_dir = config.outdir_path / ".atlases"
        config.atlas_dir.mkdir(exist_ok=True)
        atexit.register(lambda p: shutil.rmtree, config.atlas_dir)
        # XCP-D Atlases download
        res = requests.get("https://upenn.box.com/shared/static/4amxp72grenmp1up689k5oyn1i6nhunq.tar.gz")
        with open(config.atlas_dir / "XCPDAtlases.tar.gz", "wb") as f:
            f.write(res.content)
        with tarfile.open(config.atlas_dir / "XCPDAtlases.tar.gz", "r:*") as tar:
            tar.extractall(path=config.atlas_dir)
        (config.atlas_dir / "XCPDAtlases.tar.gz").unlink()
        if not (config.atlas_dir / "XCPDAtlases").is_dir():
            raise FileNotFoundError("Could not find XCPDAtlases directory after downloading")
    except Exception:  # will make this more granular
        logger.warning("Problem pulling atlases")
        shutil.rmtree(config.atlas_dir)
        return False
    return True


def fetch_atlas_dlabel_and_tsv(atlas_name: str, space: str) -> tuple[nib.cifti2.cifti2.Cifti2Image, pd.DataFrame]:
    """
    Fetch an atlas .dlabel and .tsv file.
    """
    if atlas_name not in VALID_ATLASES:
        raise ValueError(f"Atlas {atlas_name} not available. Available atlases: {','.join(VALID_ATLASES)}")
    if config.atlas_dir is None or not config.atlas_dir.is_dir():
        raise FileNotFoundError("No atlases have been downloaded yet -- must call oceangla.data.atlas.download_atlases() first.")
    try:
        dlabel = next(config.atlas_dir.rglob(f"tpl-fsLR*{atlas_name}*.dlabel.nii"))
        tsv = next(config.atlas_dir.rglob(f"tpl-fsLR*{atlas_name}*.tsv"))
    except StopIteration:
        raise FileNotFoundError("Problem pulling atlas files.")
    return (nib.load(dlabel), pd.read_csv(tsv, sep="\t"))
