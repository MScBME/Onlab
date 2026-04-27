import os
import shutil
from pathlib import Path
from src.utils.logger import get_logger

from dotenv import load_dotenv
from roboflow import Roboflow

from src.utils.paths import ANNOTATED_DIR, PROJECT_ROOT

HASH_MARKER = "_jpg.rf."

logger = get_logger(__name__)

def download_dataset(target_dir: Path = None, force: bool = False) -> Path:
    target_dir = target_dir or ANNOTATED_DIR

    if not force and target_dir.exists() and any(target_dir.glob("*.jpg")):
        print(f"Annotated data already present at {target_dir}, skipping download (use --force to refresh)")
        return target_dir

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project_name = os.getenv("ROBOFLOW_PROJECT")
    version_num = int(os.getenv("ROBOFLOW_VERSION", "1"))

    if not all([api_key, workspace, project_name]):
        raise RuntimeError("Missing Roboflow credentials in .env (ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT)")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)

    download_root = target_dir.parent / f"_rf_download_{project_name}_v{version_num}"
    if download_root.exists():
        shutil.rmtree(download_root)
    dataset = version.download("yolov11", location=str(download_root))
    dataset_path = Path(dataset.location)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("train", "valid", "test"):
        _flatten_split(dataset_path / sub, target_dir)

    shutil.rmtree(download_root)
    _strip_hash_suffix(target_dir)
    return target_dir


def _flatten_split(split_dir: Path, target_dir: Path):
    if not split_dir.exists():
        return
    images = split_dir / "images"
    labels = split_dir / "labels"
    if images.exists():
        for src in images.iterdir():
            shutil.copy2(src, target_dir / src.name)
    if labels.exists():
        for src in labels.iterdir():
            shutil.copy2(src, target_dir / src.name)


def _strip_hash_suffix(directory: Path):
    for entry in directory.iterdir():
        if HASH_MARKER not in entry.name:
            continue
        prefix = entry.name.split(HASH_MARKER)[0]
        new_name = f"{prefix}{entry.suffix}"
        new_path = entry.with_name(new_name)
        if new_path.exists():
            logger.warning(f"Duplicate detected: Removing {entry.name} because {new_name} already exists.")
            entry.unlink()
        else:
            entry.rename(new_path)
