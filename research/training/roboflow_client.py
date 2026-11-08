import json
import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from roboflow import Roboflow

from research.paths import ANNOTATED_DIR
from swimmertracker.utils.paths import PROJECT_ROOT

HASH_MARKER = "_jpg.rf."
SOURCE_MARKER = ".source.json"  # records the Roboflow project and version of the cached annotations


def configured_source() -> dict:
    """The Roboflow dataset named in .env by ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT and ROBOFLOW_VERSION."""
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "workspace": os.getenv("ROBOFLOW_WORKSPACE"),
        "project": os.getenv("ROBOFLOW_PROJECT"),
        "version": int(os.getenv("ROBOFLOW_VERSION", "1")),
    }


def cached_source(target_dir: Path = None) -> Optional[dict]:
    marker = (target_dir or ANNOTATED_DIR) / SOURCE_MARKER
    if not marker.exists():
        return None
    return json.loads(marker.read_text(encoding="utf-8"))


def download_dataset(target_dir: Path = None, force: bool = False) -> dict:
    """Download the annotated dataset or reuse the cached one, and return the source it came from.

    The cache is reused only if it came from the project and version configured in .env, so a change of
    ROBOFLOW_PROJECT or ROBOFLOW_VERSION takes effect without --force-download.
    """
    target_dir = target_dir or ANNOTATED_DIR
    source = configured_source()
    has_images = target_dir.exists() and any(target_dir.glob("*.jpg"))
    cached = cached_source(target_dir)

    if not force and has_images and cached == source:
        print(f"The annotations of {_describe(source)} are already in {target_dir}, skipping the download. "
              "Use --force-download to download them again.")
        return source
    if has_images and not force:
        reason = f"they are from {_describe(cached)}" if cached else "they have no record of their Roboflow version"
        print(f"The cached annotations are not used because {reason}, while .env names {_describe(source)}.")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not all([api_key, source["workspace"], source["project"]]):
        raise RuntimeError("Roboflow credentials are missing from .env, it needs ROBOFLOW_API_KEY, "
                           "ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(source["workspace"]).project(source["project"])
    version = project.version(source["version"])

    download_root = target_dir.parent / f"_rf_download_{source['project']}_v{source['version']}"
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
    (target_dir / SOURCE_MARKER).write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return source


def _describe(source: dict) -> str:
    return f"{source['workspace']}/{source['project']} v{source['version']}"


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
            print(f"[WARN] Removing the duplicate {entry.name} because {new_name} already exists")
            entry.unlink()
        else:
            entry.rename(new_path)
