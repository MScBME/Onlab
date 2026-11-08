"""Where the app finds its models and its video library.

Run from source with ``uv run swimmertracker``, the library is the repo's ``data/`` folder and the models come from
``models/``. In the frozen PyInstaller build the models are bundled next to the code and the library lives in the
user's Documents folder. The Documents path comes from the UI layer because Qt knows where it is, see
``default_library_dir``. Both locations can be overridden with environment variables.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent  # the repo root when running from source

FROZEN = bool(getattr(sys, "frozen", False))

LIBRARY_ENV = "SWIMMERTRACKER_LIBRARY"
MODELS_ENV = "SWIMMERTRACKER_MODELS"
LIBRARY_FOLDER_NAME = "Swimmer Tracker"

DEV_LIBRARY_DIR = PROJECT_ROOT / "data"
DEV_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class LibraryPaths:
    """A video library, that is ``videos.json``, ``clips.json`` and ``videos/`` under one root folder."""
    root: Path

    @property
    def videos_json(self) -> Path:
        return self.root / "videos.json"

    @property
    def clips_json(self) -> Path:
        return self.root / "clips.json"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"


def bundle_dir() -> Path:
    """Folder of the bundled read only files, PyInstaller's _internal in the frozen app and the repo root from source."""
    return Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))


def models_dir() -> Path:
    override = os.environ.get(MODELS_ENV)
    if override:
        return Path(override)
    return bundle_dir() / "models" if FROZEN else DEV_MODELS_DIR


def default_library_dir(documents_dir: Path = None) -> Path:
    override = os.environ.get(LIBRARY_ENV)
    if override:
        return Path(override)
    if FROZEN:
        return Path(documents_dir or Path.home() / "Documents") / LIBRARY_FOLDER_NAME
    return DEV_LIBRARY_DIR
