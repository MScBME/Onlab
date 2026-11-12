"""Locations used by the research code, that is the training, the evaluation and the legacy CSRT experiment.

All paths are inside the repo, the research code only runs from source. There are two video libraries, both with
the layout of the app's library, videos.json, clips.json and videos/. The videos of the train library in
research/training/library give train and val frames only. The videos of the test library in data/, which is also
the app's dev library, give test frames only, and its clips are the ones the models are tried on in the app.
A video id may appear in only one of them.
"""
from swimmertracker.utils.paths import DEV_LIBRARY_DIR, DEV_MODELS_DIR, PROJECT_ROOT, LibraryPaths

PUBLIC_ANNOTATIONS_DIR = PROJECT_ROOT.parent / "public_annotations"

RESEARCH_DIR = PROJECT_ROOT / "research"
RUN_CONFIG_JSON = RESEARCH_DIR / "run_config.json"
MODELS_DIR = DEV_MODELS_DIR

TRAINING_DIR = RESEARCH_DIR / "training"
TRAIN_LIBRARY = LibraryPaths(TRAINING_DIR / "library")
TEST_LIBRARY = LibraryPaths(DEV_LIBRARY_DIR)

CACHE_DIR = TRAINING_DIR / "cache"  # gitignored
DATASET_DIR = CACHE_DIR / "dataset"
TRAINING_RUNS_DIR = CACHE_DIR / "runs"
WEIGHTS_DIR = CACHE_DIR / "weights"  # base weights such as yolo11n.pt
