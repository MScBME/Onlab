from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
TRAINING_CACHE_DIR = PROJECT_ROOT / "training_cache"
CONFIG_DIR = PROJECT_ROOT / "config"

VIDEOS_JSON = DATA_DIR / "videos.json"
CLIPS_JSON = DATA_DIR / "clips.json"
EVENTS_JSON = TRAINING_DIR / "events.json"
EVENT_FRAMES_JSON = TRAINING_DIR / "event_frames.json"
CUSTOM_FRAMES_JSON = TRAINING_DIR / "custom_frames.json"
RUN_CONFIG_JSON = CONFIG_DIR / "run_config.json"
EXTRACTION_CONFIG_JSON = CONFIG_DIR / "extraction_config.json"

RAW_FRAMES_DIR = TRAINING_CACHE_DIR / "raw_frames"
ANNOTATED_DIR = TRAINING_CACHE_DIR / "annotated"
PROCESSED_DIR = TRAINING_CACHE_DIR / "processed"
DATASET_DIR = TRAINING_CACHE_DIR / "dataset"
TRAINING_RUNS_DIR = TRAINING_CACHE_DIR / "runs"
