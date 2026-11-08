"""Model metadata, the models/<name>.json file next to <name>.pt and <name>.onnx.

It is the hand-over between research and the app. The app reads ``display_name`` and ``input``, the other fields
record where the model came from, that is the training data, the split, the arguments, the metrics and the ONNX
export. ``display_name`` is meant to be edited by hand and is never overwritten.
"""
import json
import subprocess
from pathlib import Path

from swimmertracker.utils.config import save_json
from swimmertracker.utils.paths import PROJECT_ROOT
from swimmertracker.video.lane_warp import LANE_H, LANE_W

LANE_WARP_INPUT = {"type": "lane_warp", "width": LANE_W, "height": LANE_H}


def metadata_path(model_path: Path) -> Path:
    return Path(model_path).with_suffix(".json")


def load_metadata(model_path: Path) -> dict:
    path = metadata_path(model_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_metadata(model_path: Path, **sections) -> dict:
    """Merge top level sections into the metadata file of the model and save it, creating it if needed."""
    model_path = Path(model_path)
    metadata = load_metadata(model_path)
    metadata.setdefault("name", model_path.stem)
    metadata.setdefault("display_name", model_path.stem)
    metadata.setdefault("input", LANE_WARP_INPUT)
    metadata.setdefault("classes", ["swimmer"])
    metadata.update({k: v for k, v in sections.items() if k != "display_name"})
    save_json(metadata_path(model_path), metadata)
    return metadata


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {}


def metrics_from(results, images: int) -> dict:
    """The box metrics of an ultralytics validation run."""
    d = results.results_dict
    return {
        "images": images,
        "precision": round(float(d["metrics/precision(B)"]), 4),
        "recall": round(float(d["metrics/recall(B)"]), 4),
        "map50": round(float(d["metrics/mAP50(B)"]), 4),
        "map50_95": round(float(d["metrics/mAP50-95(B)"]), 4),
    }
