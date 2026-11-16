"""Model metadata, the models/<name>.json file next to <name>.pt and <name>.onnx.

It is the hand-over between research and the app. The app reads ``display_name`` and ``input``, the other fields
record where the model came from, that is the training data, the split, the arguments, the metrics and the ONNX
export. ``display_name`` is meant to be edited by hand and is never overwritten.
"""
import csv
import json
import platform
import subprocess
from pathlib import Path

from swimmertracker.utils.config import save_json
from swimmertracker.utils.paths import PROJECT_ROOT
from swimmertracker.video.lane_warp import LANE_H, LANE_W

LANE_WARP_INPUT = {"type": "lane_warp", "width": LANE_W, "height": LANE_H}


def full_frame_input(imgsz: int) -> dict:
    """Input metadata descriptor for full-frame swimmer detection models."""
    return {"type": "full_frame", "imgsz": int(imgsz)}


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


def json_safe_value(value):
    """Convert Path and container values into JSON-serializable primitives."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return str(value)


def environment_info(resolved_device: str | None = None) -> dict:
    """Capture runtime software and hardware details used for training."""
    info: dict = {"python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    if resolved_device is not None:
        info["device"] = resolved_device
    return info


def parse_results_csv(csv_path: Path) -> dict:
    """Extract epoch count, training duration, best epoch and final epoch metrics from Ultralytics results.csv."""
    if not csv_path.exists():
        return {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    cleaned_rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
    last = cleaned_rows[-1]
    best_row = max(
        cleaned_rows,
        key=lambda r: float(r.get("metrics/mAP50-95(B)") or -1.0),
    )

    final_metrics = {}
    for k, v in last.items():
        if k in ("epoch", "time"):
            continue
        try:
            final_metrics[k] = round(float(v), 6)
        except ValueError:
            continue

    summary: dict = {"epochs_completed": int(float(last["epoch"])) if "epoch" in last else len(cleaned_rows)}
    if "time" in last:
        summary["training_time_s"] = round(float(last["time"]), 2)
    if "epoch" in best_row:
        summary["best_epoch"] = int(float(best_row["epoch"]))
    if "metrics/mAP50-95(B)" in best_row:
        summary["best_val_map50_95"] = round(float(best_row["metrics/mAP50-95(B)"]), 4)
    if final_metrics:
        summary["final_epoch_metrics"] = final_metrics
    return summary


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
