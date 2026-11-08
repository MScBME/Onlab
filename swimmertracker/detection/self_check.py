"""Load every available model and run it once. The packaging smoke test of the frozen app uses this."""
from pathlib import Path

import numpy as np

from swimmertracker.catalog.model_catalog import list_models
from swimmertracker.detection.onnx_detector import OnnxDetector
from swimmertracker.video.lane_warp import LANE_H, LANE_W


def check_models(models_dir: Path) -> list:
    results = []
    for model in list_models(models_dir):
        try:
            OnnxDetector(model.path).detect(np.zeros((LANE_H, LANE_W, 3), np.uint8))
            results.append({"name": model.name, "display_name": model.display_name, "ok": True})
        except Exception as e:  # report the error instead of crashing, the report is the point
            results.append({"name": model.name, "display_name": model.display_name, "ok": False, "error": str(e)})
    return results
