"""ONNX export of a trained model for the app, the last step of train-model.

The ONNX input has the fixed shape that ultralytics itself predicts at for a 256x1024 warped lane. For models
trained with imgsz=1024 this is exactly 1024x256 in H x W order without letterbox padding, so the ONNX model
gives the same output in the app as the .pt model.
"""
import hashlib
import math
from datetime import datetime
from pathlib import Path

import onnxruntime as ort
import ultralytics
from ultralytics import YOLO

from research.training.model_metadata import LANE_WARP_INPUT, load_metadata, update_metadata

STRIDE = 32


def predict_shape(train_imgsz: int, lane_w: int, lane_h: int) -> tuple:
    """The (h, w) size that the rect letterbox of ultralytics makes of a lane_w x lane_h image at train_imgsz."""
    r = min(train_imgsz / lane_h, train_imgsz / lane_w)
    h, w = round(lane_h * r), round(lane_w * r)
    return math.ceil(h / STRIDE) * STRIDE, math.ceil(w / STRIDE) * STRIDE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(pt_path: Path) -> Path:
    """Export the .pt model to the .onnx file next to it and record the export in the model metadata."""
    pt_path = Path(pt_path)
    lane_input = load_metadata(pt_path).get("input", LANE_WARP_INPUT)
    if lane_input.get("type") != "lane_warp":
        raise ValueError(f"Only lane_warp models can be exported for the app so far, {pt_path.name} is not one")

    model = YOLO(str(pt_path))
    train_imgsz = int(model.overrides.get("imgsz") or 640)
    imgsz = predict_shape(train_imgsz, lane_input["width"], lane_input["height"])
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, device="cpu",
                                  verbose=False))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    update_metadata(
        pt_path,
        input=lane_input,
        classes=list(model.names.values()),
        onnx={
            "file": onnx_path.name,
            "imgsz": list(imgsz),
            "output_shape": list(session.get_outputs()[0].shape),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "ultralytics": ultralytics.__version__,
            "sha256": _sha256(onnx_path),
        },
    )
    return onnx_path
