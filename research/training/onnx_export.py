"""ONNX export of a trained model for the app, the last step of train-model.

Supports both legacy lane-warp models (256x1024 warped lanes) and full-frame models
(square ``imgsz`` aligned to YOLO stride 32), exporting static shapes without extra
letterbox mismatch so the ONNX model matches PyTorch validation output.
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


def _resolve_export_imgsz(input_cfg: dict, train_imgsz: int) -> tuple[int, int]:
    input_type = input_cfg.get("type", "lane_warp")
    if input_type == "lane_warp":
        return predict_shape(train_imgsz, input_cfg["width"], input_cfg["height"])
    if input_type == "full_frame":
        size = int(input_cfg.get("imgsz") or train_imgsz)
        aligned = math.ceil(size / STRIDE) * STRIDE
        return aligned, aligned
    raise ValueError(f"Unsupported model input type for ONNX export: {input_type}")


def export_onnx(pt_path: Path) -> Path:
    """Export the .pt model to the .onnx file next to it and record the export in the model metadata."""
    pt_path = Path(pt_path)
    input_cfg = load_metadata(pt_path).get("input", LANE_WARP_INPUT)

    model = YOLO(str(pt_path))
    train_imgsz = int(model.overrides.get("imgsz") or 640)
    imgsz = _resolve_export_imgsz(input_cfg, train_imgsz)
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, device="cpu",
                                  verbose=False))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    update_metadata(
        pt_path,
        input=input_cfg,
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
