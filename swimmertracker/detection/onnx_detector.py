"""The app's YOLO detector on onnxruntime. It runs on the CPU and needs neither torch nor ultralytics.

Pre and post processing follow the predict path of ultralytics, so an exported model gives the same boxes as the
.pt model it came from. That means a letterbox with gray 114 padding, RGB input scaled to 0..1, class aware NMS at
IoU 0.7 and scores strictly above the threshold. Two output layouts are handled. The classic YOLO v8 and 11 output
has the shape (1, 4 + classes, anchors) with center x, center y, width, height and the per class scores, and it
needs NMS. The end-to-end output of the default YOLO26 export has the shape (1, max_det, 6) with x1, y1, x2, y2,
score and class, and NMS is already done.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from swimmertracker.detection.types import Detection

PAD_VALUE = 114
DEFAULT_IOU = 0.7


def letterbox(image: np.ndarray, size: tuple) -> tuple:
    """Resize keeping the aspect ratio and pad to size, given as (h, w). Returns the image, the gain and the
    padding as (left, top)."""
    h0, w0 = image.shape[:2]
    h, w = size
    gain = min(h / h0, w / w0)
    new_w, new_h = int(round(w0 * gain)), int(round(h0 * gain))
    if (new_w, new_h) != (w0, h0):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (w - new_w) / 2, (h - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if top or bottom or left or right:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    return image, gain, (left, top)


def to_tensor(image: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 -> RGB NCHW float32 in 0..1."""
    return np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def decode(output: np.ndarray, confidence: float, iou: float = DEFAULT_IOU) -> np.ndarray:
    """Turn the raw model output into an (n, 6) array of x1, y1, x2, y2, score and class in network input pixels."""
    if output.shape[2] < output.shape[1]:  # end-to-end output, (1, max_det, 6)
        rows = output[0]
        return rows[rows[:, 4] > confidence].astype(np.float32)

    pred = output[0].T  # (anchors, 4 + classes)
    scores_all = pred[:, 4:]
    classes = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(pred)), classes]
    keep = scores > confidence
    if not keep.any():
        return np.zeros((0, 6), np.float32)
    xywh, scores, classes = pred[keep, :4], scores[keep], classes[keep]
    xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)

    boxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in xyxy]
    kept = cv2.dnn.NMSBoxesBatched(boxes, scores.tolist(), classes.tolist(), 0.0, iou)
    kept = np.array(kept, dtype=int).reshape(-1)
    kept = kept[np.argsort(-scores[kept], kind="stable")]
    return np.concatenate([xyxy[kept], scores[kept, None], classes[kept, None]], axis=1).astype(np.float32)


def to_image_coords(rows: np.ndarray, gain: float, pad: tuple, image_shape: tuple) -> np.ndarray:
    rows = rows.copy()
    rows[:, [0, 2]] = (rows[:, [0, 2]] - pad[0]) / gain
    rows[:, [1, 3]] = (rows[:, [1, 3]] - pad[1]) / gain
    h, w = image_shape[:2]
    rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
    rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
    return rows


class OnnxDetector:
    def __init__(self, model_path: Path, confidence: float = 0.5, iou: float = DEFAULT_IOU, providers=None):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), options, providers=providers or ["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        _, _, height, width = model_input.shape
        if not (isinstance(height, int) and isinstance(width, int)):
            raise ValueError(f"The ONNX input of {Path(model_path).name} needs a fixed size, got {model_input.shape}")
        self.input_size = (height, width)
        self.confidence = confidence
        self.iou = iou

    def detect(self, image: np.ndarray) -> list:
        padded, gain, pad = letterbox(image, self.input_size)
        output = self.session.run(None, {self.input_name: to_tensor(padded)})[0]
        rows = to_image_coords(decode(output, self.confidence, self.iou), gain, pad, image.shape)
        return [Detection(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]
