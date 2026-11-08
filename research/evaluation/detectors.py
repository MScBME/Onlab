"""Pick the detector by model file. A .onnx model runs on the app's OnnxDetector, a .pt model on ultralytics."""
from pathlib import Path

from swimmertracker.detection.onnx_detector import OnnxDetector


def make_detector(model_path: Path, confidence: float):
    model_path = Path(model_path)
    if model_path.suffix == ".onnx":
        return OnnxDetector(model_path, confidence=confidence)
    if model_path.suffix == ".pt":
        from research.evaluation.ultralytics_detector import UltralyticsDetector  # imports torch, so only when needed
        return UltralyticsDetector(model_path, confidence=confidence)
    raise ValueError(f"Unsupported model file '{model_path.name}', expected .onnx or .pt")
