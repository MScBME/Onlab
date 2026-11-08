"""The .pt counterpart of the app's OnnxDetector with the same interface, for evaluating .pt models."""
from pathlib import Path

from ultralytics import YOLO

from swimmertracker.detection.types import Detection


class UltralyticsDetector:
    def __init__(self, model_path: Path, confidence: float = 0.5):
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, image) -> list:
        """All detections in the image in pixel coordinates, as the app's detector returns them."""
        boxes = self.model(image, conf=self.confidence, verbose=False)[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)) for b, c in zip(xyxy, conf)]
