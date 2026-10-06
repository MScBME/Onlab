from ultralytics import YOLO


class SwimmerDetector:
    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """Detect swimmer in a warped lane image.

        Returns ((cx, cy), (x1, y1, x2, y2), confidence) or None.
        """
        results = self.model(frame, conf=self.confidence, verbose=False)[0]

        if len(results.boxes) == 0:
            return None

        best = results.boxes.conf.argmax()
        xyxy = results.boxes.xyxy[best].cpu().numpy()
        conf = float(results.boxes.conf[best])

        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        return (float(cx), float(cy)), (int(x1), int(y1), int(x2), int(y2)), conf