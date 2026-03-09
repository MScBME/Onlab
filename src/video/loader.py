import cv2

class VideoLoader:
    def __init__(self, path: str, start_sec: float = 0.0, end_sec: float = None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.end_sec = end_sec

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if start_sec > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    def read_frames(self):
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            current_time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = current_time_ms / 1000.0

            if self.end_sec is not None and timestamp > self.end_sec:
                break

            yield frame_id, timestamp, frame

            frame_id += 1