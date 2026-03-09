import cv2


class VideoLoader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read_frames(self):
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_id / self.fps

            yield frame_id, timestamp, frame

            frame_id += 1