from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe_video(path: str) -> VideoInfo:
    """Read basic stream properties; raises RuntimeError if the file is not a readable video."""
    loader = VideoLoader(path)
    try:
        info = loader.info
    finally:
        loader.release()
    if info.width <= 0 or info.height <= 0 or info.fps <= 0:
        raise RuntimeError(f"Not a readable video: {path}")
    return info


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

    @property
    def info(self) -> VideoInfo:
        return VideoInfo(
            fps=float(self.fps or 0.0),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def seek(self, sec: float):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sec) * 1000)

    def read(self):
        """Decode the next frame; returns (timestamp_sec, frame) or None at the end of the stream."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, frame

    def read_at(self, sec: float):
        self.seek(sec)
        return self.read()

    def release(self):
        self.cap.release()

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
