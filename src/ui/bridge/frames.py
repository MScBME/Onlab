"""numpy (BGR) frames -> Qt: QImage for thumbnails, QVideoSink for live VideoOutput items."""
from typing import Optional

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QVideoFrame, QVideoSink


def _wrap(array: np.ndarray) -> tuple:
    array = np.ascontiguousarray(array)
    h, w = array.shape[:2]
    return array, QImage(array.data, w, h, array.strides[0], QImage.Format.Format_BGR888)


def bgr_to_qimage(array: np.ndarray) -> QImage:
    _, image = _wrap(array)
    return image.copy()  # detach from the numpy buffer


class FrameSink:
    """Feeds frames into the QVideoSink of a QML VideoOutput (call on the GUI thread)."""

    def __init__(self):
        self._sink: Optional[QVideoSink] = None

    def attach(self, sink: Optional[QVideoSink]):
        self._sink = sink
        if sink is not None:
            sink.destroyed.connect(self._on_destroyed)

    def _on_destroyed(self, *_):
        self._sink = None

    def push(self, array: Optional[np.ndarray]):
        if self._sink is None or array is None:
            return
        _array, image = _wrap(array)  # _array keeps the buffer alive until QVideoFrame has copied it
        try:
            self._sink.setVideoFrame(QVideoFrame(image))
        except RuntimeError:  # the QML item (and its sink) is already gone
            self._sink = None
