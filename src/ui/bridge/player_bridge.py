"""Adapts the threaded core Player to Qt: listener callbacks -> queued signals on the GUI thread."""
import threading

from PySide6.QtCore import QObject, Qt, Signal, Slot

from src.playback.player import Player, PlayerListener, PlayerState, default_detector_factory


class _Listener(PlayerListener):
    """Runs on the player thread; only stores data and emits internal signals."""

    def __init__(self, bridge: "PlayerBridge"):
        self.bridge = bridge

    def on_opened(self, info):
        self.bridge._opened.emit(info)

    def on_frame(self, result):
        bridge = self.bridge
        with bridge._lock:
            if result.run_id is not None:
                bridge._samples.append((result.run_id, result.timestamp, result.speed_m_s or 0.0))
            bridge._latest = result
            notify = not bridge._frame_pending
            bridge._frame_pending = True
        if notify:  # coalesce: the GUI only ever renders the newest frame
            bridge._frame.emit()

    def on_state(self, state: PlayerState, message: str):
        self.bridge._state.emit(state.value, message)

    def on_summary(self, summary):
        self.bridge._summary.emit(summary)


class PlayerBridge(QObject):
    frameReady = Signal(object)  # FrameResult
    opened = Signal(object)  # VideoInfo
    stateChanged = Signal(str, str)  # PlayerState value, message
    summaryReady = Signal(object)  # AnalysisSummary

    _frame = Signal()
    _opened = Signal(object)
    _state = Signal(str, str)
    _summary = Signal(object)

    def __init__(self, parent=None, detector_factory=default_detector_factory):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._latest = None
        self._frame_pending = False
        self._samples = []
        self._frame.connect(self._deliver_frame, Qt.ConnectionType.QueuedConnection)
        self._opened.connect(self.opened, Qt.ConnectionType.QueuedConnection)
        self._state.connect(self.stateChanged, Qt.ConnectionType.QueuedConnection)
        self._summary.connect(self.summaryReady, Qt.ConnectionType.QueuedConnection)
        self.player = Player(_Listener(self), detector_factory)

    @Slot()
    def _deliver_frame(self):
        with self._lock:
            result, self._latest = self._latest, None
            self._frame_pending = False
        if result is not None:
            self.frameReady.emit(result)

    def take_samples(self, run_id: int) -> list:
        """Drain (timestamp, speed) samples of every analysed frame of the given run (none are dropped)."""
        with self._lock:
            samples, self._samples = self._samples, []
        return [(t, s) for rid, t, s in samples if rid == run_id]

    def close(self):
        self.player.close()
