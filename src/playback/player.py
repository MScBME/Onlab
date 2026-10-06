"""Threaded video player: owns the decoder and runs either preview playback or a lane analysis.

All public methods are thread-safe and only enqueue a command; the player thread executes them
in order. Results are reported through a PlayerListener, called on the player thread.
"""
import queue
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from src.analysis.session import AnalysisParams, AnalysisSession, AnalysisSummary, FrameResult, lane_preview
from src.video.lane_warp import LANE_H, LANE_W
from src.video.loader import VideoInfo, VideoLoader

LAG_RESYNC_S = 0.25  # if playback falls this far behind real time, re-anchor instead of rushing


class PlayerState(Enum):
    IDLE = "idle"
    PREVIEW_PLAYING = "preview"
    LOADING_MODEL = "loading"
    ANALYZING = "analyzing"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


class PlayerListener:
    def on_opened(self, info: VideoInfo) -> None: ...
    def on_frame(self, result: FrameResult) -> None: ...
    def on_state(self, state: PlayerState, message: str) -> None: ...
    def on_summary(self, summary: AnalysisSummary) -> None: ...


def default_detector_factory(model_path: Path):
    from src.detection.detector import SwimmerDetector  # heavy import (torch), loaded on first use
    return SwimmerDetector(str(model_path))


class Player:
    def __init__(self, listener: PlayerListener, detector_factory=default_detector_factory):
        self._listener = listener
        self._detector_factory = detector_factory
        self._detectors = {}
        self._commands = queue.Queue()
        self._seek_lock = threading.Lock()
        self._pending_seek: Optional[float] = None

        self._loader: Optional[VideoLoader] = None
        self._video_path: Optional[Path] = None
        self._info: Optional[VideoInfo] = None
        self._state = PlayerState.IDLE
        self._preview_lane = None
        self._last_frame = None  # (timestamp, frame)
        self._session: Optional[AnalysisSession] = None
        self._params: Optional[AnalysisParams] = None
        self._clock = None  # (wall time, video timestamp) anchor for real-time pacing
        self._fps = 0.0
        self._last_emit = None

        self._thread = threading.Thread(target=self._run, name="Player", daemon=True)
        self._thread.start()

    # ---- public API ---------------------------------------------------------------------
    def open(self, video_path: Path):
        self._post("open", Path(video_path))

    def set_preview_lane(self, lane_coords):
        self._post("lane", lane_coords)

    def seek(self, sec: float):
        with self._seek_lock:
            self._pending_seek = sec
        self._post("seek")

    def play_preview(self):
        self._post("play_preview")

    def pause(self):
        self._post("pause")

    def resume(self):
        self._post("resume")

    def start_analysis(self, params: AnalysisParams):
        self._post("start", params)

    def stop(self):
        self._post("stop")

    def release(self):
        """Stop whatever runs and close the video file (frees the OS file handle)."""
        self._post("release")

    def close(self, timeout: float = 2.0):
        self._post("close")
        self._thread.join(timeout)

    # ---- player thread ----------------------------------------------------------------
    def _post(self, name: str, arg=None):
        self._commands.put((name, arg))

    def _playing(self) -> bool:
        return self._state in (PlayerState.PREVIEW_PLAYING, PlayerState.ANALYZING)

    def _run(self):
        handlers = {
            "open": self._open,
            "lane": self._cmd_lane,
            "seek": lambda _arg: self._handle_seek(),
            "play_preview": self._cmd_play_preview,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "start": self._start,
            "stop": self._cmd_stop,
            "release": self._cmd_release,
        }
        while True:
            try:
                cmd = self._commands.get(timeout=self._time_to_next_frame()) if self._playing() else self._commands.get()
            except queue.Empty:
                cmd = None
            if cmd is not None and cmd[0] == "close":
                break
            try:  # keep the thread alive on errors: report them and wait for the next command
                if cmd is not None:
                    handlers[cmd[0]](cmd[1])
                else:
                    self._advance()
            except Exception as e:
                self._fail(e)

        if self._loader is not None:
            self._loader.release()

    def _time_to_next_frame(self) -> float:
        if self._clock is None or self._last_frame is None or not self._info or self._info.fps <= 0:
            return 0.0
        wall0, video0 = self._clock
        due = wall0 + (self._last_frame[0] + 1.0 / self._info.fps - video0)
        return max(0.0, due - time.monotonic())

    def _set_state(self, state: PlayerState, message: str = ""):
        self._state = state
        self._listener.on_state(state, message)

    def _fail(self, error: Exception):
        self._session = None
        self._set_state(PlayerState.ERROR, str(error))

    def _cmd_lane(self, lane_coords):
        self._preview_lane = lane_coords
        if self._state not in (PlayerState.ANALYZING, PlayerState.PAUSED, PlayerState.LOADING_MODEL)                 and self._last_frame is not None:
            self._emit(self._preview_result(*self._last_frame))

    def _cmd_play_preview(self, _arg):
        if self._loader is not None and self._state in (PlayerState.IDLE, PlayerState.FINISHED, PlayerState.ERROR):
            self._reset_clock()
            self._set_state(PlayerState.PREVIEW_PLAYING)

    def _cmd_pause(self, _arg):
        if self._state == PlayerState.PREVIEW_PLAYING:
            self._set_state(PlayerState.IDLE)
        elif self._state == PlayerState.ANALYZING:
            self._set_state(PlayerState.PAUSED)

    def _cmd_resume(self, _arg):
        if self._state == PlayerState.PAUSED:
            self._reset_clock()
            self._set_state(PlayerState.ANALYZING)

    def _cmd_stop(self, _arg):
        if self._session is not None:
            self._finish(completed=False)
        if self._state != PlayerState.IDLE:
            self._set_state(PlayerState.IDLE)

    def _cmd_release(self, _arg):
        self._session = None
        self._last_frame = None
        if self._loader is not None:
            self._loader.release()
            self._loader, self._video_path, self._info = None, None, None
        if self._state != PlayerState.IDLE:
            self._set_state(PlayerState.IDLE)

    def _open(self, path: Path):
        if self._loader is not None:
            self._loader.release()
            self._loader = None
        self._session = None
        self._last_frame = None
        loader = VideoLoader(str(path))
        info = loader.info
        if info.width <= 0 or info.fps <= 0:
            loader.release()
            raise RuntimeError(f"Not a readable video: {path}")
        self._loader, self._video_path, self._info = loader, path, info
        self._listener.on_opened(info)
        self._set_state(PlayerState.IDLE)
        self._show_at(0.0)

    def _clamp(self, sec: float) -> float:
        last = max(0.0, self._info.duration - 1.0 / self._info.fps)
        return min(max(0.0, sec), last)

    def _handle_seek(self):
        with self._seek_lock:
            sec, self._pending_seek = self._pending_seek, None
        if sec is None or self._loader is None:
            return  # coalesced into an earlier seek command
        if self._state in (PlayerState.ANALYZING, PlayerState.PAUSED, PlayerState.LOADING_MODEL):
            return
        if self._state == PlayerState.PREVIEW_PLAYING:
            self._loader.seek(self._clamp(sec))
            self._reset_clock()
            return
        if self._state in (PlayerState.FINISHED, PlayerState.ERROR):
            self._set_state(PlayerState.IDLE)
        self._show_at(sec)

    def _show_at(self, sec: float):
        read = self._loader.read_at(self._clamp(sec))
        if read is not None:
            self._last_frame = read
            self._emit(self._preview_result(*read))

    def _start(self, params: AnalysisParams):
        if self._loader is None or Path(params.video_path) != self._video_path:
            self._open(Path(params.video_path))
        self._session = None
        key = str(params.model_path)
        if key not in self._detectors:
            self._set_state(PlayerState.LOADING_MODEL, f"Loading model {Path(key).name}...")
            detector = self._detector_factory(params.model_path)
            # The first inference initializes CUDA kernels (seconds, much longer on a cold start):
            # do it here so it is shown as loading instead of a frozen analysis.
            detector.detect(np.zeros((LANE_H, LANE_W, 3), dtype=np.uint8))
            self._detectors[key] = detector
        self._params = params
        self._session = AnalysisSession(
            self._detectors[key], params.lane_coords, params.lane_length_m, params.confidence, params.start_sec,
            run_id=params.run_id,
        )
        self._loader.seek(self._clamp(params.start_sec))
        self._reset_clock()
        self._fps = 0.0
        self._set_state(PlayerState.ANALYZING)

    def _finish(self, completed: bool):
        summary = self._session.summary(completed)
        self._session = None
        if completed:
            self._set_state(PlayerState.FINISHED)
        self._listener.on_summary(summary)

    def _reset_clock(self):
        self._clock = None
        self._last_emit = None

    def _advance(self):
        read = self._loader.read()
        analyzing = self._state == PlayerState.ANALYZING
        if read is None or (analyzing and read[0] > self._params.end_sec):
            if analyzing:
                self._finish(completed=True)
            else:
                self._set_state(PlayerState.IDLE)
            return

        timestamp, frame = read
        now = time.monotonic()
        if self._clock is None or now - (self._clock[0] + timestamp - self._clock[1]) > LAG_RESYNC_S:
            self._clock = (now, timestamp)

        result = self._session.process(frame, timestamp) if analyzing else self._preview_result(timestamp, frame)
        self._last_frame = read
        self._emit(result)

    def _preview_result(self, timestamp: float, frame) -> FrameResult:
        warped = lane_preview(frame, self._preview_lane) if self._preview_lane is not None else None
        return FrameResult(timestamp=timestamp, frame=frame, warped=warped)

    def _emit(self, result: FrameResult):
        now = time.monotonic()
        if self._playing():
            if self._last_emit is not None and now > self._last_emit:
                inst = 1.0 / (now - self._last_emit)
                self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            self._last_emit = now
            result.fps = self._fps
        self._listener.on_frame(result)
