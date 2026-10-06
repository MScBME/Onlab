"""Workspace: preview/scrub a video, pick a range + lane + model, run the analysis, save clips."""
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot, Property

from src.analysis.session import AnalysisParams, AnalysisSummary, FrameResult
from src.catalog.clips import create_clip, propose_clip
from src.catalog.library import VideoLibrary
from src.catalog.models import Video
from src.ui.bridge.frames import FrameSink
from src.ui.bridge.player_bridge import PlayerBridge
from src.ui.bridge.qt_helpers import ro
from src.utils.config import read_json
from src.utils.paths import MODELS_DIR, RUN_CONFIG_JSON
from src.utils.timefmt import format_time, parse_time
from src.video.loader import VideoInfo

MIN_RANGE_S = 0.1
DEFAULT_CONFIDENCE = 0.4
RUNNING_STATES = ("loading", "analyzing", "paused")


class WorkspaceController(QObject):
    videoChanged = Signal()
    stateChanged = Signal()
    positionChanged = Signal()
    rangeChanged = Signal()
    settingsChanged = Signal()
    telemetryChanged = Signal()
    overlayChanged = Signal()
    summaryChanged = Signal()
    clipsChanged = Signal()
    chartReset = Signal()
    errorOccurred = Signal(str)
    infoMessage = Signal(str)

    def __init__(self, library: VideoLibrary, models_dir: Path = MODELS_DIR, parent=None):
        super().__init__(parent)
        self.library = library
        self.models_dir = Path(models_dir)
        self._bridge = PlayerBridge(self)
        self._bridge.frameReady.connect(self._on_frame)
        self._bridge.opened.connect(self._on_opened)
        self._bridge.stateChanged.connect(self._on_state)
        self._bridge.summaryReady.connect(self._on_summary)
        self._original, self._warped = FrameSink(), FrameSink()

        self._video: Optional[Video] = None
        self._video_id = ""
        self._file_name = ""
        self._duration = 0.0
        self._fps = 0.0
        self._frame_w = self._frame_h = 0
        self._lanes = []
        self._lane_ids = []
        self._reset_range_on_open = False

        self._state = "idle"
        self._status_message = ""
        self._position = 0.0
        self._range_start = 0.0
        self._range_end = 0.0

        self._models = []
        self._model_name = ""
        self._confidence = DEFAULT_CONFIDENCE
        self._lane_id = -1
        self._load_defaults()

        self._run_id = 0
        self._active = None  # settings snapshot of the current/last analysis run
        self._active_start = 0.0
        self._reset_results()
        self._clips = []

    # ---- properties --------------------------------------------------------------------
    videoId = ro(str, "_video_id", videoChanged)
    fileName = ro(str, "_file_name", videoChanged)
    duration = ro(float, "_duration", videoChanged)
    fps = ro(float, "_fps", videoChanged)
    frameWidth = ro(int, "_frame_w", videoChanged)
    frameHeight = ro(int, "_frame_h", videoChanged)
    lanes = ro("QVariantList", "_lanes", videoChanged)
    laneIds = ro("QVariantList", "_lane_ids", videoChanged)

    state = ro(str, "_state", stateChanged)
    statusMessage = ro(str, "_status_message", stateChanged)
    position = ro(float, "_position", positionChanged)
    rangeStart = ro(float, "_range_start", rangeChanged)
    rangeEnd = ro(float, "_range_end", rangeChanged)

    models = ro("QVariantList", "_models", settingsChanged)
    modelName = ro(str, "_model_name", settingsChanged)
    confidence = ro(float, "_confidence", settingsChanged)
    laneId = ro(int, "_lane_id", settingsChanged)

    speed = ro(float, "_speed", telemetryChanged)
    positionM = ro(float, "_position_m", telemetryChanged)
    detectionStatus = ro(str, "_detection_status", telemetryChanged)
    detectionConfidence = ro(float, "_detection_conf", telemetryChanged)
    processingFps = ro(float, "_processing_fps", telemetryChanged)
    elapsed = ro(float, "_elapsed", telemetryChanged)

    swimmerPoint = ro("QVariantList", "_point", overlayChanged)
    bbox = ro("QVariantList", "_bbox", overlayChanged)

    summary = ro("QVariantMap", "_summary", summaryChanged)
    smoothedSeries = ro("QVariantList", "_smoothed", summaryChanged)
    chartDuration = ro(float, "_chart_duration", summaryChanged)

    clips = ro("QVariantList", "_clips", clipsChanged)

    def _settings(self) -> tuple:
        return (self._lane_id, self._model_name, round(self._confidence, 3),
                round(self._range_start, 3), round(self._range_end, 3))

    def _dirty(self) -> bool:
        return self._active is not None and self._state in RUNNING_STATES + ("finished",) \
            and self._settings() != self._active

    settingsDirty = Property(bool, _dirty, notify=settingsChanged)

    def _can_analyze(self) -> bool:
        return (self._video is not None and self._lane_id in self._lane_ids and bool(self._model_name)
                and self._range_end - self._range_start >= MIN_RANGE_S and self._state != "loading")

    canAnalyze = Property(bool, _can_analyze, notify=settingsChanged)

    # ---- setup -------------------------------------------------------------------------
    def _load_defaults(self):
        """Initial model/confidence from config/run_config.json (the app never writes it)."""
        try:
            config = read_json(RUN_CONFIG_JSON)
        except Exception:
            config = {}
        self._model_name = Path(config.get("model_path", "")).name
        self._confidence = float(config.get("confidence_threshold", DEFAULT_CONFIDENCE))
        self._refresh_models()

    def _refresh_models(self):
        self._models = sorted(p.name for p in self.models_dir.glob("*.pt"))
        if self._model_name not in self._models:
            self._model_name = self._models[0] if self._models else ""

    def _reset_results(self):
        self._speed = 0.0
        self._position_m = -1.0
        self._detection_status = ""
        self._detection_conf = 0.0
        self._processing_fps = 0.0
        self._elapsed = 0.0
        self._point = []
        self._bbox = []
        self._summary = {}
        self._smoothed = []
        self._chart_duration = max(MIN_RANGE_S, self._range_end - self._range_start)

    @Slot(QObject)
    def setOriginalSink(self, sink):
        self._original.attach(sink)

    @Slot(QObject)
    def setWarpedSink(self, sink):
        self._warped.attach(sink)

    @Slot(str, result=bool)
    def openVideo(self, video_id: str) -> bool:
        try:
            video = self.library.videos.get(video_id)
        except Exception as e:
            self.errorOccurred.emit(str(e))
            return False
        path = self.library.videos.file_path(video)
        if not path.exists():
            self.errorOccurred.emit(f"Video file is missing: data/raw/{video.path}")
            return False

        self._bridge.player.stop()
        self._video, self._video_id, self._file_name = video, video.id, video.path
        self._duration = self._fps = 0.0
        self._frame_w = self._frame_h = 0
        self._position = 0.0
        self._range_start = self._range_end = 0.0
        self._reset_range_on_open = True
        self._active = None
        self._reset_results()
        self._apply_lanes(video, keep_lane=False)
        self._refresh_models()
        self._load_clips()
        self._bridge.player.open(path)
        self._sync_preview_lane()
        self.chartReset.emit()
        for signal in (self.videoChanged, self.positionChanged, self.rangeChanged, self.settingsChanged,
                       self.telemetryChanged, self.overlayChanged, self.summaryChanged):
            signal.emit()
        return True

    @Slot()
    def closeVideo(self):
        self._bridge.player.release()
        self._video = None

    @Slot()
    def reloadVideo(self):
        """Re-read the open video after its lanes were edited."""
        if self._video is None:
            return
        try:
            video = self.library.videos.get(self._video_id)
        except KeyError:
            return
        self._video = video
        self._apply_lanes(video, keep_lane=True)
        self._sync_preview_lane()
        self._load_clips()
        self.videoChanged.emit()
        self.settingsChanged.emit()

    def _apply_lanes(self, video: Video, keep_lane: bool):
        self._lanes = [{"id": lane.id, "points": lane.coordinates, "lengthM": lane.length_m} for lane in video.lanes]
        self._lane_ids = [lane.id for lane in video.lanes]
        if not (keep_lane and self._lane_id in self._lane_ids):
            self._lane_id = self._lane_ids[0] if self._lane_ids else -1

    def _sync_preview_lane(self):
        if self._video is not None and self._lane_id in self._lane_ids:
            self._bridge.player.set_preview_lane(self._video.lane(self._lane_id).coordinates)

    def _load_clips(self):
        if self._video is None:
            self._clips = []
        else:
            clips = sorted(self.library.clips.for_video(self._video_id), key=lambda c: c.start_time)
            self._clips = [{
                "id": c.id,
                "lane": c.lane,
                "start": c.start_sec,
                "end": c.end_sec,
                "startText": c.start_time,
                "endText": c.end_time,
                "description": c.description or "",
            } for c in clips]
        self.clipsChanged.emit()

    # ---- player events -----------------------------------------------------------------
    def _on_opened(self, info: VideoInfo):
        self._duration, self._fps = info.duration, info.fps
        self._frame_w, self._frame_h = info.width, info.height
        if self._reset_range_on_open:
            self._reset_range_on_open = False
            self._range_start, self._range_end = 0.0, info.duration
            self.rangeChanged.emit()
            self.settingsChanged.emit()
        self.videoChanged.emit()

    def _on_frame(self, result: FrameResult):
        self._original.push(result.frame)
        self._warped.push(result.warped)
        self._position = result.timestamp
        self.positionChanged.emit()

        if result.run_id is None:
            if self._point or self._bbox:
                self._point, self._bbox = [], []
                self.overlayChanged.emit()
            return
        if result.run_id != self._run_id:
            return
        self._speed = result.speed_m_s or 0.0
        self._position_m = result.position_m if result.position_m is not None else -1.0
        self._detection_status = result.status.value if result.status else ""
        self._detection_conf = result.confidence or 0.0
        self._processing_fps = result.fps
        self._elapsed = max(0.0, result.timestamp - self._active_start)
        self._point = list(result.point) if result.point else []
        self._bbox = list(result.bbox) if result.bbox else []
        self.telemetryChanged.emit()
        self.overlayChanged.emit()

    def _on_state(self, state: str, message: str):
        self._state = state
        self._status_message = message
        if state == "error" and message:
            self.errorOccurred.emit(message)
        self.stateChanged.emit()
        self.settingsChanged.emit()

    def _on_summary(self, summary: AnalysisSummary):
        if summary.run_id != self._run_id:
            return
        self._summary = {
            "completed": summary.completed,
            "duration": summary.duration_s,
            "endText": format_time(summary.end_sec, always_ms=True),
            "framesProcessed": summary.frames_processed,
            "framesDetected": summary.frames_detected,
            "detectionRate": summary.detection_rate,
            "avgSpeed": summary.avg_speed_m_s if summary.avg_speed_m_s is not None else -1.0,
            "maxSpeed": summary.max_speed_m_s if summary.max_speed_m_s is not None else -1.0,
        }
        self._smoothed = [[t, s] for t, s in zip(summary.smoothed_times, summary.smoothed_speeds)]
        self.summaryChanged.emit()

    # ---- navigation ------------------------------------------------------------------
    def _scrub_locked(self) -> bool:
        return self._state in RUNNING_STATES

    @Slot(float)
    def seek(self, sec: float):
        if self._video is None or self._scrub_locked():
            return
        sec = min(max(0.0, sec), self._duration)
        self._position = sec
        self.positionChanged.emit()
        self._bridge.player.seek(sec)

    @Slot(float)
    def stepSeconds(self, delta: float):
        self.seek(self._position + delta)

    @Slot(int)
    def stepFrames(self, count: int):
        if self._fps > 0:
            self.seek(self._position + count / self._fps)

    @Slot()
    def togglePreview(self):
        if self._state == "preview":
            self._bridge.player.pause()
        elif self._state in ("idle", "finished", "error"):
            self._bridge.player.play_preview()

    # ---- range -------------------------------------------------------------------------
    def _set_range(self, start: float, end: float):
        start = min(max(0.0, start), max(0.0, self._duration - MIN_RANGE_S))
        end = min(max(start + MIN_RANGE_S, end), self._duration)
        if (start, end) != (self._range_start, self._range_end):
            self._range_start, self._range_end = start, end
            self.rangeChanged.emit()
            self.settingsChanged.emit()

    @Slot(float)
    def setRangeStart(self, sec: float):
        self._set_range(sec, self._range_end if sec < self._range_end - MIN_RANGE_S else self._duration)

    @Slot(float)
    def setRangeEnd(self, sec: float):
        self._set_range(self._range_start if sec > self._range_start + MIN_RANGE_S else 0.0, sec)

    @Slot()
    def setRangeStartToPosition(self):
        self.setRangeStart(self._position)

    @Slot()
    def setRangeEndToPosition(self):
        self.setRangeEnd(self._position)

    @Slot(str, result=bool)
    def setRangeStartText(self, text: str) -> bool:
        try:
            self.setRangeStart(parse_time(text))
            return True
        except ValueError:
            return False

    @Slot(str, result=bool)
    def setRangeEndText(self, text: str) -> bool:
        try:
            self.setRangeEnd(parse_time(text))
            return True
        except ValueError:
            return False

    # ---- settings ----------------------------------------------------------------------
    @Slot(int)
    def setLane(self, lane_id: int):
        if lane_id in self._lane_ids and lane_id != self._lane_id:
            self._lane_id = lane_id
            if not self._scrub_locked():
                self._sync_preview_lane()
            self.settingsChanged.emit()

    @Slot(str)
    def setModel(self, name: str):
        if name in self._models and name != self._model_name:
            self._model_name = name
            self.settingsChanged.emit()

    @Slot(float)
    def setConfidence(self, value: float):
        value = round(min(max(0.05, value), 0.95), 2)
        if value != self._confidence:
            self._confidence = value
            self.settingsChanged.emit()

    # ---- analysis ----------------------------------------------------------------------
    @Slot()
    def analyze(self):
        if not self._can_analyze():
            return
        lane = self._video.lane(self._lane_id)
        self._run_id += 1
        self._active = self._settings()
        self._active_start = self._range_start
        self._reset_results()
        self._chart_duration = self._range_end - self._range_start
        self._bridge.take_samples(self._run_id)  # drop anything left over from earlier runs
        params = AnalysisParams(
            video_path=self.library.videos.file_path(self._video),
            lane_id=lane.id,
            lane_coords=tuple(tuple(p) for p in lane.coordinates),
            lane_length_m=lane.length_m,
            start_sec=self._range_start,
            end_sec=self._range_end,
            model_path=self.models_dir / self._model_name,
            confidence=self._confidence,
            run_id=self._run_id,
        )
        self._bridge.player.set_preview_lane(lane.coordinates)
        self._bridge.player.start_analysis(params)
        self.chartReset.emit()
        for signal in (self.telemetryChanged, self.overlayChanged, self.summaryChanged, self.settingsChanged):
            signal.emit()

    @Slot()
    def primaryAction(self):
        if self._state == "analyzing":
            self._bridge.player.pause()
        elif self._state == "paused":
            self._bridge.player.resume()
        else:
            self.analyze()

    @Slot()
    def restart(self):
        self.analyze()

    @Slot()
    def stop(self):
        self._bridge.player.stop()

    @Slot(result="QVariantList")
    def takeChartPoints(self) -> list:
        return [[t - self._active_start, s] for t, s in self._bridge.take_samples(self._run_id)]

    # ---- clips -------------------------------------------------------------------------
    @Slot(str)
    def applyClip(self, clip_id: str):
        clip = next((c for c in self._clips if c["id"] == clip_id), None)
        if clip is None:
            return
        if clip["lane"] not in self._lane_ids:
            self.errorOccurred.emit(f"Clip '{clip_id}' uses lane {clip['lane']}, which this video does not have.")
            return
        self.setLane(clip["lane"])
        self._set_range(clip["start"], clip["end"])
        self.seek(clip["start"])

    @Slot(result="QVariantMap")
    def proposeClip(self) -> dict:
        if self._video is None or self._lane_id not in self._lane_ids:
            return {"error": "Select a lane first."}
        clip, duplicate = propose_clip(self.library.clips, self._video_id, self._lane_id,
                                       self._range_start, self._range_end)
        return {
            "id": clip.id,
            "videoId": clip.video_id,
            "lane": clip.lane,
            "start": clip.start_time,
            "end": clip.end_time,
            "duration": self._range_end - self._range_start,
            "duplicate": duplicate.id if duplicate else "",
        }

    @Slot(str, result=str)
    def createClip(self, description: str) -> str:
        try:
            clip = create_clip(self.library.clips, self._video_id, self._lane_id,
                               self._range_start, self._range_end, description)
        except Exception as e:
            return str(e)
        self._load_clips()
        self.infoMessage.emit(f"Clip saved as '{clip.id}'.")
        return ""

    # ---- formatting helpers for QML ------------------------------------------------------
    @Slot(float, result=str)
    def formatTime(self, sec: float) -> str:
        return format_time(sec)

    @Slot(float, result=str)
    def formatTimeMs(self, sec: float) -> str:
        return format_time(sec, always_ms=True)

    def shutdown(self):
        self._bridge.close()
