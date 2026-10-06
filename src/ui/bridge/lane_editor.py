"""Lane editor: define/edit the 4-corner lane quads of a video on any of its frames."""
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, Signal, Slot, Property

from src.catalog.library import MAX_LANE_ID, VideoLibrary, prepare_lanes
from src.catalog.models import DEFAULT_LANE_LENGTH_M, Lane
from src.ui.bridge.frames import FrameSink
from src.ui.bridge.player_bridge import PlayerBridge
from src.ui.bridge.qt_helpers import ro
from src.video.lane_warp import normalize_corners, validate_corners
from src.video.loader import VideoInfo, probe_video


TRAINING_NOTICE = "Lane coordinates are shared with the training pipeline (train_model.py)."


def _midpoint(a, b) -> list:
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


class LaneListModel(QAbstractListModel):
    """Rows keep their delegates alive on edits (dataChanged), so text fields don't lose focus."""
    ROLES = ("laneId", "points", "lengthM", "isNew", "deletable", "refText", "startLabel", "endLabel", "valid")
    countChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def roleNames(self):
        base = int(Qt.ItemDataRole.UserRole) + 1
        return {base + i: name.encode() for i, name in enumerate(self.ROLES)}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        i = role - int(Qt.ItemDataRole.UserRole) - 1
        if 0 <= i < len(self.ROLES):
            return self.rows[index.row()].get(self.ROLES[i])
        return None

    def reset(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()
        self.countChanged.emit()

    def append(self, row):
        self.beginInsertRows(QModelIndex(), len(self.rows), len(self.rows))
        self.rows.append(row)
        self.endInsertRows()
        self.countChanged.emit()

    def remove(self, i):
        self.beginRemoveRows(QModelIndex(), i, i)
        del self.rows[i]
        self.endRemoveRows()
        self.countChanged.emit()

    def touch(self, i):
        idx = self.index(i)
        self.dataChanged.emit(idx, idx)

    count = Property(int, lambda self: len(self.rows), notify=countChanged)


class LaneEditorController(QObject):
    changed = Signal()
    selectionChanged = Signal()
    frameTimeChanged = Signal()
    saved = Signal(str)
    errorOccurred = Signal(str)

    def __init__(self, library: VideoLibrary, parent=None):
        super().__init__(parent)
        self.library = library
        self._model = LaneListModel(self)
        self._bridge = PlayerBridge(self)
        self._bridge.frameReady.connect(self._on_frame)
        self._frame_sink, self._warp_sink = FrameSink(), FrameSink()
        self._last_result = None

        self._mode = "new"
        self._video_id = ""
        self._info: Optional[VideoInfo] = None
        self._frame_w = self._frame_h = 0
        self._duration = 0.0
        self._frame_time = 0.0
        self._original_ids = set()
        self._references = {}  # lane id -> LaneReferences (edit mode)
        self._selected = -1
        self._errors = []
        self._notice = ""
        self._copy_sources = []

    # ---- properties --------------------------------------------------------------------
    lanes = Property(QObject, lambda self: self._model, constant=True)
    mode = ro(str, "_mode", changed)
    videoId = ro(str, "_video_id", changed)
    frameWidth = ro(int, "_frame_w", changed)
    frameHeight = ro(int, "_frame_h", changed)
    duration = ro(float, "_duration", changed)
    frameTime = ro(float, "_frame_time", frameTimeChanged)
    selectedIndex = ro(int, "_selected", selectionChanged)
    errors = ro("QVariantList", "_errors", changed)
    notice = ro(str, "_notice", changed)
    copySources = ro("QVariantList", "_copy_sources", changed)
    maxLaneId = Property(int, lambda self: MAX_LANE_ID, constant=True)

    def _selected_points(self):
        if 0 <= self._selected < len(self._model.rows):
            return self._model.rows[self._selected]["points"]
        return []

    selectedPoints = Property("QVariantList", _selected_points, notify=selectionChanged)
    selectedLaneId = Property(
        int,
        lambda self: self._model.rows[self._selected]["laneId"] if 0 <= self._selected < len(self._model.rows) else 0,
        notify=selectionChanged,
    )
    canSave = Property(bool, lambda self: bool(self._model.rows) and not self._errors, notify=changed)

    # ---- opening -----------------------------------------------------------------------
    def open_for_new(self, path: Path, info: VideoInfo):
        self._mode, self._video_id = "new", ""
        self._original_ids, self._references = set(), {}
        self._load(path, info, [])

    @Slot(str, result=bool)
    def openForEdit(self, video_id: str) -> bool:
        try:
            video = self.library.videos.get(video_id)
            path = self.library.videos.file_path(video)
            info = probe_video(str(path))
        except Exception as e:
            self.errorOccurred.emit(f"Cannot open video '{video_id}': {e}")
            return False
        self._mode, self._video_id = "edit", video_id
        self._original_ids = {lane.id for lane in video.lanes}
        self._references = {lane.id: self.library.lane_references(video_id, lane.id) for lane in video.lanes}
        self._load(path, info, video.lanes)
        return True

    def _load(self, path: Path, info: VideoInfo, lanes: list):
        self._info = info
        self._frame_w, self._frame_h, self._duration = info.width, info.height, info.duration
        self._frame_time = 0.0
        self._notice = TRAINING_NOTICE if self._mode == "edit" else ""
        self._copy_sources = sorted(i for i in self.library.videos.ids() if i != self._video_id)
        self._model.reset([self._row(lane) for lane in lanes])
        self._selected = 0 if lanes else -1
        self._bridge.player.open(path)
        self._validate()
        self._sync_preview_lane()
        self.selectionChanged.emit()
        self.frameTimeChanged.emit()

    def _row(self, lane: Lane) -> dict:
        refs = self._references.get(lane.id)
        row = {
            "laneId": lane.id,
            "points": [list(map(float, p)) for p in lane.coordinates],
            "lengthM": float(lane.length_m),
            "isNew": lane.id not in self._original_ids,
            "deletable": not (refs and refs.any),
            "refText": f"Used by {refs.describe()}" if refs and refs.any else "",
        }
        self._update_geometry(row)
        return row

    def _update_geometry(self, row: dict):
        try:
            ordered = normalize_corners(row["points"])
            valid = not validate_corners(ordered, (self._frame_w, self._frame_h))
        except ValueError:
            ordered, valid = None, False
        row["valid"] = valid
        row["startLabel"] = _midpoint(ordered[0], ordered[1]) if valid else []
        row["endLabel"] = _midpoint(ordered[2], ordered[3]) if valid else []

    # ---- frames -----------------------------------------------------------------------
    @Slot(QObject)
    def setFrameSink(self, sink):
        self._frame_sink.attach(sink)
        if self._last_result is not None:  # the frame may have arrived before this view was attached
            self._frame_sink.push(self._last_result.frame)

    @Slot(QObject)
    def setWarpSink(self, sink):
        self._warp_sink.attach(sink)
        if self._last_result is not None:
            self._warp_sink.push(self._last_result.warped)

    @Slot(float)
    def seek(self, sec: float):
        self._bridge.player.seek(sec)

    def _on_frame(self, result):
        self._last_result = result
        self._frame_sink.push(result.frame)
        self._warp_sink.push(result.warped)
        self._frame_time = result.timestamp
        self.frameTimeChanged.emit()

    def _sync_preview_lane(self):
        rows = self._model.rows
        row = rows[self._selected] if 0 <= self._selected < len(rows) else None
        self._bridge.player.set_preview_lane(row["points"] if row and row["valid"] else None)

    # ---- editing ----------------------------------------------------------------------
    def _clamp(self, x: float, y: float) -> list:
        return [min(max(0.0, float(x)), float(self._frame_w)), min(max(0.0, float(y)), float(self._frame_h))]

    @Slot("QVariantList")
    def addLane(self, points):
        if len(points) != 4:
            return
        pts = [self._clamp(p[0], p[1]) for p in points]
        try:
            pts = normalize_corners(pts)
        except ValueError:
            pass
        used = {row["laneId"] for row in self._model.rows}
        lane_id = next((i for i in range(1, MAX_LANE_ID + 1) if i not in used), MAX_LANE_ID)
        self._model.append(self._row(Lane(lane_id, pts, DEFAULT_LANE_LENGTH_M)))
        self.selectLane(len(self._model.rows) - 1)
        self._validate()

    @Slot(int, int, float, float)
    def moveCorner(self, row_index: int, corner: int, x: float, y: float):
        if not 0 <= row_index < len(self._model.rows) or not 0 <= corner < 4:
            return
        row = self._model.rows[row_index]
        row["points"][corner] = self._clamp(x, y)
        self._update_geometry(row)
        self._model.touch(row_index)
        if row_index == self._selected:
            self.selectionChanged.emit()
            self._sync_preview_lane()
        self._validate()

    @Slot(int, int)
    def setLaneId(self, row_index: int, lane_id: int):
        if 0 <= row_index < len(self._model.rows) and self._model.rows[row_index]["isNew"]:
            self._model.rows[row_index]["laneId"] = int(lane_id)
            self._model.touch(row_index)
            if row_index == self._selected:
                self.selectionChanged.emit()
            self._validate()

    @Slot(int, str)
    def setLaneLength(self, row_index: int, text: str):
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            return
        if 0 <= row_index < len(self._model.rows):
            self._model.rows[row_index]["lengthM"] = value
            self._model.touch(row_index)
            self._validate()

    @Slot(int)
    def removeLane(self, row_index: int):
        if not 0 <= row_index < len(self._model.rows):
            return
        row = self._model.rows[row_index]
        if not row["deletable"]:
            self.errorOccurred.emit(f"Lane {row['laneId']} cannot be removed. {row['refText']}.")
            return
        self._model.remove(row_index)
        if self._selected >= len(self._model.rows):
            self._selected = len(self._model.rows) - 1
        self.selectionChanged.emit()
        self._sync_preview_lane()
        self._validate()

    @Slot(int)
    def selectLane(self, row_index: int):
        self._selected = row_index if 0 <= row_index < len(self._model.rows) else -1
        self.selectionChanged.emit()
        self._sync_preview_lane()

    @Slot(str)
    def copyLanesFrom(self, video_id: str):
        try:
            source = self.library.videos.get(video_id)
            source_info = probe_video(str(self.library.videos.file_path(source)))
            notice = ""
            if (source_info.width, source_info.height) != (self._frame_w, self._frame_h):
                notice = (f"Lanes were copied from a {source_info.width}×{source_info.height} video, "
                          f"this one is {self._frame_w}×{self._frame_h}. Check the corners.")
        except KeyError:
            self.errorOccurred.emit(f"Video '{video_id}' not found.")
            return
        except Exception:
            notice = "Could not read the source video; check the copied corners."
        lanes = [Lane(lane.id, [self._clamp(*p) for p in lane.coordinates], lane.length_m) for lane in source.lanes]
        self._model.reset([self._row(lane) for lane in lanes])
        self._notice = notice
        self.selectLane(0)
        self._validate()

    def _current_lanes(self) -> list:
        return [Lane(r["laneId"], r["points"], r["lengthM"]) for r in self._model.rows]

    def _validate(self):
        _, errors = prepare_lanes(self._current_lanes(), (self._frame_w, self._frame_h))
        present = {r["laneId"] for r in self._model.rows}
        for lane_id in sorted(self._original_ids - present):
            refs = self._references.get(lane_id)
            if refs and refs.any:
                errors.append(f"Lane {lane_id} cannot be removed: it is used by {refs.describe()}.")
        self._errors = errors
        self.changed.emit()

    # ---- saving -----------------------------------------------------------------------
    def lanes_for_save(self) -> list:
        return self._current_lanes()

    def frame_size(self) -> tuple:
        return self._frame_w, self._frame_h

    @Slot(result=bool)
    def save(self) -> bool:
        if self._mode != "edit":
            return False
        try:
            self.library.save_lanes(self._video_id, self._current_lanes(), self.frame_size())
        except Exception as e:
            self._errors = str(e).splitlines()
            self.changed.emit()
            return False
        self.saved.emit(self._video_id)
        self.close()
        return True

    @Slot()
    def close(self):
        self._last_result = None
        self._bridge.player.release()

    def shutdown(self):
        self._bridge.close()
