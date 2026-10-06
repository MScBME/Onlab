"""Video library: list model, background probing and thumbnails."""
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, unquote

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, QUrl, Signal, Slot, Property
from PySide6.QtGui import QColor, QImage
from PySide6.QtQuick import QQuickImageProvider

from src.catalog.library import VideoLibrary
from src.catalog.repository import VideoRepository
from src.ui.bridge.frames import bgr_to_qimage
from src.ui.bridge.qt_helpers import ro
from src.utils.timefmt import format_time
from src.video.loader import VideoLoader, probe_video

THUMB_WIDTH = 480
THUMB_AT_SEC = 1.0


class VideoListModel(QAbstractListModel):
    ROLES = ("videoId", "fileName", "durationText", "detailText", "laneCount", "clipCount", "missing", "thumbnail")
    countChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []

    def roleNames(self):
        base = int(Qt.ItemDataRole.UserRole) + 1
        return {base + i: name.encode() for i, name in enumerate(self.ROLES)}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        i = role - int(Qt.ItemDataRole.UserRole) - 1
        if 0 <= i < len(self.ROLES):
            return self._rows[index.row()].get(self.ROLES[i])
        return None

    def set_rows(self, rows: list):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()
        self.countChanged.emit()

    def update_row(self, video_id: str, **changes):
        for i, row in enumerate(self._rows):
            if row["videoId"] == video_id:
                row.update(changes)
                idx = self.index(i)
                self.dataChanged.emit(idx, idx)
                return

    count = Property(int, lambda self: len(self._rows), notify=countChanged)


class LibraryController(QObject):
    videosChanged = Signal()  # catalog content changed (video added, lanes edited)
    errorOccurred = Signal(str)
    _probed = Signal(str, object)

    def __init__(self, library: VideoLibrary, parent=None):
        super().__init__(parent)
        self.library = library
        self._model = VideoListModel(self)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="probe")
        self._probe_cache = {}
        self._revision = 0
        self._probed.connect(self._on_probed, Qt.ConnectionType.QueuedConnection)

    videos = Property(QObject, lambda self: self._model, constant=True)
    rawDirUrl = Property(QUrl, lambda self: QUrl.fromLocalFile(str(self.library.videos.raw_dir)), constant=True)

    @Slot()
    def refresh(self):
        try:
            videos = self.library.videos.list()
            clip_counts = Counter(c.video_id for c in self.library.clips.list())
        except Exception as e:
            self.errorOccurred.emit(f"Could not read the video catalog: {e}")
            return
        self._revision += 1
        rows = []
        for video in videos:
            path = self.library.videos.file_path(video)
            missing = not path.exists()
            rows.append({
                "videoId": video.id,
                "fileName": video.path,
                "durationText": "",
                "detailText": "File missing" if missing else "",
                "laneCount": len(video.lanes),
                "clipCount": clip_counts[video.id],
                "missing": missing,
                "thumbnail": "" if missing else f"image://thumbs/{quote(video.id)}?r={self._revision}",
            })
            if not missing:
                self._probe_async(video.id, path)
        self._model.set_rows(rows)

    @Slot()
    def notifyChanged(self):
        self.refresh()
        self.videosChanged.emit()

    def _probe_async(self, video_id, path):
        stat = path.stat()
        key = (str(path), stat.st_mtime, stat.st_size)
        if key in self._probe_cache:
            self._apply_info(video_id, self._probe_cache[key])
            return

        def work():
            try:
                info = probe_video(str(path))
                self._probe_cache[key] = info
            except Exception as e:
                info = e
            self._probed.emit(video_id, info)

        self._pool.submit(work)

    @Slot(str, object)
    def _on_probed(self, video_id, info):
        self._apply_info(video_id, info)

    def _apply_info(self, video_id, info):
        if isinstance(info, Exception):
            self._model.update_row(video_id, detailText="Unreadable video")
            return
        self._model.update_row(
            video_id,
            durationText=format_time(info.duration),
            detailText=f"{info.width}×{info.height} · {info.fps:.0f} fps",
        )

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


class ThumbnailProvider(QQuickImageProvider):
    """image://thumbs/<videoId>?r=<n> — a frame near the start of the video, cached in memory."""

    def __init__(self, repo: VideoRepository):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._repo = repo
        self._cache = {}
        self._lock = threading.Lock()

    def requestImage(self, image_id, size, requested_size):
        video_id = unquote(image_id.split("?")[0])
        with self._lock:
            cached = self._cache.get(video_id)
        if cached is not None:
            return cached
        image = self._render(video_id)
        with self._lock:
            self._cache[video_id] = image
        return image

    def _render(self, video_id: str) -> QImage:
        try:
            path = self._repo.file_path(self._repo.get(video_id))
            loader = VideoLoader(str(path))
            try:
                read = loader.read_at(min(THUMB_AT_SEC, loader.info.duration / 2))
            finally:
                loader.release()
            if read is not None:
                return bgr_to_qimage(read[1]).scaledToWidth(THUMB_WIDTH, Qt.TransformationMode.SmoothTransformation)
        except Exception:
            pass
        placeholder = QImage(THUMB_WIDTH, THUMB_WIDTH * 9 // 16, QImage.Format.Format_RGB32)
        placeholder.fill(QColor("#1b1f24"))
        return placeholder
