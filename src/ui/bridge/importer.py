"""'Add video' flow: probe the file, copy it into data/raw in the background, register it with its lanes."""
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot

from src.catalog.library import (
    VIDEO_EXTENSIONS,
    CopyCancelled,
    VideoLibrary,
    copy_with_progress,
    import_target,
    suggest_video_id,
    validate_video_id,
)
from src.ui.bridge.lane_editor import LaneEditorController
from src.ui.bridge.qt_helpers import ro
from src.utils.timefmt import format_time
from src.video.loader import probe_video


class ImportController(QObject):
    changed = Signal()
    progressChanged = Signal()
    finished = Signal(str)  # new video id
    errorOccurred = Signal(str)
    _copyProgress = Signal(int, float)  # import generation, fraction
    _copyEnded = Signal(int, str, str)  # import generation, outcome ("done" | "failed" | "cancelled"), message

    def __init__(self, library: VideoLibrary, editor: LaneEditorController, parent=None):
        super().__init__(parent)
        self.library = library
        self.editor = editor
        self._copyProgress.connect(self._on_copy_progress, Qt.ConnectionType.QueuedConnection)
        self._copyEnded.connect(self._on_copy_ended, Qt.ConnectionType.QueuedConnection)
        self._cancel: Optional[threading.Event] = None
        self._generation = 0
        self._reset()

    def _reset(self):
        self._generation += 1  # results of earlier (abandoned) copies are ignored from now on
        self._source: Optional[Path] = None
        self._target: Optional[Path] = None
        self._frame_size = (0, 0)
        self._file_name = ""
        self._info_text = ""
        self._target_text = ""
        self._needs_copy = False
        self._copy_state = ""  # "", "copying", "done", "failed", "cancelled"
        self._copy_progress = 0.0
        self._video_id = ""
        self._video_id_error = ""
        self._save_pending = False

    fileName = ro(str, "_file_name", changed)
    infoText = ro(str, "_info_text", changed)
    targetText = ro(str, "_target_text", changed)
    needsCopy = ro(bool, "_needs_copy", changed)
    copyState = ro(str, "_copy_state", changed)
    copyProgress = ro(float, "_copy_progress", progressChanged)
    videoId = ro(str, "_video_id", changed)
    videoIdError = ro(str, "_video_id_error", changed)
    savePending = ro(bool, "_save_pending", changed)

    @Slot(QUrl, result=bool)
    def begin(self, url: QUrl) -> bool:
        self.cancel()
        source = Path(url.toLocalFile())
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            self.errorOccurred.emit(f"Unsupported file type '{source.suffix}'. Use one of: {', '.join(VIDEO_EXTENSIONS)}.")
            return False
        try:
            info = probe_video(str(source))
        except Exception:
            self.errorOccurred.emit(f"'{source.name}' cannot be read as a video.")
            return False

        raw_dir = self.library.videos.raw_dir
        self._source, self._target = source, import_target(source, raw_dir)
        self._frame_size = (info.width, info.height)
        self._file_name = source.name
        self._needs_copy = self._target != source
        self._target_text = f"data/raw/{self._target.name}"
        self._info_text = f"{format_time(info.duration)} · {info.width}×{info.height} · {info.fps:.0f} fps"
        users = [v.id for v in self.library.videos.list() if not self._needs_copy
                 and (raw_dir / v.path).resolve() == source.resolve()]
        if users:
            self._info_text += f" · already registered as: {', '.join(users)}"
        existing = self.library.videos.ids()
        self._video_id = suggest_video_id(source.name, existing)
        self._video_id_error = validate_video_id(self._video_id, existing) or ""
        self._copy_state = "copying" if self._needs_copy else "done"
        self._copy_progress = 0.0 if self._needs_copy else 1.0
        self.changed.emit()
        self.progressChanged.emit()

        if self._needs_copy:
            self._start_copy(source, self._target)
        self.editor.open_for_new(source, info)
        return True

    def _start_copy(self, source: Path, target: Path):
        cancel = threading.Event()
        self._cancel = cancel
        generation = self._generation
        last = [0.0]

        def on_progress(fraction):
            if fraction - last[0] >= 0.005 or fraction >= 1.0:
                last[0] = fraction
                self._copyProgress.emit(generation, fraction)

        def work():
            try:
                copy_with_progress(source, target, on_progress, cancel)
                self._copyEnded.emit(generation, "done", "")
            except CopyCancelled:
                self._copyEnded.emit(generation, "cancelled", "")
            except Exception as e:
                self._copyEnded.emit(generation, "failed", str(e))

        threading.Thread(target=work, name="video-copy", daemon=True).start()

    @Slot(int, float)
    def _on_copy_progress(self, generation, fraction):
        if generation == self._generation and self._copy_state == "copying":
            self._copy_progress = fraction
            self.progressChanged.emit()

    @Slot(int, str, str)
    def _on_copy_ended(self, generation, outcome, message):
        if generation != self._generation or self._copy_state != "copying":
            return  # a cancelled/abandoned import finishing in the background
        self._copy_state = outcome
        if outcome == "failed":
            self._save_pending = False
            self.errorOccurred.emit(f"Copying the video failed: {message}")
        self.changed.emit()
        if outcome == "done" and self._save_pending:
            self._complete_save()

    @Slot(str)
    def setVideoId(self, text: str):
        self._video_id = text.strip()
        self._video_id_error = validate_video_id(self._video_id, self.library.videos.ids()) or ""
        self.changed.emit()

    @Slot()
    def save(self):
        if self._source is None:
            return
        self.setVideoId(self._video_id)
        if self._video_id_error or not self.editor.canSave:
            self.errorOccurred.emit(self._video_id_error or "Fix the lane errors before saving.")
            return
        if self._copy_state == "copying":
            self._save_pending = True
            self.changed.emit()
            return
        if self._copy_state != "done":
            self.errorOccurred.emit("The video file is not available in data/raw.")
            return
        self._complete_save()

    def _complete_save(self):
        self._save_pending = False
        try:
            video = self.library.add_video(self._video_id, self._target, self.editor.lanes_for_save(), self._frame_size)
        except Exception as e:
            self.changed.emit()
            self.errorOccurred.emit(str(e))
            return
        self.editor.close()
        self._reset()
        self.changed.emit()
        self.finished.emit(video.id)

    @Slot()
    def cancel(self):
        """Abort the import: stop a running copy and remove a file this import created."""
        if self._source is None:
            return
        if self._copy_state == "copying" and self._cancel is not None:
            self._cancel.set()  # the copy thread removes its .part file
        elif self._needs_copy and self._copy_state == "done" and self._target is not None:
            self._target.unlink(missing_ok=True)
        self.editor.close()
        self._reset()
        self.changed.emit()
        self.progressChanged.emit()
