"""Desktop app bootstrap: Qt Quick (QML) UI on top of the UI-independent core."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "FluentWinUI3")

import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickWindow  # noqa: F401  (root objects are returned as QQuickWindow)

from src.catalog.library import VideoLibrary
from src.ui.bridge.importer import ImportController
from src.ui.bridge.lane_editor import LaneEditorController
from src.ui.bridge.library import LibraryController, ThumbnailProvider
from src.ui.bridge.workspace import WorkspaceController

QML_DIR = Path(__file__).resolve().parent / "qml"


@dataclass
class AppContext:
    engine: QQmlApplicationEngine
    library: LibraryController
    lane_editor: LaneEditorController
    importer: ImportController
    workspace: WorkspaceController
    thumbnails: ThumbnailProvider

    @property
    def window(self):
        return self.engine.rootObjects()[0]

    def shutdown(self):
        self.importer.cancel()
        self.workspace.shutdown()
        self.lane_editor.shutdown()
        self.library.shutdown()
        # tear the QML scene down before the controllers it binds to are garbage-collected
        shiboken6.delete(self.engine)


def create_application(argv) -> QGuiApplication:
    app = QGuiApplication(argv)
    app.setApplicationName("Swimmer Tracker")
    app.setOrganizationName("Onlab")
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    return app


def build(library: VideoLibrary = None) -> AppContext:
    library = library or VideoLibrary()
    library_ctrl = LibraryController(library)
    lane_editor = LaneEditorController(library)
    importer = ImportController(library, lane_editor)
    workspace = WorkspaceController(library)

    library_ctrl.videosChanged.connect(workspace.reloadVideo)
    lane_editor.saved.connect(lambda _video_id: library_ctrl.notifyChanged())
    importer.finished.connect(lambda _video_id: library_ctrl.notifyChanged())

    engine = QQmlApplicationEngine()
    thumbnails = ThumbnailProvider(library.videos)
    engine.addImageProvider("thumbs", thumbnails)
    ctx = engine.rootContext()
    ctx.setContextProperty("library", library_ctrl)
    ctx.setContextProperty("laneEditor", lane_editor)
    ctx.setContextProperty("importer", importer)
    ctx.setContextProperty("workspace", workspace)

    engine.load(QML_DIR / "Main.qml")
    if not engine.rootObjects():
        raise RuntimeError("Failed to load the QML user interface")
    library_ctrl.refresh()
    return AppContext(engine, library_ctrl, lane_editor, importer, workspace, thumbnails)


def main() -> int:
    app = create_application(sys.argv)
    try:
        context = build()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    exit_code = app.exec()
    context.shutdown()
    return exit_code
