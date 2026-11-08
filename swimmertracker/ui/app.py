"""Desktop app bootstrap, the Qt Quick UI on top of the UI independent core."""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "FluentWinUI3")

import shiboken6
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickWindow  # noqa: F401, root objects are returned as QQuickWindow

from swimmertracker.catalog.library import VideoLibrary
from swimmertracker.detection.self_check import check_models
from swimmertracker.ui.bridge.importer import ImportController
from swimmertracker.ui.bridge.lane_editor import LaneEditorController
from swimmertracker.ui.bridge.library import LibraryController, ThumbnailProvider
from swimmertracker.ui.bridge.workspace import WorkspaceController
from swimmertracker.utils.paths import LibraryPaths, default_library_dir, models_dir

QML_DIR = Path(__file__).resolve().parent / "qml"
APP_NAME = "Swimmer Tracker"
SMOKE_TEST_ENV = "SWIMMERTRACKER_SMOKE_TEST"  # a file path, the app starts, writes a startup report there and quits


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
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    return app


def parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swimmertracker", description="Swimmer Tracker desktop app.")
    parser.add_argument("--library", type=Path,
                        help="Library folder with videos.json, clips.json and videos/. The default is the repo's "
                             "data folder from source and Documents/Swimmer Tracker when installed")
    args, _qt_args = parser.parse_known_args(argv[1:])
    return args


def library_dir(args: argparse.Namespace) -> Path:
    if args.library:
        return args.library.resolve()
    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return default_library_dir(Path(documents) if documents else None)


def user_settings() -> QSettings:
    """Small disposable per user state such as the last model and confidence, in an ini file under %APPDATA%."""
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, APP_NAME, APP_NAME)


def build(library: VideoLibrary, settings: QSettings) -> AppContext:
    for folder in (library.videos.path.parent, library.videos.videos_dir):
        folder.mkdir(parents=True, exist_ok=True)
    library_ctrl = LibraryController(library)
    lane_editor = LaneEditorController(library)
    importer = ImportController(library, lane_editor)
    workspace = WorkspaceController(library, models_dir(), settings)

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


def write_smoke_report(path: Path, context: AppContext):
    models = check_models(models_dir())
    report = {
        "qml_loaded": bool(context.engine.rootObjects()),
        "library": str(context.library.library.videos.path.parent),
        "videos": context.library.videos.rowCount(),
        "models": models,
        "ok": bool(models) and all(m["ok"] for m in models),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args(sys.argv)
    app = create_application(sys.argv)
    smoke_report = os.environ.get(SMOKE_TEST_ENV)
    try:
        context = build(VideoLibrary.open(LibraryPaths(library_dir(args))), user_settings())
    except RuntimeError as e:
        print(e, file=sys.stderr)
        if smoke_report:
            Path(smoke_report).write_text(json.dumps({"ok": False, "error": str(e)}), encoding="utf-8")
        return 1
    if smoke_report:
        def finish_smoke_test():
            try:
                write_smoke_report(Path(smoke_report), context)
            finally:  # quit even if the report fails, the build then reports the missing report
                app.quit()

        QTimer.singleShot(500, finish_smoke_test)
    exit_code = app.exec()
    context.shutdown()
    return exit_code
