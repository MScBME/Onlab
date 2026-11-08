# PyInstaller spec of the installed app, a onedir build run through packaging/build.py,
# which stages the release models into packaging/build/models first.
from pathlib import Path

PACKAGING = Path(SPECPATH)
ROOT = PACKAGING.parent

# Qt modules that the PySide6 hooks collect with all QML plugins but the app never uses. WebEngine alone is about
# 200 MB. Quick3D stays, because QtGraphs of the speed chart is built on it.
UNUSED_QT = ("webengine", "qt6pdf", "qtpdf", "qt63d", "qml/qt3d", "location", "datavisualization", "charts",
             "qt6test", "qml/qttest")


def used(entry) -> bool:
    dest = entry[0].replace("\\", "/").lower()
    return not any(part in dest for part in UNUSED_QT)


a = Analysis(
    [str(PACKAGING / "launcher.py")],
    pathex=[str(ROOT)],
    datas=[
        (str(ROOT / "swimmertracker" / "ui" / "qml"), "swimmertracker/ui/qml"),
        (str(PACKAGING / "build" / "models"), "models"),
    ],
    hiddenimports=["swimmertracker.detection.onnx_detector"],  # imported lazily by the player
    excludes=[
        # the research and training stack is never part of the app, see tests/test_layering.py
        "research", "torch", "torchvision", "ultralytics", "roboflow", "matplotlib", "pandas", "polars", "scipy",
        "onnx", "onnxslim", "yaml", "dotenv", "PyQt5", "tkinter", "IPython", "pytest",
    ],
    noarchive=False,
)
a.binaries = [b for b in a.binaries if used(b)]
a.datas = [d for d in a.datas if used(d)]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SwimmerTracker",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SwimmerTracker")
