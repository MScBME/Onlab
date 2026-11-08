"""Architecture guards.

The app package swimmertracker never imports the research code or the training stack, which keeps torch,
ultralytics and the rest out of the installer. Inside the app the UI layer stays free of CV and ML code and the core
stays free of Qt.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "swimmertracker"
APP_FORBIDDEN = ("research", "torch", "torchvision", "ultralytics", "roboflow", "matplotlib", "pandas", "dotenv",
                 "yaml", "onnx", "PyQt5")
UI_FORBIDDEN = ("cv2", "onnxruntime")
CORE_FORBIDDEN = ("PySide6", "swimmertracker.ui")


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _violations(paths, forbidden):
    return [
        f"{p.relative_to(ROOT)}: {m}"
        for p in paths
        for m in _imported_modules(p)
        if any(m == f or m.startswith(f + ".") for f in forbidden)
    ]


def test_app_does_not_import_research_or_training_stack():
    assert _violations(sorted(APP.rglob("*.py")), APP_FORBIDDEN) == []


def test_ui_does_not_import_cv_or_ml():
    assert _violations(sorted((APP / "ui").rglob("*.py")), UI_FORBIDDEN) == []


@pytest.mark.parametrize("package", ["analysis", "catalog", "detection", "playback", "tracking", "utils", "video"])
def test_core_does_not_import_qt(package):
    assert _violations(sorted((APP / package).rglob("*.py")), CORE_FORBIDDEN) == []
