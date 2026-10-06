"""Architecture guard: the UI layer stays free of CV/ML code and the core stays free of Qt."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
UI_FORBIDDEN = ("cv2", "ultralytics", "torch")
CORE_FORBIDDEN = ("PySide6", "PyQt5", "src.ui")


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _violations(paths, forbidden):
    return [
        f"{p.relative_to(SRC.parent)}: {m}"
        for p in paths
        for m in _imported_modules(p)
        if any(m == f or m.startswith(f + ".") for f in forbidden)
    ]


def test_ui_does_not_import_cv_or_ml():
    assert _violations(sorted((SRC / "ui").rglob("*.py")), UI_FORBIDDEN) == []


@pytest.mark.parametrize("package", ["analysis", "catalog", "detection", "playback", "tracking", "utils", "video"])
def test_core_does_not_import_qt(package):
    assert _violations(sorted((SRC / package).rglob("*.py")), CORE_FORBIDDEN) == []
