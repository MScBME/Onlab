"""Where the app looks for its library and models, and which models it offers."""
import json
from pathlib import Path

from swimmertracker.catalog.model_catalog import list_models
from swimmertracker.utils import paths


def test_library_dir_from_source_env_and_frozen(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.LIBRARY_ENV, raising=False)
    monkeypatch.setattr(paths, "FROZEN", False)
    assert paths.default_library_dir(tmp_path / "Docs") == paths.PROJECT_ROOT / "data"

    monkeypatch.setattr(paths, "FROZEN", True)
    assert paths.default_library_dir(tmp_path / "Docs") == tmp_path / "Docs" / "Swimmer Tracker"

    monkeypatch.setenv(paths.LIBRARY_ENV, str(tmp_path / "mine"))
    assert paths.default_library_dir(tmp_path / "Docs") == tmp_path / "mine"


def test_models_dir_from_source_frozen_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.MODELS_ENV, raising=False)
    monkeypatch.setattr(paths, "FROZEN", False)
    assert paths.models_dir() == paths.PROJECT_ROOT / "models"

    monkeypatch.setattr(paths, "FROZEN", True)
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path / "_internal"), raising=False)
    assert paths.models_dir() == tmp_path / "_internal" / "models"

    monkeypatch.setenv(paths.MODELS_ENV, str(tmp_path / "m"))
    assert paths.models_dir() == tmp_path / "m"


def test_library_layout():
    lib = paths.LibraryPaths(Path("root"))
    assert (lib.videos_json, lib.clips_json, lib.videos_dir) == (
        Path("root/videos.json"), Path("root/clips.json"), Path("root/videos"))


def test_list_models_uses_metadata_and_skips_unsupported_inputs(tmp_path):
    for name in ("swimmer_all", "swimmer_freestyle", "frame_model"):
        (tmp_path / f"{name}.onnx").write_bytes(b"")
    (tmp_path / "swimmer_old.pt").write_bytes(b"")
    (tmp_path / "swimmer_all.json").write_text(json.dumps({"display_name": "All strokes"}))
    (tmp_path / "frame_model.json").write_text(json.dumps({"input": {"type": "frame"}}))
    (tmp_path / "swimmer_freestyle.json").write_text("{ not json")

    models = list_models(tmp_path)
    assert [(m.name, m.display_name) for m in models] == [
        ("swimmer_all", "All strokes"), ("swimmer_freestyle", "swimmer_freestyle")]
    assert models[0].path == tmp_path / "swimmer_all.onnx"
