"""Models the app can run, the <name>.onnx files of the models folder, each with an optional <name>.json metadata file.

The training writes the metadata file, see research/training/model_metadata.py. It gives the name shown in the UI
and the input type of the model. Only lane_warp models, which detect on the warped lane image, are supported here.
"""
import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_INPUT = "lane_warp"


@dataclass(frozen=True)
class ModelInfo:
    name: str  # file stem, the stable id that the settings store
    path: Path
    display_name: str


def read_metadata(onnx_path: Path) -> dict:
    try:
        return json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def list_models(models_dir: Path) -> list:
    models = []
    for path in sorted(Path(models_dir).glob("*.onnx")):
        metadata = read_metadata(path)
        if metadata.get("input", {}).get("type", SUPPORTED_INPUT) != SUPPORTED_INPUT:
            continue
        models.append(ModelInfo(name=path.stem, path=path, display_name=metadata.get("display_name") or path.stem))
    return models
