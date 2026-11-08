"""Reading and atomic writing of the JSON files of the library and the models."""
import json
import os
import tempfile
from pathlib import Path


def read_json(file_path):
    """Load a JSON file. Raises FileNotFoundError or json.JSONDecodeError on failure."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text_atomic(file_path, text: str):
    """Write through a temporary file and a rename, so a crash never leaves a half written file behind."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_json(file_path, data, indent=2):
    write_text_atomic(file_path, json.dumps(data, indent=indent) + "\n")
