import json
import os
import sys
import tempfile
from pathlib import Path

from src.utils.timefmt import parse_time


def read_json(file_path):
    """Load a JSON file, raising FileNotFoundError / json.JSONDecodeError on failure."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(file_path):
    """CLI variant of read_json: prints the problem and exits instead of raising."""
    try:
        return read_json(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}.\nDetails: {e}")
        sys.exit(1)


def write_text_atomic(file_path, text: str):
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


def time_to_seconds(time_str):
    if not time_str:
        return None
    try:
        return parse_time(time_str)
    except ValueError:
        print(f"Error: Time format must be HH:MM:SS, got '{time_str}'")
        sys.exit(1)
