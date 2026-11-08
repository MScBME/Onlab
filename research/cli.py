"""Helpers of the research commands that print the problem and exit instead of raising."""
import json
import sys

from swimmertracker.utils.config import read_json
from swimmertracker.utils.timefmt import parse_time


def load_json(file_path):
    """Read a JSON file, or print the problem and exit."""
    try:
        return read_json(file_path)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"The file {file_path} is not valid JSON. {e}")
        sys.exit(1)


def time_to_seconds(time_str):
    """Parse an HH:MM:SS or HH:MM:SS.mmm time into seconds, or print the problem and exit. Empty gives None."""
    if not time_str:
        return None
    try:
        return parse_time(time_str)
    except ValueError:
        print(f"Times must look like HH:MM:SS, got '{time_str}'")
        sys.exit(1)
