import json
import sys

def load_json(file_path):
    """A generic helper to load any JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}.\nDetails: {e}")
        sys.exit(1)