import json
import sys

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}.\nDetails: {e}")
        sys.exit(1)

def time_to_seconds(time_str):
    if not time_str:
        return None
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        print(f"Error: Time format must be HH:MM:SS, got '{time_str}'")
        sys.exit(1)