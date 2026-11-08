"""Research code that trains and evaluates the models, and legacy experiments. The app never imports it."""
import os

# ultralytics pip installs missing optional packages on the fly with whatever pip is first on PATH. The .venv of uv
# has no pip, so that can be another environment. Dependencies belong in pyproject.toml instead.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
