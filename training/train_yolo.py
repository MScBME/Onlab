"""Train YOLOv11 swimmer detection model locally.

Usage:
    python training/train_yolo.py

Prerequisites:
    - Dataset downloaded to training/dataset/ (run download_dataset.py first)
    - training/dataset/data.yaml must exist
"""

import os
import shutil
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_YAML = os.path.join(TRAINING_DIR, "dataset", "data.yaml")
RUNS_DIR = os.path.join(TRAINING_DIR, "runs")
TARGET_MODEL = os.path.join(PROJECT_ROOT, "models", "swimmer.pt")


def main():
    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset not found at {DATASET_YAML}")
        print("Run 'python training/download_dataset.py' first.")
        return

    # YOLOv11 nano - fast and sufficient for single-class detection
    model = YOLO("yolo11n.pt")

    model.train(
        data=DATASET_YAML,
        epochs=50,
        imgsz=1024,
        rect=True,
        batch=16,
        project=RUNS_DIR,
        name="swimmer",
        exist_ok=True,
    )

    # Copy best model to models/swimmer.pt
    best_model = os.path.join(RUNS_DIR, "swimmer", "weights", "best.pt")
    if os.path.exists(best_model):
        os.makedirs(os.path.dirname(TARGET_MODEL), exist_ok=True)
        shutil.copy2(best_model, TARGET_MODEL)
        print(f"\nBest model copied to: {TARGET_MODEL}")
        print("You can now run: python scripts/run_yolo_offline.py")
    else:
        print("Warning: best.pt not found after training.")


if __name__ == "__main__":
    main()
