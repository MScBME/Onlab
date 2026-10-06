import shutil
from pathlib import Path

from ultralytics import YOLO

from src.utils.paths import MODELS_DIR, TRAINING_RUNS_DIR


def train(
    model_name: str,
    dataset_yaml: Path,
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 16,
    rect: bool = True,
    base_weights: str = "yolo11n.pt",
    workers: int = 2,
) -> Path:
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    model = YOLO(base_weights)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        rect=rect,
        batch=batch,
        project=str(TRAINING_RUNS_DIR),
        name=model_name,
        exist_ok=True,
        workers=workers,
    )

    best_src = TRAINING_RUNS_DIR / model_name / "weights" / "best.pt"
    if not best_src.exists():
        raise FileNotFoundError(f"best.pt not produced by training: {best_src}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / f"swimmer_{model_name}.pt"
    shutil.copy2(best_src, target)
    return target
