import shutil
from pathlib import Path

from ultralytics import YOLO

from research.paths import MODELS_DIR, TRAINING_RUNS_DIR, WEIGHTS_DIR


def resolve_base_weights(base_weights: str) -> str:
    """A bare file name such as yolo11n.pt means the copy in cache/weights, ultralytics downloads it there."""
    path = Path(base_weights)
    if path.parent == Path("."):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS_DIR / path.name)
    return str(path)


def train(
    model_name: str,
    dataset_yaml: Path,
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 16,
    rect: bool = True,
    seed: int = 0,
    base_weights: str = "yolo11n.pt",
    workers: int = 2,
) -> Path:
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML {dataset_yaml} not found")

    model = YOLO(resolve_base_weights(base_weights))
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        rect=rect,
        batch=batch,
        seed=seed,
        project=str(TRAINING_RUNS_DIR),
        name=model_name,
        exist_ok=True,
        workers=workers,
    )

    best_src = TRAINING_RUNS_DIR / model_name / "weights" / "best.pt"
    if not best_src.exists():
        raise FileNotFoundError(f"The training did not produce {best_src}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / f"swimmer_{model_name}.pt"
    shutil.copy2(best_src, target)
    return target
