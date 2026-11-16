import shutil
from pathlib import Path

from ultralytics import YOLO

from research.paths import MODELS_DIR, TRAINING_RUNS_DIR, WEIGHTS_DIR
from research.training.model_metadata import json_safe_value, parse_results_csv


def resolve_base_weights(base_weights: str) -> str:
    """A bare file name such as yolo11n.pt means the copy in cache/weights, ultralytics downloads it there."""
    path = Path(base_weights)
    if path.parent == Path("."):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS_DIR / path.name)
    return str(path)


def _extract_trainer_info(model: YOLO, run_dir: Path) -> dict:
    """Collect resolved runtime parameters and training summary from Ultralytics trainer and run files."""
    trainer = getattr(model, "trainer", None)
    info: dict = {}
    if trainer is not None:
        if getattr(trainer, "batch_size", None) is not None:
            info["resolved_batch_size"] = int(trainer.batch_size)
        if getattr(trainer, "device", None) is not None:
            info["resolved_device"] = str(trainer.device)
        if getattr(trainer, "amp", None) is not None:
            info["amp"] = bool(trainer.amp)
        if getattr(trainer, "epoch", None) is not None:
            info["epochs_completed"] = int(trainer.epoch) + 1
        trainer_args = getattr(trainer, "args", None)
        if trainer_args is not None:
            raw_args = vars(trainer_args) if hasattr(trainer_args, "__dict__") else dict(trainer_args)
            info["ultralytics_args"] = json_safe_value(raw_args)

    csv_summary = parse_results_csv(run_dir / "results.csv")
    for key in ("epochs_completed", "training_time_s", "best_epoch", "best_val_map50_95", "final_epoch_metrics"):
        if key in csv_summary:
            info[key] = csv_summary[key]
    return info


def train_with_info(
    model_name: str,
    dataset_yaml: Path,
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 16,
    rect: bool = True,
    seed: int = 0,
    base_weights: str = "yolo11n.pt",
    workers: int = 2,
    device: str | None = None,
) -> tuple[Path, dict]:
    """Train YOLO and return ``(copied_model_pt_path, resolved_training_info)``."""
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML {dataset_yaml} not found")

    model = YOLO(resolve_base_weights(base_weights))
    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "rect": rect,
        "batch": batch,
        "seed": seed,
        "project": str(TRAINING_RUNS_DIR),
        "name": model_name,
        "exist_ok": True,
        "workers": workers,
    }
    if device is not None:
        train_kwargs["device"] = device
    model.train(**train_kwargs)

    run_dir = TRAINING_RUNS_DIR / model_name
    best_src = run_dir / "weights" / "best.pt"
    if not best_src.exists():
        raise FileNotFoundError(f"The training did not produce {best_src}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / f"swimmer_{model_name}.pt"
    shutil.copy2(best_src, target)
    return target, _extract_trainer_info(model, run_dir)


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
    device: str | None = None,
) -> Path:
    model_path, _ = train_with_info(
        model_name=model_name,
        dataset_yaml=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        rect=rect,
        seed=seed,
        base_weights=base_weights,
        workers=workers,
        device=device,
    )
    return model_path
