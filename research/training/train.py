"""Full-frame swimmer detector training command (train-model).

Loads locally curated datasets from disk (e.g. ``public_annotations/``), filters
samples by rejection status, quality ratings and dataset name, builds a
video-grouped full-frame YOLO train/val dataset without lane warping, trains and
evaluates the model, and records metadata (plus optional ONNX export).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from research.paths import DATASET_DIR, PUBLIC_ANNOTATIONS_DIR, TRAINING_RUNS_DIR
from research.training.curated_dataset import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    IS_REJECT_CHOICES,
    SPLITS,
    build_full_frame_dataset,
    filter_curated_samples,
    load_curated_samples,
    validate_datasets,
)
from research.training.model_metadata import (
    environment_info,
    full_frame_input,
    git_info,
    metrics_from,
    update_metadata,
)
from research.training.onnx_export import export_onnx
from research.training.trainer import train_with_info
from swimmertracker.utils.paths import PROJECT_ROOT

STEPS = 6


def step(n: int, text: str) -> None:
    print(f"\n[{n}/{STEPS}] {text}")


def describe_counts(counts: dict) -> str:
    return ", ".join(f"{key} {value}" for key, value in counts.items()) or "none"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train-model",
        description="Train a full-frame YOLO swimmer detector on locally curated datasets without lane warping.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Model name suffix, the output is models/swimmer_<name>.pt and .json",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=PUBLIC_ANNOTATIONS_DIR,
        help="Directory with curation_ratings.json and dataset folders (default %(default)s)",
    )
    parser.add_argument(
        "--is-reject",
        type=str.lower,
        choices=IS_REJECT_CHOICES,
        default="false",
        help="Filter by curation is_reject flag: 'false' (default), 'true' (only rejected), or 'all'",
    )
    parser.add_argument(
        "--min-image-quality",
        type=int,
        default=None,
        help="Minimum image_quality score required (samples with null rating are excluded)",
    )
    parser.add_argument(
        "--min-annot-quality",
        type=int,
        default=None,
        help="Minimum annot_quality score required (samples with null rating are excluded)",
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        default=None,
        help="Dataset folder names to include (default: all datasets present in curation_ratings.json)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Target validation fraction when falling back to seeded random splitting, default %(default)s",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed for fallback random train/val splitting, default %(default)s",
    )
    parser.add_argument(
        "--val-videos",
        nargs="+",
        default=None,
        help="Filename glob patterns reserved for val (overrides --val-fraction)",
    )
    parser.add_argument(
        "--val-split-dir",
        type=Path,
        default=None,
        help="Directory with curated validation images (e.g. split_clean or its validate/ subfolder; overrides --val-videos and --val-fraction)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs, default %(default)s")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size in pixels, default %(default)s")
    parser.add_argument("--batch", type=int, default=16, help="Batch size, default %(default)s")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed of ultralytics, default %(default)s",
    )
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rectangular batches without mosaic augmentation (default --no-rect uses mosaic on full frames)",
    )
    parser.add_argument(
        "--base-weights",
        default="yolo11n.pt",
        help="Starting weights, default %(default)s. A bare file name uses research/training/cache/weights",
    )
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers, default %(default)s")
    parser.add_argument("--device", default=None, help="Training device override, e.g. '0' or 'cpu'")
    parser.add_argument("--skip-train", action="store_true", help="Stop after preparing the YOLO dataset")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Also export the trained model to ONNX after evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    annotations_dir = args.annotations_dir.resolve()

    model_name = f"swimmer_{args.name}"
    print("=" * 60)
    print(f"Training full-frame model {model_name}")
    print(f"Annotations dir   {annotations_dir}")
    print(f"Filter is_reject  {args.is_reject}")
    print(f"Min image quality {args.min_image_quality if args.min_image_quality is not None else 'none'}")
    print(f"Min annot quality {args.min_annot_quality if args.min_annot_quality is not None else 'none'}")
    print(f"Datasets          {', '.join(args.datasets) if args.datasets else 'all'}")
    val_split_dir = args.val_split_dir.resolve() if args.val_split_dir is not None else None
    if val_split_dir is not None:
        print(f"Val split dir     {val_split_dir}")
    elif args.val_videos:
        print(f"Val video filters {', '.join(args.val_videos)}")
    else:
        print(f"Val split         {args.val_fraction:.2f} (seed={args.split_seed}, random)")
    print("=" * 60)

    step(1, "Loading curated annotations from disk...")
    try:
        all_samples, available_datasets = load_curated_samples(annotations_dir)
        selected_datasets = validate_datasets(args.datasets, available_datasets)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(f"Loaded {len(all_samples)} curated entries across {len(available_datasets)} datasets.")

    step(2, "Filtering samples...")
    selected = filter_curated_samples(
        all_samples,
        is_reject=args.is_reject,
        min_image_quality=args.min_image_quality,
        min_annot_quality=args.min_annot_quality,
        datasets=selected_datasets,
    )
    print(f"{len(selected)} of {len(all_samples)} samples selected.")
    if not selected:
        print("No samples match the filters. Aborting.")
        sys.exit(1)

    step(3, "Preparing full-frame train/val YOLO dataset (no lane warping)...")
    try:
        split = build_full_frame_dataset(
            selected,
            out_dir=DATASET_DIR,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            val_videos=args.val_videos,
            val_split_dir=val_split_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    for name in SPLITS:
        s = split[name]
        print(
            f"  {name:5} {s['images']:4} images ({s['boxes']:5} boxes, "
            f"{s['empty_label_images']} empty), {describe_counts(s['by_dataset'])}"
        )
    if split["missing_image"] or split["missing_label"]:
        print(
            f"  Skipped assets: {split['missing_image']} missing images, "
            f"{split['missing_label']} missing label files"
        )
    print(f"Dataset YAML written to {split['yaml_path']}")

    if args.skip_train:
        step(4, "Stopping before training as --skip-train was given.")
        return

    step(4, "Training YOLO on full frames...")
    model_path, trainer_info = train_with_info(
        model_name=args.name,
        dataset_yaml=split["yaml_path"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        rect=args.rect,
        seed=args.seed,
        base_weights=args.base_weights,
        workers=args.workers,
        device=args.device,
    )
    print(f"\nModel saved to {model_path}")

    step(5, "Evaluating on val...")
    final_model = YOLO(model_path)
    metrics = {
        "val": metrics_from(
            final_model.val(split="val", project=str(TRAINING_RUNS_DIR), name=f"{args.name}_val", exist_ok=True),
            split["val"]["images"],
        ),
        "test": None,
    }
    if "final_epoch_metrics" in trainer_info:
        metrics["final_epoch"] = trainer_info["final_epoch_metrics"]
    print(f"Val mAP50-95  {metrics['val']['map50_95']:.4f}")

    filters_meta = {
        "is_reject": args.is_reject,
        "min_image_quality": args.min_image_quality,
        "min_annot_quality": args.min_annot_quality,
        "datasets": selected_datasets,
    }
    resolved_meta = {
        k: trainer_info[k]
        for k in ("amp", "epochs_completed", "training_time_s", "best_epoch", "best_val_map50_95")
        if k in trainer_info
    }
    if "resolved_batch_size" in trainer_info:
        resolved_meta["batch_size"] = trainer_info["resolved_batch_size"]
    if "resolved_device" in trainer_info:
        resolved_meta["device"] = trainer_info["resolved_device"]

    training_meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "environment": environment_info(trainer_info.get("resolved_device")),
        "annotations_dir": annotations_dir.as_posix(),
        "filters": filters_meta,
        "split": {
            k: split[k]
            for k in (
                "method",
                "val_fraction",
                "seed",
                "val_videos_patterns",
                "val_split_dir",
                "missing_image",
                "missing_label",
                "empty_label",
                "total_boxes",
                "train",
                "val",
            )
        },
        "args": {
            "base_weights": args.base_weights,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "rect": args.rect,
            "seed": args.seed,
            "workers": args.workers,
            "device": args.device,
            "val_fraction": args.val_fraction,
            "split_seed": args.split_seed,
            "val_videos": args.val_videos,
            "val_split_dir": val_split_dir.as_posix() if val_split_dir is not None else None,
            "export_onnx": args.export_onnx,
        },
        "resolved": resolved_meta,
        "run_dir": (
            (TRAINING_RUNS_DIR / args.name).relative_to(PROJECT_ROOT).as_posix()
            if (TRAINING_RUNS_DIR / args.name).is_relative_to(PROJECT_ROOT)
            else (TRAINING_RUNS_DIR / args.name).as_posix()
        ),
    }
    if "ultralytics_args" in trainer_info:
        training_meta["ultralytics_args"] = trainer_info["ultralytics_args"]

    update_metadata(
        model_path,
        input=full_frame_input(args.imgsz),
        classes=["swimmer"],
        training=training_meta,
        metrics=metrics,
    )

    if args.export_onnx:
        step(6, "Exporting to ONNX...")
        try:
            onnx_path = export_onnx(model_path)
            print(f"ONNX exported to {onnx_path}")
        except ModuleNotFoundError as exc:
            print(f"[WARN] ONNX export skipped because a dependency is missing: {exc}")
    else:
        step(6, "Skipping ONNX export (pass --export-onnx to export).")

    metadata_name = model_path.with_suffix(".json").name
    print(f"Done. {model_path.name}, {metadata_name} are in {model_path.parent}")


if __name__ == "__main__":
    main()
