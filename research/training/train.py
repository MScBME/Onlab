"""Pipeline 2, the train-model command.

Downloads the annotated frames, warps the lanes, splits the data, trains and evaluates the model, then exports it
to ONNX for the app. The result is models/swimmer_<name>.pt, .onnx and the .json metadata file.
"""
import argparse
import sys
from datetime import datetime

from ultralytics import YOLO

from research.paths import ANNOTATED_DIR, PROCESSED_DIR, TRAINING_RUNS_DIR
from research.training.dataset_builder import DEFAULT_GAP_S, DEFAULT_SPLIT_SEED, DEFAULT_VAL_FRACTION, build_dataset
from research.training.lane_processor import process_records
from research.training.metadata import filter_records, load_frame_records, load_video_catalog, parse_filter_args
from research.training.model_metadata import git_info, metrics_from, update_metadata
from research.training.onnx_export import export_onnx
from research.training.roboflow_client import cached_source, download_dataset
from research.training.trainer import train
from swimmertracker.utils.paths import PROJECT_ROOT

STEPS = 7


def step(n: int, text: str):
    print(f"\n[{n}/{STEPS}] {text}")


def describe_counts(counts: dict) -> str:
    return ", ".join(f"{key} {value}" for key, value in counts.items()) or "none"


def main():
    parser = argparse.ArgumentParser(
        prog="train-model",
        description="Pipeline 2. Download the annotated frames, warp the lanes, split the data, train and evaluate "
                    "the model and export it to ONNX for the app.",
    )
    parser.add_argument("--name", required=True,
                        help="Model name suffix, the output is models/swimmer_<name>.pt, .onnx and .json")
    parser.add_argument("--filter", nargs="*", default=[],
                        help="Frame filters as key=value pairs that must all match. The keys are video_id, stroke "
                             "and gender, for example stroke=freestyle gender=male")
    parser.add_argument("--frames", help="Comma separated glob patterns of the frame file names to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs, default %(default)s")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size in pixels, default %(default)s")
    parser.add_argument("--batch", type=int, default=16, help="Batch size, default %(default)s")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed of ultralytics, default %(default)s. Vary it to measure the spread")
    parser.add_argument("--rect", action=argparse.BooleanOptionalAction, default=True,
                        help="Rectangular batches without mosaic augmentation, the default. "
                             "--no-rect trains on square images with mosaic")
    parser.add_argument("--base-weights", default="yolo11n.pt",
                        help="Starting weights, default %(default)s. A bare file name means the copy in "
                             "research/training/cache/weights, ultralytics downloads it there if needed")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
                        help="Share of the train library images of each stroke that goes to val, default %(default)s")
    parser.add_argument("--split-gap-s", type=float, default=DEFAULT_GAP_S,
                        help="Frames of a video closer than this many seconds stay on the same side of the split, "
                             "default %(default)s")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED,
                        help="Seed of the split, default %(default)s")
    parser.add_argument("--force-download", action="store_true",
                        help="Download the annotations again even if the cache is up to date")
    parser.add_argument("--skip-download", action="store_true", help="Use the cached annotations as they are")
    parser.add_argument("--skip-train", action="store_true", help="Stop after the dataset split without training")
    args = parser.parse_args()

    filters = parse_filter_args(args.filter)
    frame_patterns = None
    if args.frames:
        frame_patterns = [p.strip() for p in args.frames.split(",") if p.strip()]

    model_name = f"swimmer_{args.name}"
    print("=" * 60)
    print(f"Training model {model_name}")
    print(f"Filters        {' '.join(f'{key}={value}' for key, value in filters.items()) or 'none'}")
    if frame_patterns:
        print(f"Frame patterns {', '.join(frame_patterns)}")
    print("=" * 60)

    if not args.skip_download:
        step(1, "Downloading the annotated frames from Roboflow...")
        source = download_dataset(target_dir=ANNOTATED_DIR, force=args.force_download)
    else:
        step(1, "Skipping the download, using the cached annotations.")
        source = cached_source(ANNOTATED_DIR)

    step(2, "Filtering frames...")
    video_catalog = load_video_catalog()
    all_records = load_frame_records(ANNOTATED_DIR, video_catalog)
    selected = filter_records(all_records, filters, frame_patterns)
    print(f"{len(selected)} of {len(all_records)} frames selected.")
    if not selected:
        print("No frames match the filter. Aborting.")
        sys.exit(1)

    step(3, "Warping the lanes and transforming the labels...")
    process_summary = process_records(selected, video_catalog)
    print(
        f"{process_summary['processed']} lane images, "
        f"{process_summary['missing_image']} missing images, "
        f"{process_summary['missing_label']} missing labels, "
        f"{process_summary['no_labels_in_lane']} lanes without labels, "
        f"{process_summary['missing_lane']} lanes missing from videos.json"
    )
    if process_summary["processed"] == 0:
        print("No lane images were produced. Aborting.")
        sys.exit(1)

    step(4, "Splitting the dataset into train, val and test...")
    split = build_dataset(process_summary["items"], video_catalog, source_dir=PROCESSED_DIR,
                          val_fraction=args.val_fraction, gap_s=args.split_gap_s, seed=args.split_seed)
    for name in ("train", "val", "test"):
        s = split[name]
        print(f"  {name:5} {s['lane_images']:4} lane images from {s['frames']:3} frames, "
              f"{describe_counts(s['by_stroke'])}")
    print(f"  {split['groups']['train']} time groups in train, {split['groups']['val']} in val")
    if not split["has_test"]:
        print("  There are no annotated frames from test library videos yet, so the model will not be tested.")
    print(f"Dataset YAML written to {split['yaml_path']}")

    if args.skip_train:
        step(5, "Stopping before the training as --skip-train was given.")
        return

    step(5, "Training YOLO...")
    model_path = train(
        model_name=args.name,
        dataset_yaml=split["yaml_path"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        rect=args.rect,
        seed=args.seed,
        base_weights=args.base_weights,
    )
    print(f"\nModel saved to {model_path}")

    step(6, "Evaluating on val and test...")
    final_model = YOLO(model_path)
    metrics = {"val": metrics_from(
        final_model.val(split="val", project=str(TRAINING_RUNS_DIR), name=f"{args.name}_val", exist_ok=True),
        split["val"]["lane_images"],
    ), "test": None}
    print(f"Val mAP50-95  {metrics['val']['map50_95']:.4f}")
    if split["has_test"]:
        metrics["test"] = metrics_from(
            final_model.val(split="test", project=str(TRAINING_RUNS_DIR), name=f"{args.name}_test", exist_ok=True),
            split["test"]["lane_images"],
        )
        print(f"Test mAP50-95 {metrics['test']['map50_95']:.4f}")
    else:
        print("Test evaluation skipped, there are no test frames.")

    update_metadata(
        model_path,
        training={
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "git": git_info(),
            "roboflow": source,
            "filters": filters,
            "frame_patterns": frame_patterns,
            "split": {k: split[k] for k in ("method", "val_fraction", "gap_s", "seed", "groups", "train", "val", "test")},
            "args": {"base_weights": args.base_weights, "epochs": args.epochs, "imgsz": args.imgsz,
                     "batch": args.batch, "rect": args.rect, "seed": args.seed},
            "run_dir": (TRAINING_RUNS_DIR / args.name).relative_to(PROJECT_ROOT).as_posix(),
        },
        metrics=metrics,
    )

    step(7, "Exporting to ONNX for the app...")
    onnx_path = export_onnx(model_path)
    metadata_name = model_path.with_suffix(".json").name
    print(f"Done. {model_path.name}, {onnx_path.name} and {metadata_name} are in {model_path.parent}")


if __name__ == "__main__":
    main()
