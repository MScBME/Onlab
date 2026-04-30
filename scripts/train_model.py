import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.dataset_builder import build_dataset
from src.training.lane_processor import process_records
from src.training.metadata import (
    filter_records,
    load_frame_records,
    load_video_catalog,
    parse_filter_args,
)
from src.training.roboflow_client import download_dataset
from src.training.trainer import train
from src.utils.paths import ANNOTATED_DIR, PROCESSED_DIR, TRAINING_RUNS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline 2: download annotated dataset, process lanes, split, and train YOLO.",
    )
    parser.add_argument("--name", required=True, help="Model name suffix (output: models/swimmer_<name>.pt)")
    parser.add_argument(
        "--filter",
        nargs="*",
        default=[],
        help="Category filters as key=value (e.g. stroke=freestyle gender=male)",
    )
    parser.add_argument(
        "--frames",
        help="Comma-separated filename patterns to include (glob, e.g. 'd2du_*.jpg,custom_*.jpg')",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--force-download", action="store_true", help="Re-download even if annotated/ exists")
    parser.add_argument("--skip-download", action="store_true", help="Skip Roboflow download stage")
    parser.add_argument("--skip-train", action="store_true", help="Stop after dataset split (no training)")
    args = parser.parse_args()

    filters = parse_filter_args(args.filter)
    frame_patterns = None
    if args.frames:
        frame_patterns = [p.strip() for p in args.frames.split(",") if p.strip()]

    print("=" * 60)
    print(f"Training model: swimmer_{args.name}")
    print(f"Filters: {filters or '(none)'}")
    if frame_patterns:
        print(f"Frame patterns: {frame_patterns}")
    print("=" * 60)

    if not args.skip_download:
        print("\n[1/5] Downloading annotated dataset from Roboflow...")
        download_dataset(target_dir=ANNOTATED_DIR, force=args.force_download)
    else:
        print("\n[1/5] Skipping download stage.")

    print("\n[2/5] Filtering frames...")
    video_catalog = load_video_catalog()
    all_records = load_frame_records(ANNOTATED_DIR)
    selected = filter_records(all_records, filters, frame_patterns)
    print(f"Selected {len(selected)} / {len(all_records)} frame(s) after filtering.")
    if not selected:
        print("No frames match the filter. Aborting.")
        sys.exit(1)

    print("\n[3/5] Processing lanes (perspective warp + label transform)...")
    process_summary = process_records(selected, video_catalog)
    print(
        f"Processed: {process_summary['processed']} per-lane images "
        f"(missing images: {process_summary['missing_image']}, "
        f"missing labels: {process_summary['missing_label']}, "
        f"lanes without labels: {process_summary['no_labels_in_lane']})"
    )
    if process_summary["processed"] == 0:
        print("No per-lane images produced. Aborting.")
        sys.exit(1)

    print("\n[4/5] Splitting dataset (80/10/10)...")
    split_summary = build_dataset(source_dir=PROCESSED_DIR)
    print(
        f"Total: {split_summary['total']} | "
        f"train: {split_summary['train']}, val: {split_summary['val']}, test: {split_summary['test']}"
    )
    print(f"Dataset YAML: {split_summary['yaml_path']}")

    if args.skip_train:
        print("\n[5/5] Skipping training stage.")
        return

    print("\n[5/5] Training YOLO...")
    model_path = train(
        model_name=args.name,
        dataset_yaml=split_summary["yaml_path"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    print(f"\nDone. Model saved to: {model_path}")

    print("\n[6/6] Final Evaluation on Test Set...")
    final_model = YOLO(model_path)
    results = final_model.val(
        split="test",
        project=str(TRAINING_RUNS_DIR),
        name=f"{args.name}_test",
        exist_ok=True,
    )
    print(f"Test Set mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")


if __name__ == "__main__":
    main()
