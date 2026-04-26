import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.frame_extractor import extract_frames
from src.training.metadata import parse_filter_args
from src.utils.paths import RAW_FRAMES_DIR


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline 1: extract raw training frames from videos based on events.json. "
            "Clears raw_frames/ and regenerates per the current event config; entries "
            "already in event_frames.json are skipped (assumed to be on Roboflow)."
        ),
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        default=[],
        help="Event filters as key=value (e.g. video_id=d2du stroke=freestyle gender=male)",
    )
    parser.add_argument(
        "--frame-count",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help=(
            "Activate frame_count limit per event. Without value: use extraction_config.json. "
            "With value: override."
        ),
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=None,
        help="Override the default sampling interval (seconds).",
    )
    args = parser.parse_args()

    criteria = parse_filter_args(args.filter)
    use_frame_count = args.frame_count is not None
    frame_count = args.frame_count if (use_frame_count and args.frame_count != -1) else None

    summary = extract_frames(
        filter_criteria=criteria or None,
        frame_count=frame_count,
        use_frame_count=use_frame_count,
        interval_sec_override=args.interval_sec,
    )

    print()
    print(f"Extracted: {summary['extracted']} new frame(s) -> {RAW_FRAMES_DIR}")
    print(f"Skipped:   {summary['skipped']} (already in event_frames.json)")
    if summary["missing_video"]:
        print(f"Missing video files: {summary['missing_video']}")

    if summary["extracted"] > 0:
        print()
        print(f"Next step: upload the new frames in {RAW_FRAMES_DIR} to Roboflow,")
        print("annotate them, then run:")
        print("    python scripts/train_model.py --name <model_name> [--filter stroke=... gender=...]")


if __name__ == "__main__":
    main()
