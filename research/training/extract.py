"""Pipeline 1, the train-extract command. Extracts raw frames of the events in events.json to annotate on Roboflow."""
import argparse

from research.paths import RAW_FRAMES_DIR
from research.training.frame_extractor import extract_frames
from research.training.metadata import parse_filter_args


def main():
    parser = argparse.ArgumentParser(
        prog="train-extract",
        description=(
            "Pipeline 1. Extract raw training frames from the videos of the events in events.json. "
            "Clears raw_frames and fills it again from the current events. Frames already in "
            "event_frames.json are skipped, they are assumed to be on Roboflow."
        ),
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        default=[],
        help="Event filters as key=value pairs that must all match. A key is an event field, lanes=<n> selects "
             "the events with that lane, for example video_id=d2du stroke=freestyle",
    )
    parser.add_argument(
        "--event-ids",
        help="Comma separated positions of the events in events.json, counted from 1, for example 1,5,7",
    )
    parser.add_argument(
        "--frame-count",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help="Extract exactly this many frames per event. Without a value the frame_count of "
             "extraction_config.json is used",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=None,
        help="Sampling interval in seconds instead of the default_interval_sec of extraction_config.json",
    )
    args = parser.parse_args()

    criteria = parse_filter_args(args.filter)
    use_frame_count = args.frame_count is not None
    frame_count = args.frame_count if (use_frame_count and args.frame_count != -1) else None
    event_ids = [int(s) for s in args.event_ids.split(",") if s.strip()] if args.event_ids else None

    summary = extract_frames(
        filter_criteria=criteria or None,
        event_ids=event_ids,
        frame_count=frame_count,
        use_frame_count=use_frame_count,
        interval_sec_override=args.interval_sec,
    )

    print()
    print(f"Extracted {summary['extracted']} new frames into {RAW_FRAMES_DIR}")
    for role, count in sorted(summary["extracted_by_role"].items()):
        usage = "train and val only" if role == "train" else "test only"
        print(f"  {count} from {role} library videos, for {usage}")
    print(f"Skipped {summary['skipped']} frames that are already in event_frames.json")
    if summary["missing_video"]:
        print(f"Missing video files of {', '.join(summary['missing_video'])}")

    if summary["extracted"] > 0:
        print()
        print(f"Next, upload the new frames in {RAW_FRAMES_DIR} to Roboflow, annotate them and run")
        print("    uv run train-model --name <name> [--filter <key>=<value> ...]")


if __name__ == "__main__":
    main()
