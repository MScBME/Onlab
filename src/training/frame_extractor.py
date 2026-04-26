import shutil

import cv2

from src.training.metadata import (
    load_event_catalog,
    load_extraction_config,
    load_event_frames,
    load_video_catalog,
)
from src.utils.config import time_to_seconds
from src.utils.paths import EVENT_FRAMES_JSON, RAW_FRAMES_DIR, RAW_VIDEO_DIR


def seconds_to_hms_ms(total_seconds: float) -> tuple:
    total_ms = int(round(total_seconds * 1000))
    hh = total_ms // 3_600_000
    mm = (total_ms % 3_600_000) // 60_000
    ss = (total_ms % 60_000) // 1000
    ms = total_ms % 1000
    return hh, mm, ss, ms


def frame_id(video_id: str, total_seconds: float) -> str:
    hh, mm, ss, ms = seconds_to_hms_ms(total_seconds)
    return f"{video_id}_{hh:02d}_{mm:02d}_{ss:02d}_{ms:03d}"


def generate_timestamps(event: dict, interval_sec: float, frame_count: int = None) -> list:
    start = time_to_seconds(event["start_time"])
    end = time_to_seconds(event["end_time"])
    if end <= start:
        raise ValueError(
            f"Event {event['video_id']}@{event['start_time']}: end_time must be after start_time"
        )

    if frame_count is not None:
        timestamps = [start + i * interval_sec for i in range(frame_count)]
        if timestamps[-1] > end:
            raise ValueError(
                f"Event {event['video_id']}@{event['start_time']}: "
                f"last frame at +{(frame_count - 1) * interval_sec}s exceeds end_time"
            )
        return timestamps

    timestamps = []
    ts = start
    while ts <= end + 1e-9:
        timestamps.append(ts)
        ts += interval_sec
    return timestamps


def _resolve_extraction_params(config, frame_count, use_frame_count, interval_sec_override):
    interval_sec = interval_sec_override if interval_sec_override is not None else config.get("default_interval_sec", 2)
    if use_frame_count and frame_count is None:
        frame_count = config.get("frame_count")
        if frame_count is None:
            raise ValueError("--frame-count flag set but no frame_count in extraction_config.json or CLI")
    return interval_sec, frame_count


def _group_by_video(events: list) -> dict:
    grouped = {}
    for event in events:
        grouped.setdefault(event["video_id"], []).append(event)
    return grouped


def _extract_event_frames(cap, fps, event, video_id, interval_sec, frame_count, existing_by_id):
    extracted = 0
    skipped = 0
    new_entries = []
    timestamps = generate_timestamps(event, interval_sec, frame_count)
    for ts in timestamps:
        fid = frame_id(video_id, ts)
        if fid in existing_by_id:
            skipped += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * fps)))
        success, frame = cap.read()
        if not success:
            print(f"[WARN] Could not read frame at {ts:.3f}s for {fid}")
            continue

        cv2.imwrite(str(RAW_FRAMES_DIR / f"{fid}.jpg"), frame)

        entry = {
            "id": fid,
            "lanes": event["lanes"],
            "stroke": event["stroke"],
            "gender": event["gender"],
        }
        new_entries.append(entry)
        existing_by_id[fid] = entry
        extracted += 1
    return extracted, skipped, new_entries


def _process_video(video_path, vid_events, video_id, interval_sec, frame_count, existing_by_id):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)

    extracted = 0
    skipped = 0
    new_entries = []
    for event in vid_events:
        e, s, n = _extract_event_frames(cap, fps, event, video_id, interval_sec, frame_count, existing_by_id)
        extracted += e
        skipped += s
        new_entries.extend(n)

    cap.release()
    return extracted, skipped, new_entries


def extract_frames(
    filter_criteria: dict = None,
    frame_count: int = None,
    use_frame_count: bool = False,
    interval_sec_override: float = None,
) -> dict:
    video_catalog = load_video_catalog()
    event_catalog = load_event_catalog()
    config = load_extraction_config()

    interval_sec, frame_count_resolved = _resolve_extraction_params(
        config, frame_count, use_frame_count, interval_sec_override
    )
    effective_frame_count = frame_count_resolved if use_frame_count else None

    events = event_catalog.filter(filter_criteria) if filter_criteria else event_catalog.all()
    if not events:
        print("No events matched the filter; nothing to extract.")
        return {"extracted": 0, "skipped": 0, "missing_video": []}

    existing_by_id = {e["id"]: e for e in load_event_frames()}

    if RAW_FRAMES_DIR.exists():
        shutil.rmtree(RAW_FRAMES_DIR)
    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"extracted": 0, "skipped": 0, "missing_video": []}
    any_new = False

    for video_id, vid_events in _group_by_video(events).items():
        video = video_catalog.get_video(video_id)
        video_path = RAW_VIDEO_DIR / video["path"]
        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            summary["missing_video"].append(video_id)
            continue

        result = _process_video(video_path, vid_events, video_id, interval_sec, effective_frame_count, existing_by_id)
        if result is None:
            summary["missing_video"].append(video_id)
            continue
        e, s, new_entries = result
        summary["extracted"] += e
        summary["skipped"] += s
        if new_entries:
            any_new = True

    if any_new:
        all_entries = sorted(existing_by_id.values(), key=lambda e: e["id"])
        _save_event_frames(all_entries)

    return summary


def _save_event_frames(entries: list):
    lines = ['{', '  "event_frames": [']
    for i, e in enumerate(entries):
        lanes_str = "[" + ", ".join(str(x) for x in e["lanes"]) + "]"
        line = (
            f'    {{"id": "{e["id"]}", "lanes": {lanes_str}, '
            f'"stroke": "{e["stroke"]}", "gender": "{e["gender"]}"}}'
        )
        if i < len(entries) - 1:
            line += ","
        lines.append(line)
    lines.append("  ]")
    lines.append("}")
    EVENT_FRAMES_JSON.write_text("\n".join(lines) + "\n", encoding="utf-8")
