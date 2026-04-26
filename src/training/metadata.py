import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.config import load_json
from src.utils.paths import (
    CUSTOM_FRAMES_JSON,
    EVENTS_JSON,
    EXTRACTION_CONFIG_JSON,
    EVENT_FRAMES_JSON,
    VIDEOS_JSON,
)

EVENT_REQUIRED_FIELDS = ("video_id", "start_time", "end_time", "stroke", "gender", "lanes")
EVENT_FRAME_REQUIRED = ("id", "lanes", "stroke", "gender")
CUSTOM_FRAME_REQUIRED = ("id", "video_id", "lanes", "stroke", "gender")

EVENT_FRAME_ID_PATTERN = re.compile(r"^([A-Za-z0-9]+)_(\d{2})_(\d{2})_(\d{2})_(\d{3})$")
CUSTOM_ID_PATTERN = re.compile(r"^custom_\d+$")


@dataclass
class VideoCatalog:
    videos_by_id: dict

    def get_video(self, video_id: str) -> dict:
        if video_id not in self.videos_by_id:
            raise KeyError(f"Video '{video_id}' not found in videos.json")
        return self.videos_by_id[video_id]

    def get_lane_coordinates(self, video_id: str, lane_id: int) -> list:
        video = self.get_video(video_id)
        for lane in video.get("lanes", []):
            if lane["id"] == lane_id:
                return lane["coordinates"]
        raise KeyError(f"Lane {lane_id} not found for video '{video_id}'")


@dataclass
class EventCatalog:
    events: list

    def all(self) -> list:
        return list(self.events)

    def filter(self, criteria: dict, events: list = None) -> list:
        source = events if events is not None else self.events
        return [e for e in source if all(_event_matches(e, k, v) for k, v in criteria.items())]


@dataclass
class FrameRecord:
    id: str
    video_id: str
    lanes: list
    stroke: str
    gender: str
    is_custom: bool = False
    note: Optional[str] = None
    annotated_path: Optional[Path] = field(default=None, repr=False)

    @property
    def filename(self) -> str:
        return f"{self.id}.jpg"


def _event_matches(event: dict, key: str, value: str) -> bool:
    if key == "lanes":
        try:
            return int(value) in event.get("lanes", [])
        except ValueError:
            return False
    return str(event.get(key)) == value


def load_video_catalog() -> VideoCatalog:
    data = load_json(VIDEOS_JSON)
    videos_by_id = {v["id"]: v for v in data.get("videos", [])}
    return VideoCatalog(videos_by_id=videos_by_id)


def load_event_catalog() -> EventCatalog:
    data = load_json(EVENTS_JSON)
    events = data.get("events", [])
    for i, event in enumerate(events):
        missing = [f for f in EVENT_REQUIRED_FIELDS if f not in event]
        if missing:
            raise ValueError(
                f"Event at index {i} ({event.get('video_id')}@{event.get('start_time')}): "
                f"missing required fields: {missing}"
            )
    return EventCatalog(events=events)


def load_extraction_config() -> dict:
    if not EXTRACTION_CONFIG_JSON.exists():
        return {"default_interval_sec": 2}
    return load_json(EXTRACTION_CONFIG_JSON)


def load_event_frames() -> list:
    if not EVENT_FRAMES_JSON.exists():
        return []
    data = load_json(EVENT_FRAMES_JSON)
    return data.get("event_frames", [])


def load_custom_frames() -> list:
    if not CUSTOM_FRAMES_JSON.exists():
        return []
    data = load_json(CUSTOM_FRAMES_JSON)
    return data.get("custom_frames", [])


def parse_event_frame_id(frame_id: str) -> tuple:
    match = EVENT_FRAME_ID_PATTERN.match(frame_id)
    if not match:
        raise ValueError(f"Invalid event frame id: '{frame_id}'")
    video_id, hh, mm, ss, ms = match.groups()
    return video_id, int(hh), int(mm), int(ss), int(ms)


def is_custom_id(frame_id: str) -> bool:
    return bool(CUSTOM_ID_PATTERN.match(frame_id))


def load_frame_records(annotated_dir: Path) -> list:
    generated_by_id = {e["id"]: e for e in load_event_frames()}
    custom_by_id = {e["id"]: e for e in load_custom_frames()}

    records = []
    for img_path in annotated_dir.glob("*.jpg"):
        stem = img_path.stem
        if stem in generated_by_id:
            entry = generated_by_id[stem]
            video_id, *_ = parse_event_frame_id(stem)
            records.append(FrameRecord(
                id=stem,
                video_id=video_id,
                lanes=entry["lanes"],
                stroke=entry["stroke"],
                gender=entry["gender"],
                is_custom=False,
                annotated_path=img_path,
            ))
        elif stem in custom_by_id:
            entry = custom_by_id[stem]
            records.append(FrameRecord(
                id=stem,
                video_id=entry["video_id"],
                lanes=entry["lanes"],
                stroke=entry["stroke"],
                gender=entry["gender"],
                is_custom=True,
                note=entry.get("note"),
                annotated_path=img_path,
            ))
        else:
            raise ValueError(
                f"Annotated image '{img_path.name}' has no matching entry in "
                f"event_frames.json or custom_frames.json"
            )
    return records


def filter_records(records: list, criteria: dict, frame_patterns: Optional[list] = None) -> list:
    result = []
    for rec in records:
        if frame_patterns and not any(fnmatch.fnmatch(rec.filename, p) for p in frame_patterns):
            continue
        if all(getattr(rec, k, None) == v for k, v in criteria.items()):
            result.append(rec)
    return result


def parse_filter_args(filter_args: list) -> dict:
    criteria = {}
    for arg in filter_args or []:
        if "=" not in arg:
            raise ValueError(f"Filter must be key=value, got '{arg}'")
        key, value = arg.split("=", 1)
        criteria[key.strip()] = value.strip()
    return criteria
