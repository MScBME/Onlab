import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from research.cli import load_json
from research.paths import (
    CUSTOM_FRAMES_JSON,
    EVENTS_JSON,
    EVENT_FRAMES_JSON,
    EXTRACTION_CONFIG_JSON,
    TEST_LIBRARY,
    TRAIN_LIBRARY,
)
from swimmertracker.catalog.models import Video
from swimmertracker.catalog.repository import VideoRepository
from swimmertracker.utils.paths import LibraryPaths

# A video of the train library gives train and val frames, a video of the test library gives test frames, never both.
TRAIN, TEST = "train", "test"

EVENT_REQUIRED_FIELDS = ("video_id", "start_time", "end_time", "stroke", "gender", "lanes")
EVENT_FRAME_REQUIRED = ("id", "lanes", "stroke", "gender")
CUSTOM_FRAME_REQUIRED = ("id", "video_id", "lanes", "stroke", "gender")

EVENT_FRAME_ID_PATTERN = re.compile(r"^(\w+)_(\d{2})_(\d{2})_(\d{2})_(\d{3})$")
CUSTOM_ID_PATTERN = re.compile(r"^custom_\d+$")


@dataclass(frozen=True)
class VideoSource:
    video: Video
    role: str  # TRAIN or TEST
    library: LibraryPaths

    @property
    def file_path(self) -> Path:
        return self.library.videos_dir / self.video.path


@dataclass
class VideoCatalog:
    """The videos of the train and the test library, each tagged with its role."""
    sources: dict  # video id -> VideoSource

    def get_video(self, video_id: str) -> VideoSource:
        if video_id not in self.sources:
            raise KeyError(f"Video '{video_id}' is in neither the train nor the test library")
        return self.sources[video_id]

    def role(self, video_id: str) -> str:
        return self.get_video(video_id).role

    def file_path(self, video_id: str) -> Path:
        return self.get_video(video_id).file_path

    def get_lane_coordinates(self, video_id: str, lane_id: int) -> list:
        return self.get_video(video_id).video.lane(lane_id).coordinates

    def get_lane_length(self, video_id: str, lane_id: int) -> float:
        """Real length of the marked lane section in meters. Lanes without length_m default to 25 m."""
        return self.get_video(video_id).video.lane(lane_id).length_m


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
    role: str = TRAIN
    is_custom: bool = False
    note: Optional[str] = None
    timestamp_s: Optional[float] = None  # position in the video in seconds, None for custom frames
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


def load_video_catalog(train: LibraryPaths = TRAIN_LIBRARY, test: LibraryPaths = TEST_LIBRARY) -> VideoCatalog:
    sources = {}
    files = {}
    for role, library in ((TRAIN, train), (TEST, test)):
        for video in VideoRepository(library.videos_json, library.videos_dir).list():
            if video.id in sources:
                raise ValueError(f"Video id '{video.id}' is in both the train and the test library, "
                                 "but a video can only be in one of them")
            source = VideoSource(video, role, library)
            sources[video.id] = source
            files.setdefault(Path(video.path).name.lower(), set()).add(role)
    shared = sorted(name for name, roles in files.items() if len(roles) > 1)
    if shared:
        raise ValueError(f"The video files {', '.join(shared)} are referenced from both the train and the test library")
    return VideoCatalog(sources=sources)


def load_event_catalog() -> EventCatalog:
    data = load_json(EVENTS_JSON)
    events = data.get("events", [])
    for i, event in enumerate(events):
        missing = [f for f in EVENT_REQUIRED_FIELDS if f not in event]
        if missing:
            raise ValueError(
                f"The event at index {i}, video {event.get('video_id')} at {event.get('start_time')}, "
                f"is missing the fields {', '.join(missing)}"
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
        raise ValueError(f"Invalid event frame id '{frame_id}'")
    video_id, hh, mm, ss, ms = match.groups()
    return video_id, int(hh), int(mm), int(ss), int(ms)


def event_frame_timestamp(frame_id: str) -> float:
    _, hh, mm, ss, ms = parse_event_frame_id(frame_id)
    return hh * 3600 + mm * 60 + ss + ms / 1000


def is_custom_id(frame_id: str) -> bool:
    return bool(CUSTOM_ID_PATTERN.match(frame_id))


def load_frame_records(annotated_dir: Path, catalog: VideoCatalog) -> list:
    """Annotated images matched to their registry entry. The role comes from the library of the video.

    Frames whose video is in neither library, for example because the video was removed, are skipped with a warning.
    """
    generated_by_id = {e["id"]: e for e in load_event_frames()}
    custom_by_id = {e["id"]: e for e in load_custom_frames()}

    records = []
    for img_path in sorted(annotated_dir.glob("*.jpg")):
        stem = img_path.stem
        if stem in generated_by_id:
            entry = generated_by_id[stem]
            video_id = parse_event_frame_id(stem)[0]
            record = FrameRecord(
                id=stem,
                video_id=video_id,
                lanes=entry["lanes"],
                stroke=entry["stroke"],
                gender=entry["gender"],
                timestamp_s=event_frame_timestamp(stem),
                annotated_path=img_path,
            )
        elif stem in custom_by_id:
            entry = custom_by_id[stem]
            record = FrameRecord(
                id=stem,
                video_id=entry["video_id"],
                lanes=entry["lanes"],
                stroke=entry["stroke"],
                gender=entry["gender"],
                is_custom=True,
                note=entry.get("note"),
                annotated_path=img_path,
            )
        else:
            raise ValueError(
                f"Annotated image '{img_path.name}' has no matching entry in "
                f"event_frames.json or custom_frames.json"
            )
        if record.video_id not in catalog.sources:
            print(f"[WARN] Skipped {img_path.name}, its video '{record.video_id}' is in neither library")
            continue
        record.role = catalog.role(record.video_id)
        records.append(record)
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
