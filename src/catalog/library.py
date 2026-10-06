"""Video library use cases: importing videos into data/raw and editing their lanes."""
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.catalog.models import Lane, Video
from src.catalog.repository import ClipRepository, VideoRepository
from src.training.metadata import parse_event_frame_id
from src.utils.config import read_json
from src.utils.paths import CUSTOM_FRAMES_JSON, EVENT_FRAMES_JSON, EVENTS_JSON
from src.video.lane_warp import normalize_corners, validate_corners

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
MAX_LANE_ID = 20
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class CopyCancelled(Exception):
    pass


@dataclass
class LaneReferences:
    clips: list = field(default_factory=list)  # clip ids
    events: int = 0
    event_frames: int = 0
    custom_frames: int = 0

    @property
    def any(self) -> bool:
        return bool(self.clips or self.events or self.event_frames or self.custom_frames)

    def describe(self) -> str:
        parts = []
        if self.clips:
            parts.append(f"{len(self.clips)} clip{'s' if len(self.clips) != 1 else ''}")
        if self.events:
            parts.append(f"{self.events} training event{'s' if self.events != 1 else ''}")
        training_frames = self.event_frames + self.custom_frames
        if training_frames:
            parts.append(f"{training_frames} training frame{'s' if training_frames != 1 else ''}")
        return ", ".join(parts)


def suggest_video_id(filename: str, existing_ids) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_") or "video"
    candidate, n = base, 2
    while candidate in existing_ids:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def validate_video_id(video_id: str, existing_ids) -> Optional[str]:
    if not video_id:
        return "Video ID is required."
    if not VIDEO_ID_PATTERN.match(video_id):
        return "Video ID may only contain letters, digits and underscores."
    if video_id in existing_ids:
        return f"Video ID '{video_id}' already exists."
    return None


def is_inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def import_target(source: Path, raw_dir: Path) -> Path:
    """Where an imported file will live: in place if already under data/raw, else a free name there."""
    if is_inside(source, raw_dir):
        return source
    candidate, n = raw_dir / source.name, 2
    while candidate.exists():
        candidate, n = raw_dir / f"{source.stem} ({n}){source.suffix}", n + 1
    return candidate


def copy_with_progress(src: Path, dst: Path, on_progress: Callable[[float], None] = None,
                       cancel: threading.Event = None):
    """Copy via a .part file; on cancel or error the partial file is removed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    total = max(1, src.stat().st_size)
    done = 0
    try:
        with open(src, "rb") as fin, open(part, "wb") as fout:
            while True:
                if cancel is not None and cancel.is_set():
                    raise CopyCancelled()
                chunk = fin.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                fout.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done / total)
        part.replace(dst)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def lane_references(video_id: str, lane_id: int, clips: ClipRepository = None) -> LaneReferences:
    """Everything in the metadata that points at (video_id, lane_id)."""
    clips = clips or ClipRepository()
    refs = LaneReferences()
    refs.clips = [c.id for c in clips.list() if c.video_id == video_id and c.lane == lane_id]

    if EVENTS_JSON.exists():
        refs.events = sum(
            1 for e in read_json(EVENTS_JSON).get("events", [])
            if e.get("video_id") == video_id and lane_id in e.get("lanes", [])
        )
    if EVENT_FRAMES_JSON.exists():
        for entry in read_json(EVENT_FRAMES_JSON).get("event_frames", []):
            try:
                frame_video_id = parse_event_frame_id(entry["id"])[0]
            except ValueError:
                continue
            if frame_video_id == video_id and lane_id in entry.get("lanes", []):
                refs.event_frames += 1
    if CUSTOM_FRAMES_JSON.exists():
        refs.custom_frames = sum(
            1 for e in read_json(CUSTOM_FRAMES_JSON).get("custom_frames", [])
            if e.get("video_id") == video_id and lane_id in e.get("lanes", [])
        )
    return refs


def prepare_lanes(lanes: list, frame_size=None) -> tuple:
    """Normalize and validate an edited lane list. Returns (normalized_lanes, errors)."""
    errors = []
    if not lanes:
        errors.append("A video needs at least one lane.")
    ids = [lane.id for lane in lanes]
    if len(ids) != len(set(ids)):
        errors.append("Lane IDs must be unique.")

    normalized = []
    for lane in lanes:
        if not 1 <= lane.id <= MAX_LANE_ID:
            errors.append(f"Lane ID must be between 1 and {MAX_LANE_ID} (got {lane.id}).")
        if not lane.length_m > 0:
            errors.append(f"Lane {lane.id}: length must be a positive number of meters.")
        coords = [[round(float(x)), round(float(y))] for x, y in lane.coordinates]
        try:
            coords = normalize_corners(coords)
        except ValueError as e:
            errors.append(f"Lane {lane.id}: {e}.")
            continue
        errors.extend(f"Lane {lane.id}: {msg}" for msg in validate_corners(coords, frame_size))
        normalized.append(Lane(id=lane.id, coordinates=coords, length_m=float(lane.length_m)))
    return normalized, errors


class VideoLibrary:
    def __init__(self, videos: VideoRepository = None, clips: ClipRepository = None):
        self.videos = videos or VideoRepository()
        self.clips = clips or ClipRepository()

    def lane_references(self, video_id: str, lane_id: int) -> LaneReferences:
        return lane_references(video_id, lane_id, self.clips)

    def add_video(self, video_id: str, file_path: Path, lanes: list, frame_size=None) -> Video:
        error = validate_video_id(video_id, self.videos.ids())
        if error:
            raise ValueError(error)
        if not is_inside(file_path, self.videos.raw_dir):
            raise ValueError(f"Video file must be inside {self.videos.raw_dir}")
        normalized, errors = prepare_lanes(lanes, frame_size)
        if errors:
            raise ValueError("\n".join(errors))
        rel_path = file_path.resolve().relative_to(self.videos.raw_dir.resolve()).as_posix()
        video = Video(id=video_id, path=rel_path, lanes=normalized)
        self.videos.save(video)
        return video

    def save_lanes(self, video_id: str, lanes: list, frame_size=None) -> Video:
        video = self.videos.get(video_id)
        normalized, errors = prepare_lanes(lanes, frame_size)
        new_ids = {lane.id for lane in normalized}
        for old in video.lanes:
            if old.id not in new_ids:
                refs = self.lane_references(video_id, old.id)
                if refs.any:
                    errors.append(f"Lane {old.id} cannot be removed: it is used by {refs.describe()}.")
        if errors:
            raise ValueError("\n".join(errors))
        video.lanes = normalized
        self.videos.save(video)
        return video
