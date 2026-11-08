"""Video library use cases, importing videos into the library's videos/ folder and editing their lanes."""
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from swimmertracker.catalog.models import Lane, Video
from swimmertracker.catalog.repository import ClipRepository, VideoRepository
from swimmertracker.utils.paths import LibraryPaths
from swimmertracker.video.lane_warp import normalize_corners, validate_corners

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
VIDEO_ID_PATTERN = re.compile(r"\w+", re.ASCII)  # ASCII only, the id ends up in frame and clip file names
MAX_LANE_ID = 20
COPY_CHUNK_BYTES = 8 * 1024 * 1024


class CopyCancelled(Exception):
    pass


@dataclass
class LaneReferences:
    clips: list = field(default_factory=list)  # clip ids

    @property
    def any(self) -> bool:
        return bool(self.clips)

    def describe(self) -> str:
        count = len(self.clips)
        if count == 0:
            return ""
        return "1 clip" if count == 1 else f"{count} clips"


def suggest_video_id(filename: str, existing_ids) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_") or "video"
    candidate, n = base, 2
    while candidate in existing_ids:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def validate_video_id(video_id: str, existing_ids) -> Optional[str]:
    if not video_id:
        return "Video ID is required."
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
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


def import_target(source: Path, videos_dir: Path) -> Path:
    """Where an imported file will live. A file already under videos/ stays in place, any other gets a free name there."""
    if is_inside(source, videos_dir):
        return source
    candidate, n = videos_dir / source.name, 2
    while candidate.exists():
        candidate, n = videos_dir / f"{source.stem} ({n}){source.suffix}", n + 1
    return candidate


def copy_with_progress(src: Path, dst: Path, on_progress: Callable[[float], None] = None,
                       cancel: threading.Event = None):
    """Copy through a .part file. On cancel or error the partial file is removed."""
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


def lane_references(video_id: str, lane_id: int, clips: ClipRepository) -> LaneReferences:
    """Everything in the library that refers to lane lane_id of video video_id."""
    return LaneReferences(clips=[c.id for c in clips.list() if c.video_id == video_id and c.lane == lane_id])


def prepare_lanes(lanes: list, frame_size=None) -> tuple:
    """Normalize and validate an edited lane list. Returns the normalized lanes and the errors."""
    errors = []
    if not lanes:
        errors.append("A video needs at least one lane.")
    ids = [lane.id for lane in lanes]
    if len(ids) != len(set(ids)):
        errors.append("Lane IDs must be unique.")

    normalized = []
    for lane in lanes:
        if not 1 <= lane.id <= MAX_LANE_ID:
            errors.append(f"Lane ID must be between 1 and {MAX_LANE_ID}, got {lane.id}.")
        if not math.isfinite(lane.length_m) or lane.length_m <= 0:  # a hand edited videos.json may hold NaN or Infinity
            errors.append(f"The length of lane {lane.id} must be a positive number of meters.")
        coords = [[round(float(x)), round(float(y))] for x, y in lane.coordinates]
        try:
            coords = normalize_corners(coords)
        except ValueError as e:
            errors.append(f"Lane {lane.id} is invalid. {e}.")
            continue
        errors.extend(f"Lane {lane.id} is invalid. {msg}" for msg in validate_corners(coords, frame_size))
        normalized.append(Lane(id=lane.id, coordinates=coords, length_m=float(lane.length_m)))
    return normalized, errors


class VideoLibrary:
    def __init__(self, videos: VideoRepository, clips: ClipRepository):
        self.videos = videos
        self.clips = clips

    @classmethod
    def open(cls, paths: LibraryPaths) -> "VideoLibrary":
        return cls(VideoRepository(paths.videos_json, paths.videos_dir), ClipRepository(paths.clips_json))

    def lane_references(self, video_id: str, lane_id: int) -> LaneReferences:
        return lane_references(video_id, lane_id, self.clips)

    def add_video(self, video_id: str, file_path: Path, lanes: list, frame_size=None) -> Video:
        error = validate_video_id(video_id, self.videos.ids())
        if error:
            raise ValueError(error)
        if not is_inside(file_path, self.videos.videos_dir):
            raise ValueError(f"Video file must be inside {self.videos.videos_dir}")
        normalized, errors = prepare_lanes(lanes, frame_size)
        if errors:
            raise ValueError("\n".join(errors))
        rel_path = file_path.resolve().relative_to(self.videos.videos_dir.resolve()).as_posix()
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
                    errors.append(f"Lane {old.id} is used by {refs.describe()} and cannot be removed.")
        if errors:
            raise ValueError("\n".join(errors))
        video.lanes = normalized
        self.videos.save(video)
        return video
