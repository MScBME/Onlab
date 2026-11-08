"""Shared by the evaluation commands. Reads the defaults of run_config.json and finds a clip in the test or train library."""
import sys
from dataclasses import dataclass
from pathlib import Path

from research.cli import load_json
from research.paths import RUN_CONFIG_JSON, TEST_LIBRARY, TRAIN_LIBRARY
from swimmertracker.catalog.library import VideoLibrary
from swimmertracker.catalog.models import Clip, Lane, Video
from swimmertracker.utils.paths import PROJECT_ROOT

LIBRARIES = (("test", TEST_LIBRARY), ("train", TRAIN_LIBRARY))


@dataclass
class ClipContext:
    clip: Clip
    video: Video
    lane: Lane
    role: str  # the library of the clip's video, "test" or "train"
    video_path: Path


def load_run_config() -> dict:
    return load_json(RUN_CONFIG_JSON)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_clip(clip_id: str, lane_id: int = None, libraries=LIBRARIES) -> ClipContext:
    for role, paths in libraries:
        library = VideoLibrary.open(paths)
        clip = next((c for c in library.clips.list() if c.id == clip_id), None)
        if clip is None:
            continue
        video = library.videos.get(clip.video_id)
        lane = video.lane(lane_id or clip.lane)
        if role == "train":
            print(f"[NOTE] Clip '{clip_id}' is on the training video {video.id}, the model has seen this recording")
        return ClipContext(clip, video, lane, role, library.videos.file_path(video))
    names = " nor ".join(str(p.clips_json.relative_to(PROJECT_ROOT)) for _, p in libraries)
    print(f"Clip id '{clip_id}' is in neither {names}")
    sys.exit(1)


def clip_from_args(args, config: dict) -> ClipContext:
    clip_id = args.clip or config.get("active_clip_id")
    if not clip_id:
        print("No clip id was given. Use --clip or set active_clip_id in research/run_config.json")
        sys.exit(1)
    return find_clip(clip_id, getattr(args, "lane", None))
