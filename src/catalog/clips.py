"""Creating inference clips (data/clips.json) from a viewed video range."""
from src.catalog.models import Clip
from src.catalog.repository import ClipRepository
from src.utils.timefmt import format_time

SAME_TIME_TOLERANCE_S = 0.0005


def suggest_clip_id(video_id: str, lane_id: int, start_sec: float, existing_ids) -> str:
    hh, mm, ss = format_time(start_sec).split(".")[0].split(":")
    base = f"{video_id}_lane{lane_id}_{hh}_{mm}_{ss}"
    candidate, n = base, 2
    while candidate in existing_ids:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def find_duplicate(clips: list, video_id: str, lane_id: int, start_sec: float, end_sec: float):
    for clip in clips:
        if (clip.video_id == video_id and clip.lane == lane_id
                and abs(clip.start_sec - start_sec) < SAME_TIME_TOLERANCE_S
                and abs(clip.end_sec - end_sec) < SAME_TIME_TOLERANCE_S):
            return clip
    return None


def propose_clip(repo: ClipRepository, video_id: str, lane_id: int, start_sec: float, end_sec: float) -> tuple:
    """Returns (clip_without_description, duplicate_or_None)."""
    existing = repo.list()
    clip = Clip(
        id=suggest_clip_id(video_id, lane_id, start_sec, {c.id for c in existing}),
        video_id=video_id,
        lane=lane_id,
        start_time=format_time(start_sec),
        end_time=format_time(end_sec),
    )
    return clip, find_duplicate(existing, video_id, lane_id, start_sec, end_sec)


def create_clip(repo: ClipRepository, video_id: str, lane_id: int, start_sec: float, end_sec: float,
                description: str = "") -> Clip:
    if end_sec <= start_sec:
        raise ValueError("The end of the range must be after its start.")
    clip, duplicate = propose_clip(repo, video_id, lane_id, start_sec, end_sec)
    if duplicate is not None:
        raise ValueError(f"This range is already saved as clip '{duplicate.id}'.")
    clip.description = description.strip() or None
    repo.add(clip)
    return clip
