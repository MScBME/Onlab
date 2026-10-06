"""JSON-backed repositories for data/videos.json and data/clips.json.

Every write re-reads the file first and only replaces the entry being saved, so manual edits
made while the app is running are not clobbered and untouched entries keep their exact form.
"""
import json
from pathlib import Path

from src.catalog.models import Clip, Video
from src.utils.config import read_json, save_json, write_text_atomic
from src.utils.paths import CLIPS_JSON, RAW_VIDEO_DIR, VIDEOS_JSON


def format_videos_json(data: dict) -> str:
    """Serialize videos.json with indent 2 but one line per lane, as the file is maintained by hand."""
    videos = data.get("videos", [])
    lines = ["{", '  "videos": [']
    for i, video in enumerate(videos):
        lines.append("    {")
        items = list(video.items())
        for j, (key, value) in enumerate(items):
            comma = "," if j < len(items) - 1 else ""
            if key == "lanes" and value:
                lines.append('      "lanes": [')
                for k, lane in enumerate(value):
                    lane_comma = "," if k < len(value) - 1 else ""
                    lines.append("        " + json.dumps(lane) + lane_comma)
                lines.append("      ]" + comma)
            else:
                lines.append(f"      {json.dumps(key)}: {json.dumps(value)}{comma}")
        lines.append("    }" + ("," if i < len(videos) - 1 else ""))
    lines.append("  ]")

    others = {k: v for k, v in data.items() if k != "videos"}
    if others:
        lines[-1] += ","
        body = json.dumps(others, indent=2)[1:-1].strip("\n")
        lines.append(body)
    lines.append("}")
    return "\n".join(lines) + "\n"


class VideoRepository:
    def __init__(self, path: Path = VIDEOS_JSON, raw_dir: Path = RAW_VIDEO_DIR):
        self.path = Path(path)
        self.raw_dir = Path(raw_dir)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"videos": []}
        return read_json(self.path)

    def list(self) -> list:
        return [Video.from_dict(v) for v in self._read().get("videos", [])]

    def get(self, video_id: str) -> Video:
        for video in self.list():
            if video.id == video_id:
                return video
        raise KeyError(f"Video '{video_id}' not found in videos.json")

    def ids(self) -> set:
        return {v["id"] for v in self._read().get("videos", [])}

    def file_path(self, video: Video) -> Path:
        return self.raw_dir / video.path

    def save(self, video: Video):
        """Insert a new video (appended) or replace the entry with the same id (in place)."""
        data = self._read()
        videos = data.setdefault("videos", [])
        entry = video.to_dict()
        for i, existing in enumerate(videos):
            if existing["id"] == video.id:
                videos[i] = entry
                break
        else:
            videos.append(entry)
        write_text_atomic(self.path, format_videos_json(data))


class ClipRepository:
    def __init__(self, path: Path = CLIPS_JSON):
        self.path = Path(path)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"clips": []}
        return read_json(self.path)

    def list(self) -> list:
        return [Clip.from_dict(c) for c in self._read().get("clips", [])]

    def for_video(self, video_id: str) -> list:
        return [c for c in self.list() if c.video_id == video_id]

    def add(self, clip: Clip):
        data = self._read()
        clips = data.setdefault("clips", [])
        if any(c["id"] == clip.id for c in clips):
            raise ValueError(f"Clip id '{clip.id}' already exists")
        clips.append(clip.to_dict())
        save_json(self.path, data)
