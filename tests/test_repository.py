import json
import shutil

from src.catalog.models import DEFAULT_LANE_LENGTH_M, Clip, Lane, Video
from src.catalog.repository import ClipRepository, VideoRepository, format_videos_json
from src.utils.paths import CLIPS_JSON, VIDEOS_JSON


def _read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_videos_writer_reproduces_the_hand_maintained_file():
    text = _read_text(VIDEOS_JSON)
    assert format_videos_json(json.loads(text)) == text


def test_clips_file_matches_save_json_format():
    text = _read_text(CLIPS_JSON)
    assert json.dumps(json.loads(text), indent=2) + "\n" == text


def test_missing_length_defaults_to_25m():
    lane = Lane.from_dict({"id": 3, "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    assert lane.length_m == DEFAULT_LANE_LENGTH_M == 25.0


def test_cli_catalog_reads_lane_length_with_default(tmp_path, monkeypatch):
    from src.training import metadata

    path = tmp_path / "videos.json"
    path.write_text(json.dumps({"videos": [{"id": "v", "path": "v.mp4", "lanes": [
        {"id": 1, "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        {"id": 2, "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]], "length_m": 12.5}]}]}))
    monkeypatch.setattr(metadata, "VIDEOS_JSON", path)
    catalog = metadata.load_video_catalog()
    assert catalog.get_lane_length("v", 1) == 25.0
    assert catalog.get_lane_length("v", 2) == 12.5


def test_save_replaces_only_the_target_entry(tmp_path):
    path = tmp_path / "videos.json"
    shutil.copy(VIDEOS_JSON, path)
    original = _read_text(path)
    repo = VideoRepository(path, raw_dir=tmp_path)

    video = repo.get("d3du")
    video.lanes[0].length_m = 12.5
    repo.save(video)

    lines_before = original.splitlines()
    lines_after = _read_text(path).splitlines()
    changed = [(a, b) for a, b in zip(lines_before, lines_after) if a != b]
    assert len(lines_before) == len(lines_after)
    assert len(changed) == 9  # every d3du lane line now carries length_m, nothing else changed
    assert repo.get("d3du").lanes[0].length_m == 12.5
    assert "length_m" not in json.dumps(json.loads(_read_text(path))["videos"][0])


def test_save_appends_new_video(tmp_path):
    path = tmp_path / "videos.json"
    shutil.copy(VIDEOS_JSON, path)
    repo = VideoRepository(path, raw_dir=tmp_path)
    count = len(repo.list())

    repo.save(Video(id="new_one", path="new one.mp4", lanes=[Lane(1, [[0, 0], [9, 0], [9, 9], [0, 9]])]))

    videos = repo.list()
    assert len(videos) == count + 1
    assert videos[-1].id == "new_one"
    assert '{"id": 1, "coordinates": [[0, 0], [9, 0], [9, 9], [0, 9]], "length_m": 25.0}' in _read_text(path)


def test_clip_repository_add_and_filter(tmp_path):
    repo = ClipRepository(tmp_path / "clips.json")
    repo.add(Clip(id="a", video_id="v1", lane=3, start_time="00:00:01", end_time="00:00:05"))
    repo.add(Clip(id="b", video_id="v2", lane=1, start_time="00:00:01", end_time="00:00:05", description="x"))
    assert [c.id for c in repo.for_video("v1")] == ["a"]
    assert json.loads(_read_text(tmp_path / "clips.json"))["clips"][1]["description"] == "x"
    assert "description" not in json.loads(_read_text(tmp_path / "clips.json"))["clips"][0]
