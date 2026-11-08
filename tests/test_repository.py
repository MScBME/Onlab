import json

import pytest

from research.paths import TEST_LIBRARY, TRAIN_LIBRARY
from swimmertracker.catalog.models import DEFAULT_LANE_LENGTH_M, Clip, Lane, Video
from swimmertracker.catalog.repository import ClipRepository, VideoRepository, format_videos_json

LIBRARIES = [pytest.param(TEST_LIBRARY, id="test-library"), pytest.param(TRAIN_LIBRARY, id="train-library")]

HAND_MAINTAINED = """{
  "videos": [
    {
      "id": "a",
      "path": "a.mp4",
      "lanes": [
        {"id": 1, "coordinates": [[0, 912], [101, 834], [1618, 818], [1710, 891]]},
        {"id": 2, "coordinates": [[0, 834], [86, 773], [1543, 759], [1618, 818]], "length_m": 22.0}
      ]
    },
    {
      "id": "b",
      "path": "b.mp4",
      "lanes": [
        {"id": 1, "coordinates": [[0, 894], [106, 815], [1627, 803], [1725, 880]]},
        {"id": 2, "coordinates": [[0, 815], [89, 754], [1556, 746], [1627, 803]]},
        {"id": 3, "coordinates": [[0, 754], [75, 705], [1494, 697], [1556, 746]]}
      ]
    }
  ]
}
"""


def _read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("library", LIBRARIES)
def test_videos_writer_reproduces_the_hand_maintained_file(library):
    text = _read_text(library.videos_json)
    assert format_videos_json(json.loads(text)) == text


@pytest.mark.parametrize("library", LIBRARIES)
def test_clips_file_matches_save_json_format(library):
    text = _read_text(library.clips_json)
    assert json.dumps(json.loads(text), indent=2) + "\n" == text


def test_missing_length_defaults_to_25m():
    lane = Lane.from_dict({"id": 3, "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    assert lane.length_m == DEFAULT_LANE_LENGTH_M == 25.0


@pytest.fixture
def videos_json(tmp_path):
    path = tmp_path / "videos.json"
    path.write_text(HAND_MAINTAINED, encoding="utf-8")
    assert format_videos_json(json.loads(HAND_MAINTAINED)) == HAND_MAINTAINED
    return path


def test_save_replaces_only_the_target_entry(videos_json, tmp_path):
    repo = VideoRepository(videos_json, videos_dir=tmp_path)

    video = repo.get("b")
    video.lanes[0].length_m = 12.5
    repo.save(video)

    lines_before = HAND_MAINTAINED.splitlines()
    lines_after = _read_text(videos_json).splitlines()
    changed = [(a, b) for a, b in zip(lines_before, lines_after) if a != b]
    assert len(lines_before) == len(lines_after)
    assert len(changed) == 3  # every lane line of "b" now carries length_m, nothing else changed
    assert repo.get("b").lanes[0].length_m == 12.5
    assert "length_m" not in json.dumps(json.loads(_read_text(videos_json))["videos"][0]["lanes"][0])


def test_save_appends_new_video(videos_json, tmp_path):
    repo = VideoRepository(videos_json, videos_dir=tmp_path)
    count = len(repo.list())

    repo.save(Video(id="new_one", path="new one.mp4", lanes=[Lane(1, [[0, 0], [9, 0], [9, 9], [0, 9]])]))

    videos = repo.list()
    assert len(videos) == count + 1
    assert videos[-1].id == "new_one"
    assert '{"id": 1, "coordinates": [[0, 0], [9, 0], [9, 9], [0, 9]], "length_m": 25.0}' in _read_text(videos_json)
    assert repo.file_path(videos[-1]) == tmp_path / "new one.mp4"


def test_clip_repository_add_and_filter(tmp_path):
    repo = ClipRepository(tmp_path / "clips.json")
    repo.add(Clip(id="a", video_id="v1", lane=3, start_time="00:00:01", end_time="00:00:05"))
    repo.add(Clip(id="b", video_id="v2", lane=1, start_time="00:00:01", end_time="00:00:05", description="x"))
    assert [c.id for c in repo.for_video("v1")] == ["a"]
    assert json.loads(_read_text(tmp_path / "clips.json"))["clips"][1]["description"] == "x"
    assert "description" not in json.loads(_read_text(tmp_path / "clips.json"))["clips"][0]
