import pytest

from src.catalog.clips import create_clip, propose_clip, suggest_clip_id
from src.catalog.repository import ClipRepository


def test_suggest_clip_id():
    assert suggest_clip_id("d2du", 3, 474.4, set()) == "d2du_lane3_00_07_54"
    assert suggest_clip_id("d2du", 3, 474.4, {"d2du_lane3_00_07_54"}) == "d2du_lane3_00_07_54_2"


def test_create_clip_fills_everything_but_the_description(tmp_path):
    repo = ClipRepository(tmp_path / "clips.json")
    clip = create_clip(repo, "d2du", 3, 474.0, 487.25, "  lane 3, right  ")
    assert clip.to_dict() == {
        "id": "d2du_lane3_00_07_54",
        "video_id": "d2du",
        "lane": 3,
        "start_time": "00:07:54",
        "end_time": "00:08:07.250",
        "description": "lane 3, right",
    }
    assert [c.id for c in repo.list()] == ["d2du_lane3_00_07_54"]


def test_duplicate_range_is_rejected(tmp_path):
    repo = ClipRepository(tmp_path / "clips.json")
    create_clip(repo, "d2du", 3, 474.0, 487.0)
    _, duplicate = propose_clip(repo, "d2du", 3, 474.0, 487.0)
    assert duplicate is not None
    with pytest.raises(ValueError, match="already saved"):
        create_clip(repo, "d2du", 3, 474.0, 487.0)
    other_lane = create_clip(repo, "d2du", 4, 474.0, 487.0)
    assert other_lane.id == "d2du_lane4_00_07_54"


def test_empty_range_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        create_clip(ClipRepository(tmp_path / "clips.json"), "d2du", 3, 10.0, 10.0)
