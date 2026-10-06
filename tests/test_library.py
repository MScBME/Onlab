import json
import threading

import pytest

from src.catalog import library
from src.catalog.library import (
    CopyCancelled,
    VideoLibrary,
    copy_with_progress,
    import_target,
    prepare_lanes,
    suggest_video_id,
    validate_video_id,
)
from src.catalog.models import Lane, Video
from src.catalog.repository import ClipRepository, VideoRepository

LANE_A = [[0, 774], [90, 724], [1487, 713], [1548, 761]]
LANE_B = [[90, 724], [165, 685], [1438, 675], [1487, 713]]


def test_suggest_video_id_follows_existing_convention():
    assert suggest_video_id("2026-09-29 14.40.31.mp4", set()) == "2026_09_29_14_40_31"
    assert suggest_video_id("Csik D2DU.MP4", {"csik_d2du"}) == "csik_d2du_2"
    assert suggest_video_id("---.mp4", set()) == "video"


def test_validate_video_id():
    assert validate_video_id("", set()) is not None
    assert validate_video_id("has space", set()) is not None
    assert validate_video_id("d2du", {"d2du"}) is not None
    assert validate_video_id("d2du_new", {"d2du"}) is None


def test_import_target(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    inside = raw / "a.mp4"
    inside.write_bytes(b"x")
    assert import_target(inside, raw) == inside

    outside = tmp_path / "a.mp4"
    outside.write_bytes(b"y")
    assert import_target(outside, raw) == raw / "a (2).mp4"


def test_copy_with_progress_and_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "COPY_CHUNK_BYTES", 4)
    src = tmp_path / "src.bin"
    src.write_bytes(b"0123456789abcdef")
    progress = []
    copy_with_progress(src, tmp_path / "out" / "dst.bin", progress.append)
    assert (tmp_path / "out" / "dst.bin").read_bytes() == src.read_bytes()
    assert progress[-1] == 1.0

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CopyCancelled):
        copy_with_progress(src, tmp_path / "cancelled.bin", cancel=cancel)
    assert not (tmp_path / "cancelled.bin").exists()
    assert not (tmp_path / "cancelled.bin.part").exists()


def test_prepare_lanes_normalizes_and_validates():
    shuffled = [LANE_A[2], LANE_A[0], LANE_A[3], LANE_A[1]]
    lanes, errors = prepare_lanes([Lane(3, shuffled, 12.5)], frame_size=(1920, 1080))
    assert errors == []
    assert lanes[0].coordinates == LANE_A and lanes[0].length_m == 12.5

    _, errors = prepare_lanes([], None)
    assert errors == ["A video needs at least one lane."]
    _, errors = prepare_lanes([Lane(3, LANE_A), Lane(3, LANE_B)], None)
    assert "Lane IDs must be unique." in errors
    _, errors = prepare_lanes([Lane(3, LANE_A, 0)], None)
    assert any("length" in e for e in errors)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "videos.json").write_text(json.dumps({"videos": [
        {"id": "v1", "path": "v1.mp4", "lanes": [
            {"id": 3, "coordinates": LANE_A}, {"id": 4, "coordinates": LANE_B}]},
    ]}))
    (tmp_path / "events.json").write_text(json.dumps({"events": [
        {"video_id": "v1", "lanes": [4], "start_time": "00:00:01", "end_time": "00:00:02"}]}))
    (tmp_path / "event_frames.json").write_text(json.dumps({"event_frames": [
        {"id": "v1_00_00_01_000", "lanes": [4]}, {"id": "v1_00_00_02_000", "lanes": [3, 4]}]}))
    (tmp_path / "custom_frames.json").write_text(json.dumps({"custom_frames": []}))
    monkeypatch.setattr(library, "EVENTS_JSON", tmp_path / "events.json")
    monkeypatch.setattr(library, "EVENT_FRAMES_JSON", tmp_path / "event_frames.json")
    monkeypatch.setattr(library, "CUSTOM_FRAMES_JSON", tmp_path / "custom_frames.json")
    return VideoLibrary(VideoRepository(tmp_path / "videos.json", raw), ClipRepository(tmp_path / "clips.json"))


def test_lane_references(lib):
    refs = lib.lane_references("v1", 4)
    assert (refs.events, refs.event_frames, refs.custom_frames) == (1, 2, 0)
    assert refs.describe() == "1 training event, 2 training frames"
    assert not lib.lane_references("v1", 7).any


def test_referenced_lane_cannot_be_removed(lib):
    with pytest.raises(ValueError, match="Lane 4 cannot be removed"):
        lib.save_lanes("v1", [Lane(3, LANE_A)])


def test_unreferenced_lane_can_be_removed_and_edited(lib, tmp_path):
    (tmp_path / "event_frames.json").write_text(json.dumps({"event_frames": [{"id": "v1_00_00_01_000", "lanes": [4]}]}))
    video = lib.save_lanes("v1", [Lane(4, LANE_B, 50.0)])
    assert [lane.id for lane in video.lanes] == [4]
    assert lib.videos.get("v1").lanes[0].length_m == 50.0


def test_add_video_requires_file_in_raw_dir_and_unique_id(lib, tmp_path):
    inside = lib.videos.raw_dir / "new.mp4"
    inside.write_bytes(b"x")
    with pytest.raises(ValueError, match="already exists"):
        lib.add_video("v1", inside, [Lane(1, LANE_A)])
    with pytest.raises(ValueError, match="inside"):
        lib.add_video("v2", tmp_path / "elsewhere.mp4", [Lane(1, LANE_A)])
    video = lib.add_video("v2", inside, [Lane(1, LANE_A)])
    assert video.path == "new.mp4"
    assert isinstance(lib.videos.get("v2"), Video)
