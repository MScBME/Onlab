import json
import threading

import pytest

from swimmertracker.catalog import library
from swimmertracker.catalog.library import (
    CopyCancelled,
    VideoLibrary,
    copy_with_progress,
    import_target,
    prepare_lanes,
    suggest_video_id,
    validate_video_id,
)
from swimmertracker.catalog.models import Lane, Video
from swimmertracker.utils.paths import LibraryPaths

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


@pytest.mark.parametrize("video_id", ["d2du\n", "úszó", "d2du-2"])
def test_video_id_is_ascii_word_characters_only(video_id):
    assert validate_video_id(video_id, set()) is not None


def test_import_target(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    inside = videos / "a.mp4"
    inside.write_bytes(b"x")
    assert import_target(inside, videos) == inside

    outside = tmp_path / "a.mp4"
    outside.write_bytes(b"y")
    assert import_target(outside, videos) == videos / "a (2).mp4"


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


@pytest.mark.parametrize("length_m", [float("nan"), float("inf"), -12.5])
def test_lane_length_must_be_a_finite_positive_number(length_m):
    _, errors = prepare_lanes([Lane(3, LANE_A, length_m)], None)
    assert errors == ["The length of lane 3 must be a positive number of meters."]


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "library"
    (root / "videos").mkdir(parents=True)
    (root / "videos.json").write_text(json.dumps({"videos": [
        {"id": "v1", "path": "v1.mp4", "lanes": [
            {"id": 3, "coordinates": LANE_A}, {"id": 4, "coordinates": LANE_B}]},
    ]}))
    (root / "clips.json").write_text(json.dumps({"clips": [
        {"id": "c1", "video_id": "v1", "lane": 4, "start_time": "00:00:01", "end_time": "00:00:02"},
        {"id": "c2", "video_id": "v1", "lane": 4, "start_time": "00:00:03", "end_time": "00:00:04"}]}))
    return VideoLibrary.open(LibraryPaths(root))


def test_library_layout(lib, tmp_path):
    assert lib.videos.path == tmp_path / "library" / "videos.json"
    assert lib.videos.videos_dir == tmp_path / "library" / "videos"
    assert lib.clips.path == tmp_path / "library" / "clips.json"


def test_lane_references(lib):
    refs = lib.lane_references("v1", 4)
    assert refs.clips == ["c1", "c2"]
    assert refs.describe() == "2 clips"
    assert not lib.lane_references("v1", 3).any
    assert lib.lane_references("v1", 3).describe() == ""
    assert library.LaneReferences(clips=["c1"]).describe() == "1 clip"


def test_referenced_lane_cannot_be_removed(lib):
    with pytest.raises(ValueError, match="Lane 4 is used by 2 clips and cannot be removed"):
        lib.save_lanes("v1", [Lane(3, LANE_A)])


def test_unreferenced_lane_can_be_removed_and_edited(lib):
    video = lib.save_lanes("v1", [Lane(4, LANE_B, 50.0)])
    assert [lane.id for lane in video.lanes] == [4]
    assert lib.videos.get("v1").lanes[0].length_m == 50.0


def test_add_video_requires_file_in_videos_dir_and_unique_id(lib, tmp_path):
    inside = lib.videos.videos_dir / "new.mp4"
    inside.write_bytes(b"x")
    with pytest.raises(ValueError, match="already exists"):
        lib.add_video("v1", inside, [Lane(1, LANE_A)])
    with pytest.raises(ValueError, match="inside"):
        lib.add_video("v2", tmp_path / "elsewhere.mp4", [Lane(1, LANE_A)])
    video = lib.add_video("v2", inside, [Lane(1, LANE_A)])
    assert video.path == "new.mp4"
    assert isinstance(lib.videos.get("v2"), Video)
