import json

import pytest
import yaml

from research.training import metadata
from research.training.dataset_builder import build_dataset, choose_val_groups, frame_groups
from research.training.lane_processor import _transform_labels, lane_image_name
from research.training.metadata import TEST, TRAIN, FrameRecord, load_video_catalog
from swimmertracker.utils.paths import LibraryPaths

LANE = [[0, 774], [90, 724], [1487, 713], [1548, 761]]


def rec(frame_id, video_id, t, stroke="freestyle", role=TRAIN, lanes=(1,)):
    return FrameRecord(id=frame_id, video_id=video_id, lanes=list(lanes), stroke=stroke, gender="male",
                       role=role, timestamp_s=t)


def test_frame_groups_chain_close_frames_of_one_file():
    records = [rec("a1", "a", 10), rec("a2", "a", 12), rec("a3", "a", 70), rec("a4", "a", 200),
               rec("b1", "a_before", 11), rec("c1", "c", 10),
               FrameRecord(id="custom_1", video_id="a", lanes=[1], stroke="freestyle", gender="male")]
    files = {"a": "a.mp4", "a_before": "a.mp4", "c": "c.mp4"}
    groups = frame_groups(records, lambda r: files[r.video_id], gap_s=60)
    assert groups["a1"] == groups["a2"] == groups["b1"] == groups["a3"]  # 10, 11, 12 and 70 s, no gap above 60 s
    assert groups["a4"] != groups["a1"]
    assert groups["c1"] != groups["a1"]
    assert groups["custom_1"] == "single:custom_1"


def test_val_groups_follow_each_strokes_share():
    groups = {f"fr{i}": ("freestyle", 10) for i in range(10)}
    groups.update({f"br{i}": ("breaststroke", 10) for i in range(4)})
    val = choose_val_groups(groups, val_fraction=0.2, seed=1)
    assert sum(1 for g in val if g.startswith("fr")) == 2
    assert sum(1 for g in val if g.startswith("br")) == 1  # 0.2 * 40 = 8 -> one group of 10
    assert choose_val_groups(groups, 0.2, seed=1) == val  # deterministic


def test_every_stroke_keeps_a_training_group_and_val_is_never_empty():
    assert choose_val_groups({"only": ("butterfly", 30)}, 0.9, seed=0) == set()
    val = choose_val_groups({"a": ("butterfly", 30), "b": ("freestyle", 30), "c": ("freestyle", 5)}, 0.1, seed=0)
    assert val == {"c"}  # no group brings a stroke closer to 10 %, so the smallest one goes to val


@pytest.fixture
def libraries(tmp_path):
    def library(name, videos):
        root = tmp_path / name
        (root / "videos").mkdir(parents=True)
        (root / "videos.json").write_text(json.dumps({"videos": [
            {"id": vid, "path": path, "lanes": [{"id": 1, "coordinates": LANE}]} for vid, path in videos]}))
        return LibraryPaths(root)

    return (library("train", [("tr", "train.mp4"), ("tr_before", "train.mp4")]),
            library("test", [("te", "test.mp4")]))


def test_catalog_roles_and_paths(libraries):
    train, test = libraries
    catalog = load_video_catalog(train, test)
    assert catalog.role("tr") == catalog.role("tr_before") == TRAIN
    assert catalog.role("te") == TEST
    assert catalog.file_path("te") == test.videos_dir / "test.mp4"
    assert catalog.get_lane_length("tr", 1) == 25.0
    with pytest.raises(KeyError):
        catalog.role("missing")


def test_a_video_cannot_be_in_both_libraries(libraries):
    train, test = libraries
    (test.root / "videos.json").write_text(json.dumps({"videos": [
        {"id": "tr", "path": "other.mp4", "lanes": [{"id": 1, "coordinates": LANE}]}]}))
    with pytest.raises(ValueError, match="both the train and the test library"):
        load_video_catalog(train, test)
    (test.root / "videos.json").write_text(json.dumps({"videos": [
        {"id": "copy", "path": "train.mp4", "lanes": [{"id": 1, "coordinates": LANE}]}]}))
    with pytest.raises(ValueError, match="referenced from both"):
        load_video_catalog(train, test)


def test_frame_records_get_the_role_of_their_video(libraries, tmp_path, monkeypatch, capsys):
    train, test = libraries
    frames = tmp_path / "event_frames.json"
    frames.write_text(json.dumps({"event_frames": [
        {"id": "tr_00_01_02_500", "lanes": [1], "stroke": "freestyle", "gender": "male"},
        {"id": "te_00_00_10_000", "lanes": [1], "stroke": "breaststroke", "gender": "female"},
        {"id": "gone_00_00_01_000", "lanes": [1], "stroke": "freestyle", "gender": "male"}]}))
    monkeypatch.setattr(metadata, "EVENT_FRAMES_JSON", frames)
    monkeypatch.setattr(metadata, "CUSTOM_FRAMES_JSON", tmp_path / "none.json")
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    for name in ("tr_00_01_02_500", "te_00_00_10_000", "gone_00_00_01_000"):
        (annotated / f"{name}.jpg").write_bytes(b"")

    records = {r.id: r for r in metadata.load_frame_records(annotated, load_video_catalog(train, test))}
    assert set(records) == {"tr_00_01_02_500", "te_00_00_10_000"}
    assert records["tr_00_01_02_500"].role == TRAIN and records["tr_00_01_02_500"].timestamp_s == 62.5
    assert records["te_00_00_10_000"].role == TEST
    assert "gone" in capsys.readouterr().out


def _processed(tmp_path, records):
    source = tmp_path / "processed"
    source.mkdir()
    items = []
    for r in records:
        for lane in r.lanes:
            name = lane_image_name(r.id, lane)
            (source / f"{name}.jpg").write_bytes(b"jpg")
            (source / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            items.append((name, r))
    return source, items


def test_build_dataset_keeps_test_videos_in_test_and_groups_together(libraries, tmp_path):
    catalog = load_video_catalog(*libraries)
    records = ([rec(f"tr_{i}", "tr", t, lanes=(1, 2)) for i, t in enumerate([0, 2, 4, 300, 302, 600, 602, 900])]
               + [rec("trb_0", "tr_before", 3, lanes=(1, 2))]
               + [rec(f"te_{i}", "te", 100 * i, stroke="breaststroke", role=TEST) for i in range(3)])
    source, items = _processed(tmp_path, records)

    summary = build_dataset(items, catalog, source_dir=source, base_out_dir=tmp_path / "dataset",
                            val_fraction=0.25, gap_s=60, seed=3)

    assert summary["test"]["lane_images"] == 3 and summary["test"]["videos"] == ["te"]
    assert "te" not in summary["train"]["videos"] + summary["val"]["videos"]
    assert summary["train"]["lane_images"] + summary["val"]["lane_images"] == 18
    assert summary["groups"]["val"] >= 1 and sum(summary["groups"].values()) == 4  # 0-4 s, 300 s, 600 s, 900 s
    val_names = {p.stem for p in (tmp_path / "dataset" / "images" / "val").iterdir()}
    train_names = {p.stem for p in (tmp_path / "dataset" / "images" / "train").iterdir()}
    early = {lane_image_name(i, lane) for i in ("tr_0", "tr_1", "tr_2", "trb_0") for lane in (1, 2)}
    assert early <= val_names or early <= train_names  # one group, the same file with frames 0 to 4 s apart
    data = yaml.safe_load((tmp_path / "dataset" / "swimmer_data.yaml").read_text())
    assert data["test"] == "images/test" and data["names"] == {0: "swimmer"}


def test_build_dataset_without_test_frames_has_no_test_split(libraries, tmp_path):
    catalog = load_video_catalog(*libraries)
    source, items = _processed(tmp_path, [rec("tr_0", "tr", 0), rec("tr_1", "tr", 500)])
    summary = build_dataset(items, catalog, source_dir=source, base_out_dir=tmp_path / "dataset")
    assert not summary["has_test"]
    assert "test" not in yaml.safe_load((tmp_path / "dataset" / "swimmer_data.yaml").read_text())
    assert summary["val"]["lane_images"] == 1


def test_lane_labels_are_collapsed_to_the_swimmer_class():
    import cv2
    import numpy as np
    from swimmertracker.video.lane_warp import DST_PTS

    src = np.array(LANE, dtype="float32")
    matrix = cv2.getPerspectiveTransform(src, DST_PTS)
    labels = [["2", "0.4", "0.68", "0.02", "0.04"]]  # for example turn in the body direction project
    out = _transform_labels(labels, (1080, 1920), src, matrix)
    assert len(out) == 1 and out[0].startswith("0 ")
