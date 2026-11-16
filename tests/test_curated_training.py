"""Tests for full-frame curated dataset loading, filtering, train/val splitting, and CLI preparation."""

import json
from pathlib import Path

import pytest
import yaml

from research.training import train as train_cli
from research.training.curated_dataset import (
    build_full_frame_dataset,
    filter_curated_samples,
    load_curated_samples,
    validate_datasets,
)
from research.training.model_metadata import (
    environment_info,
    full_frame_input,
    json_safe_value,
    parse_results_csv,
)
from research.training.onnx_export import _resolve_export_imgsz
from swimmertracker.catalog.model_catalog import list_models


def _make_sample(
    root: Path,
    entry_id: str,
    dataset: str,
    split: str,
    filename: str,
    *,
    is_reject: bool = False,
    image_quality: int | None = 2,
    annot_quality: int | None = 2,
    txt_lines: list[str] | None = None,
    custom_boxes: list[dict] | None = None,
) -> tuple[dict, dict]:
    split_dir = root / dataset / split
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    rel_img = f"{dataset}/{split}/{filename}"
    rel_txt = f"{dataset}/{split}/{stem}.txt"
    (root / rel_img).write_bytes(b"fakejpg")
    lines = ["6 0.5 0.5 0.1 0.1"] if txt_lines is None else txt_lines
    (root / rel_txt).write_text("\n".join(lines), encoding="utf-8")

    rating = {
        "id": entry_id,
        "dataset": dataset,
        "split": split,
        "filename": filename,
        "rel_path": rel_img,
        "image_quality": image_quality,
        "annot_quality": annot_quality,
        "is_reject": is_reject,
        "notes": "",
    }
    if custom_boxes is not None:
        rating["custom_boxes"] = custom_boxes

    cache_item = {
        "id": entry_id,
        "dataset": dataset,
        "split": split,
        "filename": filename,
        "rel_path": rel_img,
        "txt_rel_path": rel_txt,
        "initial_box_count": len(lines),
        "boxes": [],
    }
    return rating, cache_item


@pytest.fixture
def curated_dir(tmp_path: Path) -> Path:
    root = tmp_path / "public_annotations"
    root.mkdir()
    specs = [
        # Race 1 across two datasets + two original splits -> must merge into ONE group
        ("0", "detswim.v2i.darknet", "train", "Nice_100-BRASSE-DAMES-serie-5_100_jpg.rf.aaa1.jpg", False, 3, 3, ["2 0.2 0.3 0.1 0.2"], None),
        ("1", "aerial pool.v4-release-filtered.darknet", "valid", "Nice_100-BRASSE-DAMES-sA-rie-5_150_jpg.rf.bbb2.jpg", False, 3, 3, ["1 0.5 0.5 0.2 0.2"], None),
        # Race 2 in detswim (different split test -> merged into unified train+val pool)
        ("2", "detswim.v2i.darknet", "test", "Angers19_100-DOS_200_jpg.rf.ccc3.jpg", False, 2, 2, ["0 0.4 0.4 0.1 0.1"], [{"class_id": 2, "class_name": "Swimmer", "x_center": 0.6, "y_center": 0.7, "width": 0.15, "height": 0.25}]),
        # Race 3 in detswim
        ("3", "detswim.v2i.darknet", "train", "Rennes19_50-NL_300_jpg.rf.ddd4.jpg", False, 3, 2, ["0 0.3 0.3 0.1 0.1"], []),
        # Race 4 & 5 in MediaEval
        ("4", "MediaEval.v1i.darknet", "train", "ClipA_10_jpg.rf.eee5.jpg", False, 2, 3, ["0 0.8 0.8 0.1 0.1"], None),
        ("5", "MediaEval.v1i.darknet", "valid", "ClipB_20_jpg.rf.fff6.jpg", False, 2, 3, ["0 0.7 0.7 0.1 0.1"], None),
        # Unnamed dataset -> must stay strictly in train when named groups exist
        ("6", "CS-vision-dataset-swimming.v2-v2-working.darknet", "valid", "youtube-10_jpg.rf.ggg7.jpg", False, 3, 3, ["2 0.4 0.4 0.1 0.1"], None),
        # Rejected sample -> excluded by default
        ("7", "detswim.v2i.darknet", "train", "RejectedRace_99_jpg.rf.hhh8.jpg", True, None, None, ["0 0.5 0.5 0.2 0.2"], None),
    ]
    ratings = {}
    cache = []
    for eid, ds, sp, fn, rej, iq, aq, txt_lines, cboxes in specs:
        r, c = _make_sample(
            root, eid, ds, sp, fn,
            is_reject=rej, image_quality=iq, annot_quality=aq,
            txt_lines=txt_lines, custom_boxes=cboxes,
        )
        ratings[eid] = r
        cache.append(c)
    (root / "curation_ratings.json").write_text(json.dumps(ratings, indent=2), encoding="utf-8")
    (root / "curation_index_cache.json").write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return root


def test_load_and_filter_curated_samples(curated_dir: Path):
    samples, datasets = load_curated_samples(curated_dir)
    assert len(samples) == 8
    assert "detswim.v2i.darknet" in datasets

    # Default excludes is_reject == True
    default = filter_curated_samples(samples)
    assert [s.id for s in default] == ["0", "1", "2", "3", "4", "5", "6"]

    # Explicit is_reject == "true" selects only rejected
    only_rejected = filter_curated_samples(samples, is_reject="true")
    assert [s.id for s in only_rejected] == ["7"]

    # is_reject == "all" keeps all entries
    assert len(filter_curated_samples(samples, is_reject="all")) == 8

    # Minimum image/annot quality excludes null ratings too
    high_img = filter_curated_samples(samples, is_reject="all", min_image_quality=3)
    assert [s.id for s in high_img] == ["0", "1", "3", "6"]

    high_both = filter_curated_samples(samples, min_image_quality=2, min_annot_quality=3)
    assert [s.id for s in high_both] == ["0", "1", "4", "5", "6"]

    ds_only = filter_curated_samples(samples, datasets=["MediaEval.v1i.darknet"])
    assert [s.id for s in ds_only] == ["4", "5"]


def test_validate_datasets_rejects_unknown_name():
    assert validate_datasets(["ds_a", "ds_a"], ["ds_a", "ds_b"]) == ["ds_a"]
    with pytest.raises(ValueError, match="Unknown dataset"):
        validate_datasets(["missing_ds"], ["ds_a", "ds_b"])


def test_load_works_without_index_cache(curated_dir: Path):
    (curated_dir / "curation_index_cache.json").unlink()
    samples, _ = load_curated_samples(curated_dir)
    assert samples[0].txt_path == curated_dir / "detswim.v2i.darknet" / "train" / "Nice_100-BRASSE-DAMES-serie-5_100_jpg.rf.aaa1.txt"


def test_build_full_frame_dataset_random_backup_and_drops_test(curated_dir: Path, tmp_path: Path):
    samples, _ = load_curated_samples(curated_dir)
    selected = filter_curated_samples(samples)
    out_dir_a = tmp_path / "yolo_ds_a"
    out_dir_b = tmp_path / "yolo_ds_b"

    summary_a = build_full_frame_dataset(selected, out_dir=out_dir_a, val_fraction=0.3, split_seed=42)
    summary_b = build_full_frame_dataset(selected, out_dir=out_dir_b, val_fraction=0.3, split_seed=42)

    assert summary_a["method"] == "random seeded split"
    assert summary_a["train"]["sha1"] == summary_b["train"]["sha1"]
    assert summary_a["val"]["sha1"] == summary_b["val"]["sha1"]
    assert summary_a["val"]["images"] == 2
    assert summary_a["train"]["images"] == 5

    assert not (out_dir_a / "images" / "test").exists()
    data = yaml.safe_load((out_dir_a / "swimmer_data.yaml").read_text(encoding="utf-8"))
    assert "test" not in data
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"] == {0: "swimmer"}

    # Custom boxes override (id=2) and empty custom boxes (id=3) still work regardless of split
    lbl_dirs = [out_dir_a / "labels" / "train", out_dir_a / "labels" / "val"]
    id2_txt = next(d / "2_Angers19_100-DOS_200_jpg.rf.ccc3.txt" for d in lbl_dirs if (d / "2_Angers19_100-DOS_200_jpg.rf.ccc3.txt").exists())
    assert id2_txt.read_text(encoding="utf-8").strip() == "0 0.600000 0.700000 0.150000 0.250000"
    id3_txt = next(d / "3_Rennes19_50-NL_300_jpg.rf.ddd4.txt" for d in lbl_dirs if (d / "3_Rennes19_50-NL_300_jpg.rf.ddd4.txt").exists())
    assert id3_txt.read_text(encoding="utf-8") == ""


def test_explicit_val_videos_and_cli_skip_train(curated_dir: Path, tmp_path: Path, monkeypatch):
    ds_out = tmp_path / "dataset_out"
    monkeypatch.setattr(train_cli, "DATASET_DIR", ds_out)

    train_cli.main([
        "--name", "smoke",
        "--annotations-dir", str(curated_dir),
        "--val-videos", "Nice_100-BRASSE*",
        "--skip-train",
    ])

    val_imgs = {p.name for p in (ds_out / "images" / "val").iterdir()}
    assert val_imgs == {
        "0_Nice_100-BRASSE-DAMES-serie-5_100_jpg.rf.aaa1.jpg",
        "1_Nice_100-BRASSE-DAMES-sA-rie-5_150_jpg.rf.bbb2.jpg",
    }
    assert not (ds_out / "images" / "test").exists()


def test_val_split_dir_uses_curated_validate_folder(curated_dir: Path, tmp_path: Path, monkeypatch):
    split_root = tmp_path / "split_clean"
    (split_root / "train").mkdir(parents=True)
    (split_root / "validate").mkdir(parents=True)
    (split_root / "validate" / "0_Nice_100-BRASSE-DAMES-serie-5_100_jpg.rf.aaa1.jpg").write_bytes(b"img")
    (split_root / "validate" / "1_Nice_100-BRASSE-DAMES-sA-rie-5_150_jpg.rf.bbb2.jpg").write_bytes(b"img")
    (split_root / "train" / "2_Angers19_100-DOS_200_jpg.rf.ccc3.jpg").write_bytes(b"img")

    ds_out = tmp_path / "dataset_from_dir"
    monkeypatch.setattr(train_cli, "DATASET_DIR", ds_out)

    train_cli.main([
        "--name", "from_dir",
        "--annotations-dir", str(curated_dir),
        "--val-split-dir", str(split_root),
        "--skip-train",
    ])
    val_imgs = {p.name for p in (ds_out / "images" / "val").iterdir()}
    assert val_imgs == {
        "0_Nice_100-BRASSE-DAMES-serie-5_100_jpg.rf.aaa1.jpg",
        "1_Nice_100-BRASSE-DAMES-sA-rie-5_150_jpg.rf.bbb2.jpg",
    }

    samples, _ = load_curated_samples(curated_dir)
    selected = filter_curated_samples(samples)
    direct_summary = build_full_frame_dataset(
        selected,
        out_dir=tmp_path / "dataset_direct_val",
        val_split_dir=split_root / "validate",
    )
    assert direct_summary["method"] == "val_split_dir"
    assert direct_summary["val"]["images"] == 2
    assert direct_summary["train"]["images"] == 5


def test_full_frame_model_metadata_and_export_shape(tmp_path: Path):
    assert _resolve_export_imgsz(full_frame_input(1024), 1024) == (1024, 1024)
    assert _resolve_export_imgsz(full_frame_input(1000), 1000) == (1024, 1024)

    onnx_file = tmp_path / "swimmer_ff.onnx"
    onnx_file.write_bytes(b"onnx")
    onnx_file.with_suffix(".json").write_text(json.dumps({"input": full_frame_input(1024)}), encoding="utf-8")
    models = list_models(tmp_path)
    assert [m.name for m in models] == ["swimmer_ff"]

    legacy_file = tmp_path / "swimmer_warped.onnx"
    legacy_file.write_bytes(b"onnx")
    legacy_file.with_suffix(".json").write_text(json.dumps({"input": {"type": "lane_warp"}}), encoding="utf-8")
    assert [m.name for m in list_models(tmp_path)] == ["swimmer_ff"]


def test_parse_results_csv_and_metadata_helpers(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,40.0,2.3,0.50,0.52,0.48,0.18\n"
        "2,80.5,1.9,0.72,0.68,0.71,0.42\n"
        "3,120.25,1.8,0.70,0.69,0.70,0.39\n",
        encoding="utf-8",
    )
    parsed = parse_results_csv(csv_path)
    assert parsed["epochs_completed"] == 3
    assert parsed["training_time_s"] == 120.25
    assert parsed["best_epoch"] == 2
    assert parsed["best_val_map50_95"] == 0.42
    assert parsed["final_epoch_metrics"]["metrics/mAP50-95(B)"] == 0.39

    assert json_safe_value({"p": Path("a/b"), "nested": [1, Path("c")]}) == {"nested": [1, "c"], "p": "a/b"}
    env = environment_info("cuda:0")
    assert env["device"] == "cuda:0" and "python" in env


def test_cli_records_full_training_metadata(curated_dir: Path, tmp_path: Path, monkeypatch):
    ds_out = tmp_path / "dataset_out"
    runs_dir = tmp_path / "runs"
    pt_path = tmp_path / "models" / "swimmer_meta_test.pt"
    pt_path.parent.mkdir(parents=True)
    pt_path.write_bytes(b"fakept")

    monkeypatch.setattr(train_cli, "DATASET_DIR", ds_out)
    monkeypatch.setattr(train_cli, "TRAINING_RUNS_DIR", runs_dir)

    fake_trainer_info = {
        "resolved_batch_size": 6,
        "resolved_device": "cuda:0",
        "amp": True,
        "epochs_completed": 50,
        "training_time_s": 1842.5,
        "best_epoch": 43,
        "best_val_map50_95": 0.6123,
        "final_epoch_metrics": {"metrics/mAP50-95(B)": 0.601},
        "ultralytics_args": {"batch": 6, "imgsz": 800, "lr0": 0.01, "mosaic": 1.0},
    }
    monkeypatch.setattr(train_cli, "train_with_info", lambda **_kw: (pt_path, fake_trainer_info))

    class FakeValResults:
        results_dict = {
            "metrics/precision(B)": 0.81,
            "metrics/recall(B)": 0.74,
            "metrics/mAP50(B)": 0.83,
            "metrics/mAP50-95(B)": 0.6123,
        }

    class FakeYOLO:
        def __init__(self, _path):
            pass

        def val(self, **_kw):
            return FakeValResults()

    monkeypatch.setattr(train_cli, "YOLO", FakeYOLO)

    train_cli.main([
        "--name", "meta_test",
        "--annotations-dir", str(curated_dir),
        "--min-image-quality", "2",
        "--imgsz", "800",
        "--batch", "-1",
    ])

    saved = json.loads(pt_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["input"] == {"type": "full_frame", "imgsz": 800}
    assert saved["classes"] == ["swimmer"]
    assert saved["metrics"]["val"]["map50_95"] == 0.6123
    assert saved["metrics"]["final_epoch"]["metrics/mAP50-95(B)"] == 0.601
    t = saved["training"]
    assert t["filters"] == {
        "is_reject": "false",
        "min_image_quality": 2,
        "min_annot_quality": None,
        "datasets": None,
    }
    assert t["args"]["batch"] == -1
    assert t["args"]["imgsz"] == 800
    assert t["args"]["workers"] == 2
    assert t["resolved"]["batch_size"] == 6
    assert t["resolved"]["device"] == "cuda:0"
    assert t["resolved"]["best_epoch"] == 43
    assert t["ultralytics_args"]["lr0"] == 0.01
    assert "train" in t["split"] and "val" in t["split"]
    assert "python" in t["environment"]
