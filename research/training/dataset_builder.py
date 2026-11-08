"""Split of the lane images into train, val and test.

Every lane image whose video is in the test library goes to test, these videos never give training data. The
images of the train library go to train or val. Consecutive frames of a video show the same swimmers, so the split
is made by groups. Frames of the same video file closer than ``gap_s`` seconds form one group, chained, and a group
lands entirely in train or in val. Groups are chosen per stroke, so every stroke gets about ``val_fraction`` of its
images into val, as far as whole groups allow.
"""
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from research.paths import DATASET_DIR, PROCESSED_DIR
from research.training.metadata import TEST, TRAIN, VideoCatalog

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_GAP_S = 60.0
DEFAULT_SPLIT_SEED = 42
SPLITS = ("train", "val", "test")


def frame_groups(records, video_key, gap_s: float = DEFAULT_GAP_S) -> dict:
    """Map frame ids to group ids. Frames without a timestamp, the custom images, are groups of their own."""
    groups = {}
    by_video = defaultdict(list)
    for rec in records:
        if rec.timestamp_s is None:
            groups[rec.id] = f"single:{rec.id}"
        else:
            by_video[video_key(rec)].append(rec)
    for key in sorted(by_video):
        index, previous = 0, None
        for rec in sorted(by_video[key], key=lambda r: r.timestamp_s):
            if previous is not None and rec.timestamp_s - previous > gap_s:
                index += 1
            groups[rec.id] = f"{key}#{index}"
            previous = rec.timestamp_s
    return groups


def choose_val_groups(groups: dict, val_fraction: float, seed: int) -> set:
    """Return the ids of the groups that go to val.

    ``groups`` maps a group id to its stroke and its number of lane images. The groups of each stroke are taken in
    a seeded random order, and a group moves to val when that brings the val share of its stroke closer to
    val_fraction. The first group of every stroke always stays in train. If val is still empty, which happens with
    very small selections, the smallest group goes to val.
    """
    rng = random.Random(seed)
    by_stroke = defaultdict(list)
    for gid in sorted(groups):
        stroke, size = groups[gid]
        by_stroke[stroke].append((gid, size))

    val = set()
    for stroke in sorted(by_stroke):
        items = by_stroke[stroke]
        rng.shuffle(items)
        target = val_fraction * sum(size for _, size in items)
        taken = 0
        for gid, size in items[1:]:
            if abs(taken + size - target) < abs(taken - target):
                val.add(gid)
                taken += size

    if not val and len(groups) > 1:
        val.add(min(sorted(groups), key=lambda gid: groups[gid][1]))
    return val


def _digest(names) -> str:
    return hashlib.sha1("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def build_dataset(
    items: list,
    catalog: VideoCatalog,
    source_dir: Path = None,
    base_out_dir: Path = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    gap_s: float = DEFAULT_GAP_S,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    """``items`` are the pairs of lane image name and FrameRecord that lane_processor.process_records returns."""
    source_dir = source_dir or PROCESSED_DIR
    base_out_dir = base_out_dir or DATASET_DIR

    train_items = [(name, rec) for name, rec in items if rec.role == TRAIN]
    if not train_items:
        raise RuntimeError("There are no lane images from the train library, nothing to train on")

    records = {rec.id: rec for _, rec in train_items}
    group_of = frame_groups(records.values(), lambda rec: str(catalog.file_path(rec.video_id)), gap_s)
    strokes_per_group = defaultdict(Counter)
    for _, rec in train_items:
        strokes_per_group[group_of[rec.id]][rec.stroke] += 1
    groups = {gid: (counts.most_common(1)[0][0], sum(counts.values())) for gid, counts in strokes_per_group.items()}
    val_groups = choose_val_groups(groups, val_fraction, seed)

    assignment = {"train": [], "val": [], "test": []}
    for name, rec in items:
        if rec.role == TEST:
            assignment["test"].append((name, rec))
        else:
            assignment["val" if group_of[rec.id] in val_groups else "train"].append((name, rec))

    if base_out_dir.exists():
        shutil.rmtree(base_out_dir)
    for split in SPLITS:
        (base_out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name, _rec in assignment[split]:
            shutil.copy(source_dir / f"{name}.jpg", base_out_dir / "images" / split / f"{name}.jpg")
            shutil.copy(source_dir / f"{name}.txt", base_out_dir / "labels" / split / f"{name}.txt")

    yaml_content = {"path": base_out_dir.resolve().as_posix(), "train": "images/train", "val": "images/val"}
    if assignment["test"]:
        yaml_content["test"] = "images/test"
    yaml_content["names"] = {0: "swimmer"}
    yaml_path = base_out_dir / "swimmer_data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    summary = {
        "yaml_path": yaml_path,
        "has_test": bool(assignment["test"]),
        "method": "test is the test library videos, train and val are the train library videos in time groups per stroke",
        "val_fraction": val_fraction,
        "gap_s": gap_s,
        "seed": seed,
        "groups": {"train": len(groups) - len(val_groups), "val": len(val_groups)},
    }
    for split in SPLITS:
        names = [name for name, _ in assignment[split]]
        summary[split] = {
            "lane_images": len(names),
            "frames": len({rec.id for _, rec in assignment[split]}),
            "by_stroke": dict(sorted(Counter(rec.stroke for _, rec in assignment[split]).items())),
            "videos": sorted({rec.video_id for _, rec in assignment[split]}),
            "sha1": _digest(names),
        }
    return summary
