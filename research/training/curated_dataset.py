"""Full-frame curated dataset loader, filter, and YOLO train/val dataset builder.

Reads curated metadata from ``curation_ratings.json`` (and optionally
``curation_index_cache.json``) in a local annotations root such as
``public_annotations/``, filters samples by ``is_reject``, minimum quality
ratings, and dataset name, collapses every annotation box to class ``0``
(``swimmer``), and assigns images to ``train`` or ``val`` via ``val_split_dir``,
``val_videos`` filename globs, or a seeded random fallback.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from research.paths import DATASET_DIR, PUBLIC_ANNOTATIONS_DIR

SWIMMER_CLASS_ID = 0
SPLITS = ("train", "val")
IS_REJECT_CHOICES = ("false", "true", "all")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True)
class CuratedSample:
    """One curated full-frame image and its metadata."""

    id: str
    dataset: str
    source_split: str
    filename: str
    image_path: Path
    txt_path: Path
    image_quality: int | None
    annot_quality: int | None
    is_reject: bool
    custom_boxes: list[dict] | None = None


def _sort_key(entry_id: str) -> tuple[int, int | str]:
    return (0, int(entry_id)) if entry_id.isdigit() else (1, entry_id)


def load_curated_samples(
    annotations_dir: Path = PUBLIC_ANNOTATIONS_DIR,
) -> tuple[list[CuratedSample], list[str]]:
    """Load all curated samples and sorted dataset names from ``annotations_dir``."""
    annotations_dir = Path(annotations_dir)
    ratings_path = annotations_dir / "curation_ratings.json"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Curation ratings file not found: {ratings_path}")

    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))

    txt_by_id: dict[str, Path] = {}
    cache_path = annotations_dir / "curation_index_cache.json"
    if cache_path.exists():
        for item in json.loads(cache_path.read_text(encoding="utf-8")):
            txt_rel = item.get("txt_rel_path")
            if txt_rel:
                txt_by_id[str(item["id"])] = annotations_dir / txt_rel

    samples: list[CuratedSample] = []
    datasets: set[str] = set()
    for raw_id in sorted(ratings, key=_sort_key):
        entry = ratings[raw_id]
        entry_id = str(entry.get("id", raw_id))
        dataset = str(entry["dataset"])
        datasets.add(dataset)

        rel_path = entry["rel_path"]
        image_path = annotations_dir / rel_path
        txt_path = txt_by_id.get(entry_id, image_path.with_suffix(".txt"))

        samples.append(
            CuratedSample(
                id=entry_id,
                dataset=dataset,
                source_split=str(entry.get("split", "")),
                filename=str(entry.get("filename", image_path.name)),
                image_path=image_path,
                txt_path=txt_path,
                image_quality=entry.get("image_quality"),
                annot_quality=entry.get("annot_quality"),
                is_reject=bool(entry.get("is_reject", False)),
                custom_boxes=entry.get("custom_boxes"),
            )
        )

    return samples, sorted(datasets)


def validate_datasets(
    requested: Sequence[str] | None,
    available: Sequence[str],
) -> list[str] | None:
    """Validate requested dataset names against available ones and return deduplicated list."""
    if not requested:
        return None
    available_set = set(available)
    unknown = sorted(name for name in requested if name not in available_set)
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available datasets: {', '.join(available)}"
        )
    return list(dict.fromkeys(requested))


def filter_curated_samples(
    samples: Sequence[CuratedSample],
    *,
    is_reject: str = "false",
    min_image_quality: int | None = None,
    min_annot_quality: int | None = None,
    datasets: Sequence[str] | None = None,
) -> list[CuratedSample]:
    """Filter curated samples according to rejection, quality, and dataset criteria."""
    if is_reject not in IS_REJECT_CHOICES:
        raise ValueError(f"is_reject must be one of {IS_REJECT_CHOICES}, got {is_reject!r}")

    dataset_filter = set(datasets) if datasets else None
    selected: list[CuratedSample] = []
    for sample in samples:
        if is_reject == "false" and sample.is_reject:
            continue
        if is_reject == "true" and not sample.is_reject:
            continue
        if min_image_quality is not None and (
            sample.image_quality is None or sample.image_quality < min_image_quality
        ):
            continue
        if min_annot_quality is not None and (
            sample.annot_quality is None or sample.annot_quality < min_annot_quality
        ):
            continue
        if dataset_filter is not None and sample.dataset not in dataset_filter:
            continue
        selected.append(sample)
    return selected


def _format_yolo_line(x_center: float, y_center: float, width: float, height: float) -> str | None:
    xc = max(0.0, min(1.0, float(x_center)))
    yc = max(0.0, min(1.0, float(y_center)))
    w = max(0.0, min(1.0, float(width)))
    h = max(0.0, min(1.0, float(height)))
    if w <= 0.0 or h <= 0.0:
        return None
    return f"{SWIMMER_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _resolve_yolo_lines(sample: CuratedSample) -> tuple[list[str] | None, str]:
    """Resolve YOLO-formatted label lines for a sample without cropping or warping."""
    if not sample.image_path.exists():
        return None, "missing_image"

    if sample.custom_boxes is not None:
        lines: list[str] = []
        for box in sample.custom_boxes:
            formatted = _format_yolo_line(
                box["x_center"], box["y_center"], box["width"], box["height"]
            )
            if formatted is not None:
                lines.append(formatted)
        return lines, "ok"

    if not sample.txt_path.exists():
        return None, "missing_label"

    lines = []
    for raw_line in sample.txt_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        formatted = _format_yolo_line(
            float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        )
        if formatted is not None:
            lines.append(formatted)
    return lines, "ok"


def _resolve_val_image_dir(val_split_dir: Path) -> Path:
    """Locate the validation images folder inside a split directory or direct path."""
    root = Path(val_split_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Validation split directory not found: {root}")
    for sub in ("validate", "val", "images/validate", "images/val"):
        candidate = root / sub
        if candidate.is_dir():
            return candidate
    return root


def _load_val_stems_and_names(val_split_dir: Path) -> tuple[set[str], set[str]]:
    val_dir = _resolve_val_image_dir(val_split_dir)
    stems: set[str] = set()
    names: set[str] = set()
    for path in val_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stems.add(path.stem.lower())
            names.add(path.name.lower())
    if not stems:
        raise RuntimeError(f"No validation images found in {val_dir}")
    return stems, names


def _sample_in_val_dir(sample: CuratedSample, val_stems: set[str], val_names: set[str]) -> bool:
    exported_stem = f"{sample.id}_{sample.image_path.stem}".lower()
    return (
        exported_stem in val_stems
        or sample.image_path.stem.lower() in val_stems
        or sample.filename.lower() in val_names
        or sample.image_path.name.lower() in val_names
    )


def _matches_val_patterns(sample: CuratedSample, patterns: Sequence[str]) -> bool:
    candidates = (sample.filename.lower(), sample.image_path.stem.lower())
    return any(
        fnmatch.fnmatch(cand, pat.lower())
        for pat in patterns
        for cand in candidates
    )


def _assign_train_val(
    ready_items: list[tuple[CuratedSample, list[str]]],
    *,
    val_fraction: float,
    seed: int,
    val_videos: Sequence[str] | None,
    val_split_dir: Path | None,
) -> tuple[dict[str, list[tuple[CuratedSample, list[str]]]], str]:
    """Assign ready items into ``train`` and ``val``."""
    if val_split_dir is not None:
        val_stems, val_names = _load_val_stems_and_names(val_split_dir)
        val_items = [(s, l) for s, l in ready_items if _sample_in_val_dir(s, val_stems, val_names)]
        train_items = [(s, l) for s, l in ready_items if not _sample_in_val_dir(s, val_stems, val_names)]
        method = "val_split_dir"
    elif val_videos:
        val_items = [(s, l) for s, l in ready_items if _matches_val_patterns(s, val_videos)]
        train_items = [(s, l) for s, l in ready_items if not _matches_val_patterns(s, val_videos)]
        method = "val_videos patterns"
    else:
        if len(ready_items) < 2:
            raise RuntimeError("Need at least 2 usable images to form train and val splits")
        shuffled = list(ready_items)
        random.Random(seed).shuffle(shuffled)
        val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_fraction)))
        val_items = shuffled[:val_count]
        train_items = shuffled[val_count:]
        method = "random seeded split"

    if not train_items or not val_items:
        raise RuntimeError(
            f"Split resulted in empty {'train' if not train_items else 'val'} set "
            f"(train={len(train_items)}, val={len(val_items)})"
        )
    return {"train": train_items, "val": val_items}, method


def build_full_frame_dataset(
    samples: Sequence[CuratedSample],
    out_dir: Path = DATASET_DIR,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_videos: Sequence[str] | None = None,
    val_split_dir: Path | None = None,
) -> dict:
    """Copy full-frame images and collapsed YOLO labels into ``train`` and ``val`` under ``out_dir``."""
    out_dir = Path(out_dir)
    ready_items: list[tuple[CuratedSample, list[str]]] = []
    counts: Counter[str] = Counter()

    for sample in samples:
        lines, status = _resolve_yolo_lines(sample)
        if status != "ok":
            counts[status] += 1
            continue
        if not lines:
            counts["empty_label"] += 1
        counts["total_boxes"] += len(lines)
        ready_items.append((sample, lines))

    if not ready_items:
        raise RuntimeError("No usable images are available after filtering and asset checks")

    split_plan, method = _assign_train_val(
        ready_items,
        val_fraction=val_fraction,
        seed=split_seed,
        val_videos=val_videos,
        val_split_dir=val_split_dir,
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)

    summary: dict = {
        "method": method,
        "val_fraction": val_fraction,
        "seed": split_seed,
        "val_videos_patterns": list(val_videos) if val_videos else None,
        "val_split_dir": Path(val_split_dir).resolve().as_posix() if val_split_dir else None,
        "missing_image": counts["missing_image"],
        "missing_label": counts["missing_label"],
        "empty_label": counts["empty_label"],
        "total_boxes": counts["total_boxes"],
    }

    for split in SPLITS:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        written_names: list[str] = []
        split_boxes = 0
        split_empty = 0
        by_dataset: Counter[str] = Counter()

        for sample, lines in split_plan[split]:
            stem = f"{sample.id}_{sample.image_path.stem}"
            img_name = f"{stem}{sample.image_path.suffix}"
            shutil.copy2(sample.image_path, img_dir / img_name)
            label_text = ("\n".join(lines) + "\n") if lines else ""
            (lbl_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")

            written_names.append(img_name)
            split_boxes += len(lines)
            if not lines:
                split_empty += 1
            by_dataset[sample.dataset] += 1

        summary[split] = {
            "images": len(written_names),
            "boxes": split_boxes,
            "empty_label_images": split_empty,
            "by_dataset": dict(sorted(by_dataset.items())),
            "sha1": hashlib.sha1("\n".join(sorted(written_names)).encode("utf-8")).hexdigest(),
        }

    yaml_content = {
        "path": out_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {SWIMMER_CLASS_ID: "swimmer"},
    }
    yaml_path = out_dir / "swimmer_data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    summary["yaml_path"] = yaml_path
    return summary
