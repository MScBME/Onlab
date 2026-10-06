import random
import shutil
from pathlib import Path

import yaml

from src.utils.paths import DATASET_DIR, PROCESSED_DIR


def build_dataset(
    source_dir: Path = None,
    base_out_dir: Path = None,
    split: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    source_dir = source_dir or PROCESSED_DIR
    base_out_dir = base_out_dir or DATASET_DIR

    dirs = {
        "train_img": base_out_dir / "images" / "train",
        "val_img": base_out_dir / "images" / "val",
        "test_img": base_out_dir / "images" / "test",
        "train_lbl": base_out_dir / "labels" / "train",
        "val_lbl": base_out_dir / "labels" / "val",
        "test_lbl": base_out_dir / "labels" / "test",
    }

    if base_out_dir.exists():
        shutil.rmtree(base_out_dir)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    images = [f.name for f in source_dir.iterdir() if f.suffix == ".jpg"]
    valid_pairs = [
        (img, img.replace(".jpg", ".txt"))
        for img in images
        if (source_dir / img.replace(".jpg", ".txt")).exists()
    ]

    if not valid_pairs:
        raise RuntimeError(f"No image/label pairs found in {source_dir}")

    random.seed(seed)
    random.shuffle(valid_pairs)

    total = len(valid_pairs)
    train_end = int(total * split[0])
    val_end = int(total * (split[0] + split[1]))

    splits = {
        "train": (valid_pairs[:train_end], dirs["train_img"], dirs["train_lbl"]),
        "val":   (valid_pairs[train_end:val_end], dirs["val_img"], dirs["val_lbl"]),
        "test":  (valid_pairs[val_end:], dirs["test_img"], dirs["test_lbl"]),
    }

    for name, (pairs, img_dest, lbl_dest) in splits.items():
        for img_name, lbl_name in pairs:
            shutil.copy(source_dir / img_name, img_dest / img_name)
            shutil.copy(source_dir / lbl_name, lbl_dest / lbl_name)

    yaml_content = {
        "path": str(base_out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "swimmer"},
    }

    yaml_path = base_out_dir / "swimmer_data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    return {
        "total": total,
        "train": train_end,
        "val": val_end - train_end,
        "test": total - val_end,
        "yaml_path": yaml_path,
    }
