import shutil
from pathlib import Path

import cv2
import numpy as np

from src.training.metadata import FrameRecord, VideoCatalog
from src.utils.paths import PROCESSED_DIR

LANE_W = 256
LANE_H = 1024
DST_PTS = np.array([[0, 0], [LANE_W, 0], [LANE_W, LANE_H], [0, LANE_H]], dtype="float32")


def _warp_frame_for_lane(img: np.ndarray, lane_coords: list):
    src_pts = np.array(lane_coords, dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, DST_PTS)
    warped = cv2.warpPerspective(img, matrix, (LANE_W, LANE_H))
    return warped, matrix, src_pts


def _transform_labels(labels, img_shape, src_pts, matrix):
    h_orig, w_orig = img_shape[:2]
    result = []
    for label in labels:
        _, x_c, y_c, w_b, h_b = map(float, label)
        px_x, px_y = x_c * w_orig, y_c * h_orig

        if cv2.pointPolygonTest(src_pts, (px_x, px_y), False) < 0:
            continue

        pt_center = np.array([[[px_x, px_y]]], dtype="float32")
        new_center = cv2.perspectiveTransform(pt_center, matrix)[0][0]

        px_w, px_h = w_b * w_orig, h_b * h_orig
        pt_edge = np.array([[[px_x + px_w / 2, px_y + px_h / 2]]], dtype="float32")
        new_edge = cv2.perspectiveTransform(pt_edge, matrix)[0][0]

        new_x = new_center[0] / LANE_W
        new_y = new_center[1] / LANE_H
        new_wb = min((abs(new_edge[0] - new_center[0]) * 2) / LANE_W, 1.0)
        new_hb = min((abs(new_edge[1] - new_center[1]) * 2) / LANE_H, 1.0)

        result.append(f"0 {new_x:.6f} {new_y:.6f} {new_wb:.6f} {new_hb:.6f}")
    return result


def _process_lane(rec: FrameRecord, lane_id: int, img, labels, video_catalog: VideoCatalog, output_dir: Path) -> str:
    try:
        lane_coords = video_catalog.get_lane_coordinates(rec.video_id, lane_id)
    except KeyError as e:
        print(f"[WARN] {rec.filename}: {e}")
        return "missing_lane"

    warped, matrix, src_pts = _warp_frame_for_lane(img, lane_coords)
    lane_labels = _transform_labels(labels, img.shape, src_pts, matrix)
    if not lane_labels:
        return "no_labels_in_lane"

    out_name = f"{rec.id}_L{lane_id}"
    cv2.imwrite(str(output_dir / f"{out_name}.jpg"), warped)
    (output_dir / f"{out_name}.txt").write_text("\n".join(lane_labels), encoding="utf-8")
    return "processed"


def _load_record_assets(rec: FrameRecord) -> tuple:
    img_path = rec.annotated_path
    if img_path is None or not img_path.exists():
        return None, None, "missing_image"
    label_path = img_path.with_suffix(".txt")
    if not label_path.exists():
        return None, None, "missing_label"

    img = cv2.imread(str(img_path))
    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines() if line.strip()]
    return img, labels, None


def process_records(records: list, video_catalog: VideoCatalog, output_dir: Path = None) -> dict:
    output_dir = output_dir or PROCESSED_DIR
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"processed": 0, "missing_image": 0, "missing_label": 0, "no_labels_in_lane": 0, "missing_lane": 0}

    for rec in records:
        img, labels, err = _load_record_assets(rec)
        if err:
            summary[err] += 1
            continue
        for lane_id in rec.lanes:
            outcome = _process_lane(rec, lane_id, img, labels, video_catalog, output_dir)
            summary[outcome] = summary.get(outcome, 0) + 1

    return summary
