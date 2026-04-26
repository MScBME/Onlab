import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import SwimmerDetector
from src.training.lane_processor import DST_PTS, LANE_H, LANE_W
from src.training.metadata import load_video_catalog
from src.utils.config import load_json, time_to_seconds
from src.utils.paths import CLIPS_JSON, PROJECT_ROOT, RAW_VIDEO_DIR, RUN_CONFIG_JSON
from src.video.loader import VideoLoader
from src.visualization.plots import plot_speed

POOL_LENGTH_M = 25.0


def get_clip(clips_data, clip_id):
    for clip in clips_data.get("clips", []):
        if clip["id"] == clip_id:
            return clip
    print(f"Error: Clip id '{clip_id}' not found in {CLIPS_JSON}")
    sys.exit(1)


def compute_speed_from_meters(positions_m, timestamps, window_size=15):
    pos = np.array(positions_m)
    ts = np.array(timestamps)
    dp = np.abs(np.diff(pos))
    dt = np.diff(ts)
    raw_speeds = dp / dt
    kernel = np.ones(window_size) / window_size
    smoothed_speeds = np.convolve(raw_speeds, kernel, mode="valid")
    smoothed_ts = ts[1:][window_size - 1:]
    return smoothed_ts, smoothed_speeds


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a clip.")
    parser.add_argument("--clip", help="Clip id from clips.json (overrides active_clip_id in run_config.json)")
    parser.add_argument("--lane", type=int, help="Lane id (overrides the lane in clips.json)")
    parser.add_argument("--model", help="Path to YOLO model .pt (overrides run_config.json)")
    args = parser.parse_args()

    settings = load_json(RUN_CONFIG_JSON)
    clip_id = args.clip or settings.get("active_clip_id")
    if not clip_id:
        print("Error: no clip id specified (--clip or active_clip_id in run_config.json)")
        sys.exit(1)

    model_arg = args.model or settings.get("model_path", "models/swimmer.pt")
    model_path = Path(model_arg) if Path(model_arg).is_absolute() else PROJECT_ROOT / model_arg
    confidence = settings.get("confidence_threshold", 0.5)
    show_tracking = settings.get("show_tracking", True)

    clips_data = load_json(CLIPS_JSON)
    clip = get_clip(clips_data, clip_id)

    lane_id = args.lane or clip.get("lane")
    if lane_id is None:
        print(f"Error: lane not specified for clip '{clip_id}' (add 'lane' to clips.json or use --lane)")
        sys.exit(1)

    video_catalog = load_video_catalog()
    video = video_catalog.get_video(clip["video_id"])
    video_path = RAW_VIDEO_DIR / video["path"]
    lane_coords = np.array(
        video_catalog.get_lane_coordinates(clip["video_id"], lane_id), dtype=np.float32
    )

    start_sec = time_to_seconds(clip.get("start_time", "00:00:00"))
    end_sec = time_to_seconds(clip.get("end_time"))

    matrix = cv2.getPerspectiveTransform(lane_coords, DST_PTS)
    matrix_inv = cv2.getPerspectiveTransform(DST_PTS, lane_coords)

    print(f"Clip: {clip_id} | video: {clip['video_id']} | lane: {lane_id} | model: {model_path}")

    detector = SwimmerDetector(str(model_path), confidence=confidence)
    loader = VideoLoader(str(video_path), start_sec=start_sec, end_sec=end_sec)

    positions_m = []
    timestamps = []

    print("Running YOLO detection... (press ESC to stop)")

    for _frame_id, timestamp, frame in loader.read_frames():
        warped = cv2.warpPerspective(frame, matrix, (LANE_W, LANE_H))
        detection = detector.detect(warped)

        if detection is not None:
            (cx, cy), (x1, y1, x2, y2), conf = detection
            pos_m = cy * (POOL_LENGTH_M / LANE_H)
            positions_m.append(pos_m)
            timestamps.append(timestamp)

            if show_tracking:
                cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(warped, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                pt_warped = np.array([[[cx, cy]]], dtype=np.float32)
                pt_orig = cv2.perspectiveTransform(pt_warped, matrix_inv)[0][0]
                cv2.circle(frame, (int(pt_orig[0]), int(pt_orig[1])), 5, (0, 255, 0), -1)
        else:
            if show_tracking:
                cv2.putText(frame, "No detection", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if show_tracking:
            pts = lane_coords.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.imshow("Original", frame)
            cv2.imshow("Lane (bird's-eye)", warped)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()

    if len(positions_m) > 15:
        smoothed_ts, speeds = compute_speed_from_meters(positions_m, timestamps)
        smoothed_ts = smoothed_ts - smoothed_ts[0]
        plot_speed(smoothed_ts, speeds)
    else:
        print("Not enough detections to compute speed.")


if __name__ == "__main__":
    main()
