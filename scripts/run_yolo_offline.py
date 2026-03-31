import cv2
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.video.loader import VideoLoader
from src.detection.detector import SwimmerDetector
from src.visualization.plots import plot_speed
from src.utils.config import load_json, time_to_seconds

# Warped lane dimensions (must match training data from process.py)
LANE_W, LANE_H = 256, 1024

# Real-world pool dimensions
POOL_LENGTH_M = 25.0


def get_video_data(videos_data, video_id):
    for video in videos_data.get("videos", []):
        if video["id"] == video_id:
            return video
    print(f"Error: Video ID '{video_id}' not found.")
    sys.exit(1)


def compute_speed_from_meters(positions_m, timestamps, window_size=15):
    """Compute smoothed speed from position (meters) and timestamps."""
    pos = np.array(positions_m)
    ts = np.array(timestamps)

    dp = np.abs(np.diff(pos))
    dt = np.diff(ts)
    raw_speeds = dp / dt

    kernel = np.ones(window_size) / window_size
    smoothed_speeds = np.convolve(raw_speeds, kernel, mode='valid')
    smoothed_ts = ts[1:][window_size - 1:]

    return smoothed_ts, smoothed_speeds


def main():
    settings = load_json(os.path.join(PROJECT_ROOT, "config", "run_config.json"))
    active_id = settings.get("active_video_id")
    videos_path = os.path.join(PROJECT_ROOT, settings.get("video_config_path", "data/videos.json"))
    show_tracking = settings.get("show_tracking", True)
    model_path = os.path.join(PROJECT_ROOT, settings.get("model_path", "models/swimmer.pt"))
    confidence = settings.get("confidence_threshold", 0.5)

    if not active_id:
        print("Error: 'active_video_id' missing in run_config.json")
        sys.exit(1)

    print(f"Loading configuration for: {active_id}")
    videos_data = load_json(videos_path)
    video_data = get_video_data(videos_data, active_id)

    video_path = os.path.join(PROJECT_ROOT, "data", "raw", video_data["video_path"])
    start_sec = time_to_seconds(video_data.get("start_time", "00:00:00"))
    end_sec = time_to_seconds(video_data.get("end_time"))

    if "pool_corners" not in video_data:
        print(f"Error: 'pool_corners' missing for video ID '{active_id}'")
        sys.exit(1)

    lane_points = np.array(video_data["pool_corners"], dtype=np.float32)

    # Perspective transform: original lane -> bird's-eye view (same as training)
    dst_pts = np.array([[0, 0], [LANE_W, 0], [LANE_W, LANE_H], [0, LANE_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(lane_points, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, lane_points)

    detector = SwimmerDetector(model_path, confidence=confidence)
    loader = VideoLoader(video_path, start_sec=start_sec, end_sec=end_sec)

    positions_m = []
    timestamps = []

    print("Running YOLO detection... (press ESC to stop)")

    for frame_id, timestamp, frame in loader.read_frames():
        warped = cv2.warpPerspective(frame, M, (LANE_W, LANE_H))

        detection = detector.detect(warped)

        if detection is not None:
            (cx, cy), (x1, y1, x2, y2), conf = detection

            # Y-axis of warped image = pool length direction
            pos_m = cy * (POOL_LENGTH_M / LANE_H)
            positions_m.append(pos_m)
            timestamps.append(timestamp)

            if show_tracking:
                # Draw on warped view
                cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(warped, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw on original frame
                pt_warped = np.array([[[cx, cy]]], dtype=np.float32)
                pt_orig = cv2.perspectiveTransform(pt_warped, M_inv)[0][0]
                cv2.circle(frame, (int(pt_orig[0]), int(pt_orig[1])), 5, (0, 255, 0), -1)
        else:
            if show_tracking:
                cv2.putText(frame, "No detection", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if show_tracking:
            pts = lane_points.astype(np.int32).reshape((-1, 1, 2))
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