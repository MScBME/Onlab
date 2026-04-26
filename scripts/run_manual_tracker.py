import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.speed import compute_speed
from src.tracking.swimmer_tracker import SwimmerTracker
from src.training.metadata import load_video_catalog
from src.utils.config import load_json, time_to_seconds
from src.utils.paths import CLIPS_JSON, RAW_VIDEO_DIR, RUN_CONFIG_JSON
from src.video.loader import VideoLoader
from src.visualization.plots import plot_speed

POOL_LENGTH_M = 25.0
LANE_WIDTH_M = 2.5


def get_clip(clips_data, clip_id):
    for clip in clips_data.get("clips", []):
        if clip["id"] == clip_id:
            return clip
    print(f"Error: Clip id '{clip_id}' not found in {CLIPS_JSON}")
    sys.exit(1)


def resolve_clip(args, settings):
    clip_id = args.clip or settings.get("active_clip_id")
    if not clip_id:
        print("Error: no clip id specified (--clip or active_clip_id in run_config.json)")
        sys.exit(1)

    clips_data = load_json(CLIPS_JSON)
    clip = get_clip(clips_data, clip_id)

    lane_id = args.lane or clip.get("lane")
    if lane_id is None:
        print(f"Error: lane not specified for clip '{clip_id}' (add 'lane' to clips.json or use --lane)")
        sys.exit(1)

    return clip_id, clip, lane_id


def build_homography(lane_points):
    real_world_corners = np.array([
        [0.0, 0.0],
        [0.0, LANE_WIDTH_M],
        [POOL_LENGTH_M, LANE_WIDTH_M],
        [POOL_LENGTH_M, 0.0],
    ], dtype=np.float32)
    homography, _ = cv2.findHomography(lane_points, real_world_corners)
    return homography


def render_tracking(frame, pos, lane_points):
    if pos is None:
        cv2.putText(frame, "Tracking Lost!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        x, y = pos
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    pts = np.array(lane_points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.imshow("Swimmer Tracking", frame)
    return cv2.waitKey(30) & 0xFF == 27


def track_clip(loader, tracker, show_tracking):
    xs, ys, ts = [], [], []
    for _frame_id, timestamp, frame in loader.read_frames():
        if not tracker.initialized:
            tracker.initialize(frame)
            cv2.destroyWindow("Select swimmer")

        pos = tracker.update(frame)
        if pos is not None:
            x, y = pos
            xs.append(x)
            ys.append(y)
            ts.append(timestamp)

        if show_tracking and render_tracking(frame, pos, tracker.lane_points):
            break

    cv2.destroyAllWindows()
    return xs, ys, ts


def main():
    parser = argparse.ArgumentParser(description="Manual CSRT tracking on a clip (legacy reference).")
    parser.add_argument("--clip", help="Clip id from clips.json (overrides active_clip_id in run_config.json)")
    parser.add_argument("--lane", type=int, help="Override the clip's lane")
    args = parser.parse_args()

    settings = load_json(RUN_CONFIG_JSON)
    show_tracking = settings.get("show_tracking", True)

    clip_id, clip, lane_id = resolve_clip(args, settings)

    video_catalog = load_video_catalog()
    video = video_catalog.get_video(clip["video_id"])
    video_path = RAW_VIDEO_DIR / video["path"]
    lane_points = np.array(
        video_catalog.get_lane_coordinates(clip["video_id"], lane_id), dtype=np.float32
    )

    start_sec = time_to_seconds(clip.get("start_time", "00:00:00"))
    end_sec = time_to_seconds(clip.get("end_time"))

    homography = build_homography(lane_points)

    print(f"Clip: {clip_id} | video: {clip['video_id']} | lane: {lane_id}")

    loader = VideoLoader(str(video_path), start_sec=start_sec, end_sec=end_sec)
    tracker = SwimmerTracker(lane_points)

    xs, ys, ts = track_clip(loader, tracker, show_tracking)

    if len(xs) > 15:
        smoothed_ts, speeds = compute_speed(xs, ys, ts, homography)
        smoothed_ts = smoothed_ts - smoothed_ts[0]
        plot_speed(smoothed_ts, speeds)
    else:
        print("Not enough data collected to plot speed.")


if __name__ == "__main__":
    main()
