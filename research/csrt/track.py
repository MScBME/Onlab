"""The csrt-track command, manual CSRT tracking on a clip. A legacy reference for the thesis, it needs opencv-contrib."""
import argparse

import cv2
import numpy as np

from research.csrt.swimmer_tracker import SwimmerTracker
from research.evaluation.clips import clip_from_args, load_run_config
from research.evaluation.plots import plot_speed
from swimmertracker.analysis.speed import compute_speed
from swimmertracker.video.loader import VideoLoader

LANE_WIDTH_M = 2.5


def build_homography(lane_points, lane_length_m):
    real_world_corners = np.array([
        [0.0, 0.0],
        [0.0, LANE_WIDTH_M],
        [lane_length_m, LANE_WIDTH_M],
        [lane_length_m, 0.0],
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
    parser = argparse.ArgumentParser(prog="csrt-track", description="Manual CSRT tracking on a clip, the legacy reference.")
    parser.add_argument("--clip", help="Clip id, default active_clip_id of research/run_config.json")
    parser.add_argument("--lane", type=int, help="Lane id to use instead of the clip's lane")
    args = parser.parse_args()

    settings = load_run_config()
    show_tracking = settings.get("show_tracking", True)
    ctx = clip_from_args(args, settings)
    clip = ctx.clip
    lane_points = np.array(ctx.lane.coordinates, dtype=np.float32)
    homography = build_homography(lane_points, ctx.lane.length_m)

    print(f"Clip {clip.id} | video {ctx.video.id} | lane {ctx.lane.id}")

    loader = VideoLoader(str(ctx.video_path), start_sec=clip.start_sec, end_sec=clip.end_sec)
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
