import cv2
import numpy as np
import sys
import os
from src.video.loader import VideoLoader
from src.tracking.swimmer_tracker import SwimmerTracker
from src.analysis.speed import compute_speed
from src.visualization.plots import plot_speed
from src.utils.config import load_json

def get_video_data(videos_data, video_id):
    """Finds the specific video in the loaded configuration list."""
    for video in videos_data.get("videos", []):
        if video["id"] == video_id:
            return video
            
    print(f"Error: Video ID '{video_id}' not found in the configuration.")
    sys.exit(1)

def main():
    settings = load_json("config/run_config.json")
    active_id = settings.get("active_video_id")
    videos_path = settings.get("video_config_path", "data/videos.json")
    show_tracking = settings.get("show_tracking", True)

    if not active_id:
        print("Error: 'active_video_id' missing in run_config.json")
        sys.exit(1)

    print(f"Loading configuration for: {active_id}")
    videos_data = load_json(videos_path)
    video_data = get_video_data(videos_data, active_id)

    raw_video_dir = os.path.join("data", "raw")
    video_path = os.path.join(raw_video_dir, video_data["video_path"])
    
    start_sec = video_data.get("start_sec", 0.0)
    end_sec = video_data.get("end_sec", None)
    
    if "pool_corners" not in video_data:
        print(f"Error: 'pool_corners' missing for video ID '{active_id}'")
        sys.exit(1)
        
    lane_points = np.array(video_data["pool_corners"], dtype=np.float32)

    real_world_corners = np.array([
        [0.0, 0.0],  # bottom_left
        [0.0, 2.5],  # top_left
        [25.0, 2.5], # top_right
        [25.0, 0.0]  # bottom_right
    ], dtype=np.float32)

    H, _ = cv2.findHomography(lane_points, real_world_corners)

    loader = VideoLoader(video_path, start_sec=start_sec, end_sec=end_sec)
    tracker = SwimmerTracker(lane_points)

    xs, ys, ts = [], [], []

    for frame_id, timestamp, frame in loader.read_frames():
        
        if not tracker.initialized:
            tracker.initialize(frame)
            cv2.destroyWindow("Select swimmer") 

        pos = tracker.update(frame)

        if pos is None:
            cv2.putText(frame, "Tracking Lost!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if show_tracking:
                cv2.imshow("Swimmer Tracking", frame)
            if cv2.waitKey(30) & 0xFF == 27: break
            continue

        x, y = pos
        xs.append(x)
        ys.append(y)
        ts.append(timestamp)

        if show_tracking:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            pts = np.array(tracker.lane_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.imshow("Swimmer Tracking", frame)

            if cv2.waitKey(30) & 0xFF == 27:
                break

    cv2.destroyAllWindows()

    if len(xs) > 15: 
        smoothed_ts, speeds = compute_speed(xs, ys, ts, H) 
        smoothed_ts = smoothed_ts - smoothed_ts[0]
        plot_speed(smoothed_ts, speeds)
    else:
        print("Not enough data collected to plot speed.")

if __name__ == "__main__":
    main()