import cv2
import numpy as np
from src.video.loader import VideoLoader
from src.tracking.swimmer_tracker import SwimmerTracker
from src.analysis.speed import compute_speed
from src.visualization.plots import plot_speed

def main():
    video_path = "data/raw/csik_d2du.mp4"

    lane_points = np.array([
        [0, 772],    # bottom left
        [84, 725],   # top left
        [1487, 713], # top right
        [1542, 762]  # bottom right
    ], dtype=np.float32)
    
    real_world_corners = np.array([
        [0.0, 0.0], 
        [0.0, 2.5], 
        [25.0, 2.5], 
        [25.0, 0.0]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(lane_points, real_world_corners)

    loader = VideoLoader(video_path, start_sec=43 * 60 + 23, end_sec=43 * 60 + 36)
    tracker = SwimmerTracker(lane_points)

    xs = []
    ys = []
    ts = []

    for frame_id, timestamp, frame in loader.read_frames():
        
        if not tracker.initialized:
            tracker.initialize(frame)
            cv2.destroyWindow("Select swimmer") 

        pos = tracker.update(frame)

        if pos is None:
            cv2.putText(frame, "Tracking Lost!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Swimmer Tracking", frame)
            if cv2.waitKey(30) & 0xFF == 27: break
            continue

        x, y = pos
        xs.append(x)
        ys.append(y)
        ts.append(timestamp)

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