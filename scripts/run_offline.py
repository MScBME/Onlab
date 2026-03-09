from src.video.loader import VideoLoader
from src.tracking.swimmer_tracker import SwimmerTracker
from src.analysis.speed import compute_speed
from src.visualization.plots import plot_speed


def main():

    video_path = "data/raw/clip.mp4"

    loader = VideoLoader(video_path)
    tracker = SwimmerTracker()

    xs = []
    ts = []

    for frame_id, timestamp, frame in loader.read_frames():

        if not tracker.initialized:
            tracker.initialize(frame)

        pos = tracker.update(frame)

        if pos is None:
            continue

        x, y = pos

        xs.append(x)
        ts.append(timestamp)

    speeds = compute_speed(xs, ts)

    plot_speed(ts[1:], speeds)


if __name__ == "__main__":
    main()