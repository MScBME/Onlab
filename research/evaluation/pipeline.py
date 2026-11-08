"""The eval-pipeline command compares the speed signal processing levels on a clip.

The levels are the raw derivative, a moving average, the Kalman filter and the final smoothed profile. The detections
come from the app's analysis pipeline, AnalysisSession, so this shows what the app's chart smooths.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from research.evaluation.clips import clip_from_args, load_run_config, resolve_path
from research.evaluation.detectors import make_detector
from research.paths import RESEARCH_DIR
from swimmertracker.analysis.session import AnalysisSession
from swimmertracker.analysis.speed import SPEED_SMOOTH_WINDOW, filter_positions_kalman, smooth_and_cap_speed
from swimmertracker.video.loader import VideoLoader

OUTPUT_DIR = RESEARCH_DIR / "evaluation" / "output"  # gitignored


def main():
    parser = argparse.ArgumentParser(prog="eval-pipeline",
                                     description="Compare the speed signal processing levels on a clip.")
    parser.add_argument("--clip", help="Clip id, default active_clip_id of research/run_config.json")
    parser.add_argument("--model", help="Model .onnx or .pt file, default model_path of research/run_config.json")
    parser.add_argument("--confidence", type=float,
                        help="Confidence threshold, default confidence_threshold of research/run_config.json")
    args = parser.parse_args()

    config = load_run_config()
    ctx = clip_from_args(args, config)
    model_path = resolve_path(args.model or config["model_path"])
    confidence = args.confidence if args.confidence is not None else config.get("confidence_threshold", 0.5)
    clip = ctx.clip

    print(f"Running the analysis, model {model_path.name}, clip {clip.id}...")
    session = AnalysisSession(make_detector(model_path, confidence), ctx.lane.coordinates, ctx.lane.length_m,
                              confidence, clip.start_sec)
    loader = VideoLoader(str(ctx.video_path), start_sec=clip.start_sec, end_sec=clip.end_sec)
    for _, ts, frame in loader.read_frames():
        session.process(frame, ts)
    loader.release()

    raw_positions, timestamps = session.raw_positions, session.raw_timestamps
    if len(raw_positions) < 15:
        print("Not enough detections for the analysis.")
        return

    # 1. Raw speed, the absolute value of the derivative of the position, noisy
    raw_pos = np.array(raw_positions)
    raw_ts = np.array(timestamps)
    dt = np.diff(raw_ts)
    dt[dt == 0] = 1e-6
    raw_v = np.abs(np.diff(raw_pos) / dt)
    raw_v_ts = raw_ts[1:]

    # 2. Classic moving average of the raw speed over the same window as the app's smoothing
    moving_avg_v = np.convolve(raw_v, np.ones(SPEED_SMOOTH_WINDOW) / SPEED_SMOOTH_WINDOW, mode="same")

    # 3. Kalman filtered speed, then the final smoothed profile of the app
    kalman_ts, _, kalman_v = filter_positions_kalman(raw_positions, timestamps)
    final_ts, final_v = smooth_and_cap_speed(raw_positions, timestamps)

    # Plot
    plt.figure(figsize=(15, 8))

    t0 = raw_ts[0]
    plt.plot(raw_v_ts - t0, raw_v, label="1. Raw derivative, noisy", color="red", alpha=0.3, linewidth=0.8)
    plt.plot(raw_v_ts - t0, moving_avg_v, label="2. Simple moving average, with phase lag and distortion",
             color="orange", linestyle="--", linewidth=1.8)
    plt.plot(kalman_ts - t0, kalman_v, label="3. Kalman filtered speed", color="blue", linewidth=2.0)
    plt.plot(final_ts - t0, final_v, label="4. Final smoothed profile of the strokes", color="green",
             linewidth=3.0)

    plt.title(f"Comparison of speed estimation methods\nModel {model_path.name} | Clip {clip.id}", fontsize=13)
    plt.xlabel("Time [s]", fontsize=11)
    plt.ylabel("Speed [m/s]", fontsize=11)
    plt.ylim(0, 3.5)

    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"pipeline_analysis_{clip.id}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
