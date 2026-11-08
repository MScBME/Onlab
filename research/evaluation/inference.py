"""The eval-inference command runs a model on a clip with the app's own analysis pipeline, AnalysisSession.

A .onnx model runs on the app's OnnxDetector, a .pt model on ultralytics.
"""
import argparse

import cv2
import numpy as np

from research.evaluation.clips import clip_from_args, load_run_config, resolve_path
from research.evaluation.detectors import make_detector
from research.evaluation.plots import plot_speed
from swimmertracker.analysis.session import AnalysisSession, DetectionStatus
from swimmertracker.video.loader import VideoLoader

GREEN, ORANGE, BLUE, RED = (0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 0, 255)


def show(result, lane_coords) -> bool:
    """Draw one analysed frame and return True when Esc was pressed."""
    frame, warped = result.frame.copy(), result.warped.copy()
    if result.bbox:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(warped, (x1, y1), (x2, y2), GREEN, 2)
        cv2.putText(warped, f"{result.confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
    if result.point:
        color = GREEN if result.status == DetectionStatus.DETECTED else ORANGE
        cv2.circle(frame, (int(result.point[0]), int(result.point[1])), 5, color, -1)
    if result.status != DetectionStatus.DETECTED:
        cv2.putText(frame, "No detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
    pts = np.array(lane_coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=BLUE, thickness=2)
    cv2.imshow("Original", frame)
    cv2.imshow("Lane bird's-eye view", warped)
    return cv2.waitKey(1) & 0xFF == 27


def main():
    parser = argparse.ArgumentParser(prog="eval-inference",
                                     description="Run a model on a clip with the app's analysis pipeline.")
    parser.add_argument("--clip", help="Clip id, default active_clip_id of research/run_config.json")
    parser.add_argument("--lane", type=int, help="Lane id to use instead of the clip's lane")
    parser.add_argument("--model", help="Model .onnx or .pt file, default model_path of research/run_config.json")
    parser.add_argument("--confidence", type=float,
                        help="Confidence threshold, default confidence_threshold of research/run_config.json")
    parser.add_argument("--no-show", action="store_true", help="No tracking windows and no speed plot")
    args = parser.parse_args()

    config = load_run_config()
    ctx = clip_from_args(args, config)
    model_path = resolve_path(args.model or config["model_path"])
    confidence = args.confidence if args.confidence is not None else config.get("confidence_threshold", 0.5)
    show_tracking = config.get("show_tracking", True) and not args.no_show

    clip = ctx.clip
    print(f"Clip {clip.id} | video {ctx.video.id} in the {ctx.role} library | lane {ctx.lane.id} | "
          f"model {model_path.name}")
    detector = make_detector(model_path, confidence)
    session = AnalysisSession(detector, ctx.lane.coordinates, ctx.lane.length_m, confidence, clip.start_sec)
    loader = VideoLoader(str(ctx.video_path), start_sec=clip.start_sec, end_sec=clip.end_sec)

    completed = True
    print("Running detection, press Esc to stop..." if show_tracking else "Running detection...")
    for _frame_id, timestamp, frame in loader.read_frames():
        result = session.process(frame, timestamp)
        if show_tracking and show(result, ctx.lane.coordinates):
            completed = False
            break
    loader.release()
    cv2.destroyAllWindows()

    summary = session.summary(completed)
    print(f"Swimmer detected in {summary.frames_detected} of {summary.frames_processed} frames, "
          f"{summary.detection_rate:.0%}")
    if summary.avg_speed_m_s is None:
        print("Not enough detections to compute speed.")
        return
    print(f"Average speed {summary.avg_speed_m_s:.2f} m/s, max {summary.max_speed_m_s:.2f} m/s")
    if not args.no_show:
        plot_speed(summary.smoothed_times, summary.smoothed_speeds)


if __name__ == "__main__":
    main()
