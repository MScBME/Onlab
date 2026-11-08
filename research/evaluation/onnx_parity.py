"""The eval-onnx command checks whether the exported .onnx model behaves like the .pt model on a clip.

Both run frame by frame through their own AnalysisSession, the app's pipeline, on the same frames. The report shows
how often they agree on whether a swimmer was found, the IoU and confidence difference of the chosen boxes, the
final average and maximum speed, and the detector time per frame.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from research.evaluation.clips import clip_from_args, load_run_config, resolve_path
from research.evaluation.ultralytics_detector import UltralyticsDetector
from swimmertracker.analysis.session import AnalysisSession, DetectionStatus
from swimmertracker.detection.onnx_detector import OnnxDetector
from swimmertracker.video.lane_warp import LANE_H, LANE_W
from swimmertracker.video.loader import VideoLoader

# Thresholds within which the two models count as the same
MIN_AGREEMENT = 0.99
MIN_MEAN_IOU = 0.95
MAX_SPEED_DIFF = 0.02  # relative
SPEED_LABELS = {"avg_speed": "average speed", "max_speed": "max speed"}


class TimedDetector:
    def __init__(self, inner):
        self.inner = inner
        self.seconds = []

    @property
    def confidence(self):
        return self.inner.confidence

    @confidence.setter
    def confidence(self, value):
        self.inner.confidence = value

    def detect(self, image):
        start = time.perf_counter()
        detections = self.inner.detect(image)
        self.seconds.append(time.perf_counter() - start)
        return detections


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def relative_diff(a, b) -> float:
    if a is None or b is None:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / max(abs(a), 1e-9)


def compare(pt_path: Path, onnx_path: Path, ctx, confidence: float) -> dict:
    clip = ctx.clip
    detectors = {
        "pt": TimedDetector(UltralyticsDetector(pt_path, confidence)),
        "onnx": TimedDetector(OnnxDetector(onnx_path, confidence)),
    }
    warm = np.zeros((LANE_H, LANE_W, 3), np.uint8)
    for d in detectors.values():
        d.detect(warm)
        d.seconds.clear()
    sessions = {k: AnalysisSession(d, ctx.lane.coordinates, ctx.lane.length_m, confidence, clip.start_sec)
                for k, d in detectors.items()}

    found = {"both": 0, "pt_only": 0, "onnx_only": 0, "neither": 0}
    ious, conf_diffs = [], []
    loader = VideoLoader(str(ctx.video_path), start_sec=clip.start_sec, end_sec=clip.end_sec)
    for _, ts, frame in loader.read_frames():
        a, b = (sessions[k].process(frame, ts) for k in ("pt", "onnx"))
        da, db = a.status == DetectionStatus.DETECTED, b.status == DetectionStatus.DETECTED
        found["both" if da and db else "pt_only" if da else "onnx_only" if db else "neither"] += 1
        if da and db:
            ious.append(iou(a.bbox, b.bbox))
            conf_diffs.append(abs(a.confidence - b.confidence))
    loader.release()

    summaries = {k: s.summary(True) for k, s in sessions.items()}
    frames = sum(found.values())
    return {
        "frames": frames,
        "found": found,
        "agreement": (found["both"] + found["neither"]) / frames if frames else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "min_iou": float(np.min(ious)) if ious else None,
        "mean_conf_diff": float(np.mean(conf_diffs)) if conf_diffs else None,
        "max_conf_diff": float(np.max(conf_diffs)) if conf_diffs else None,
        "avg_speed": {k: s.avg_speed_m_s for k, s in summaries.items()},
        "max_speed": {k: s.max_speed_m_s for k, s in summaries.items()},
        "ms_per_frame": {k: 1000 * float(np.mean(d.seconds)) if d.seconds else None for k, d in detectors.items()},
    }


def verdict(r: dict) -> list:
    """The problems found. An empty list means the ONNX model can replace the .pt model."""
    problems = []
    if r["agreement"] < MIN_AGREEMENT:
        problems.append(f"detection agreement {r['agreement']:.1%} < {MIN_AGREEMENT:.0%}")
    if r["mean_iou"] is not None and r["mean_iou"] < MIN_MEAN_IOU:
        problems.append(f"mean IoU {r['mean_iou']:.3f} < {MIN_MEAN_IOU}")
    for key, label in SPEED_LABELS.items():
        diff = relative_diff(r[key]["pt"], r[key]["onnx"])
        if diff > MAX_SPEED_DIFF:
            problems.append(f"{label} differs by {diff:.1%}")
    return problems


def fmt(value, pattern="{:.3f}"):
    return "n/a" if value is None else pattern.format(value)


def main():
    parser = argparse.ArgumentParser(prog="eval-onnx", description="Compare a .pt model with its ONNX export on a clip.")
    parser.add_argument("--model", help="The .pt or .onnx model, the other one is the file with the same name next to "
                                        "it. Default model_path of research/run_config.json")
    parser.add_argument("--clip", action="append",
                        help="Clip id, repeat it for several clips. Default active_clip_id of research/run_config.json")
    parser.add_argument("--confidence", type=float,
                        help="Confidence threshold, default confidence_threshold of research/run_config.json")
    args = parser.parse_args()

    config = load_run_config()
    base = resolve_path(args.model or config["model_path"])
    pt_path, onnx_path = base.with_suffix(".pt"), base.with_suffix(".onnx")
    for path in (pt_path, onnx_path):
        if not path.exists():
            print(f"{path} does not exist. Both the .pt and the .onnx model are needed, train-model writes both.")
            sys.exit(1)
    confidence = args.confidence if args.confidence is not None else config.get("confidence_threshold", 0.5)

    failed = False
    for clip_id in args.clip or [None]:
        ctx = clip_from_args(argparse.Namespace(clip=clip_id, lane=None), config)
        print(f"\n{base.stem} on {ctx.clip.id}, lane {ctx.lane.id}, confidence {confidence}")
        r = compare(pt_path, onnx_path, ctx, confidence)
        f = r["found"]
        print(f"  {r['frames']} frames, found by both {f['both']}, only .pt {f['pt_only']}, "
              f"only .onnx {f['onnx_only']}, neither {f['neither']}, agreement {r['agreement']:.2%}")
        print(f"  box IoU mean {fmt(r['mean_iou'])} min {fmt(r['min_iou'])}, confidence difference "
              f"mean {fmt(r['mean_conf_diff'], '{:.4f}')} max {fmt(r['max_conf_diff'], '{:.4f}')}")
        for key, label in SPEED_LABELS.items():
            print(f"  {label} .pt {fmt(r[key]['pt'])} m/s, .onnx {fmt(r[key]['onnx'])} m/s")
        ms = r["ms_per_frame"]
        print(f"  detector time .pt {fmt(ms['pt'], '{:.1f}')} ms per frame, "
              f".onnx on CPU {fmt(ms['onnx'], '{:.1f}')} ms per frame")
        problems = verdict(r)
        print("  OK" if not problems else "  MISMATCH, " + ", ".join(problems))
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
