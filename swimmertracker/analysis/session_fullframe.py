"""New full-frame swimmer detection and tracking logic (cut lanes after detection).

Each full video frame is passed to the detector, detections overlapping the target lane
polygon are ranked by lane overlap, and a 1D Kalman filter smooths position and speed.
Supports frame skip (running detector every N frames once locked).
"""
from typing import Callable, Optional

import cv2
import numpy as np

from swimmertracker.analysis.speed import smooth_and_cap_speed
from swimmertracker.tracking.kalman_filter import SwimmerKalmanFilter1D
from swimmertracker.video.lane_warp import DST_PTS, LANE_H, LANE_W, lane_homography, warp_lane


def _transform_point(matrix: np.ndarray, x: float, y: float) -> tuple:
    pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), matrix)[0][0]
    return float(pt[0]), float(pt[1])


def lane_overlap(detection, lane_hull: np.ndarray) -> float:
    """Fraction of ``detection``'s bounding box area inside ``lane_hull`` (0.0 .. 1.0)."""
    area = max(0.0, detection.x2 - detection.x1) * max(0.0, detection.y2 - detection.y1)
    if area <= 0.0:
        return 0.0
    box = np.array(
        [[detection.x1, detection.y1], [detection.x2, detection.y1],
         [detection.x2, detection.y2], [detection.x1, detection.y2]],
        dtype=np.float32,
    )
    inter, _ = cv2.intersectConvexConvex(box, lane_hull)
    return float(min(1.0, inter / area))


def select_most_in_lane(detections: list, lane_hull: np.ndarray):
    """Pick the detection that belongs to ``lane_hull``.

    Prefers detections whose center is inside the lane polygon over neighboring swimmers
    whose limbs merely extend across the lane rope. Breaks ties by lane overlap area, then confidence.
    """
    best = None
    best_key = (False, 0.0, -1.0)
    for det in detections:
        overlap = lane_overlap(det, lane_hull)
        if overlap <= 0.0:
            continue
        center_inside = cv2.pointPolygonTest(lane_hull, (float(det.center[0]), float(det.center[1])), False) >= 0
        if not center_inside and overlap < 0.25:
            continue
        key = (center_inside, overlap, det.confidence)
        if key > best_key:
            best = det
            best_key = key
    return best


class FullFrameAnalysisSession:
    """Runs full-frame detection and tracks the swimmer overlapping the target lane."""

    def __init__(self, detector, lane_coords, lane_length_m: float, confidence: float, start_sec: float,
                 run_id: int = 0, select: Optional[Callable] = None, frame_skip: int = 0):
        self.run_id = run_id
        self.detector = detector
        self.detector.confidence = confidence
        self.select = select
        self.lane_length_m = float(lane_length_m)
        self.start_sec = start_sec
        self.frame_skip = max(0, int(frame_skip))
        self._frames_since_detect = 0
        self._active_track = False
        self._last_orig_half = None
        self._last_warped_half = None
        self._last_wx = None
        self._last_conf = None
        self.matrix = lane_homography(lane_coords)
        self.matrix_inv = cv2.getPerspectiveTransform(DST_PTS, np.array(lane_coords, dtype=np.float32))
        self._lane_hull = cv2.convexHull(np.array(lane_coords, dtype=np.float32))

        self.kf: Optional[SwimmerKalmanFilter1D] = None
        self.frames_processed = 0
        self.last_timestamp = start_sec
        self.raw_positions = []
        self.raw_timestamps = []

    def _to_original(self, x: float, y: float) -> tuple:
        return _transform_point(self.matrix_inv, x, y)

    def _to_warped(self, x: float, y: float) -> tuple:
        return _transform_point(self.matrix, x, y)

    def process(self, frame: np.ndarray, timestamp: float):
        from swimmertracker.analysis.session import DetectionStatus, FrameResult

        warped = warp_lane(frame, self.matrix)
        result = FrameResult(timestamp=timestamp, frame=frame, warped=warped, status=DetectionStatus.SEARCHING,
                             run_id=self.run_id)

        # Run detector when searching, when not currently locked onto a detection, or when the skip interval elapsed
        should_detect = (self.frame_skip <= 0 or self.kf is None or not self._active_track
                         or self._frames_since_detect >= self.frame_skip)
        detection = None
        if should_detect:
            detections = self.detector.detect(frame)
            if self.select is None:
                detection = select_most_in_lane(detections, self._lane_hull)
            else:
                detection = self.select([d for d in detections if lane_overlap(d, self._lane_hull) > 0.0])

        if detection is not None:
            self._frames_since_detect = 0
            self._active_track = True
            cx, cy = detection.center
            wx, wy = self._to_warped(cx, cy)
            pos_m = wy * (self.lane_length_m / LANE_H)
            if self.kf is None:
                self.kf = SwimmerKalmanFilter1D(pos_m, timestamp)
            else:
                self.kf.predict(timestamp)
                self.kf.update(pos_m)
            self.raw_positions.append(pos_m)
            self.raw_timestamps.append(timestamp)
            result.status = DetectionStatus.DETECTED
            result.confidence = detection.confidence

            raw_hw = (detection.x2 - detection.x1) / 2.0
            raw_hh = (detection.y2 - detection.y1) / 2.0
            corners = np.array([[[detection.x1, detection.y1], [detection.x2, detection.y1],
                                 [detection.x2, detection.y2], [detection.x1, detection.y2]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(corners, self.matrix)[0]
            wx1 = float(np.clip(projected[:, 0].min(), 0, LANE_W))
            wy1 = float(np.clip(projected[:, 1].min(), 0, LANE_H))
            wx2 = float(np.clip(projected[:, 0].max(), 0, LANE_W))
            wy2 = float(np.clip(projected[:, 1].max(), 0, LANE_H))
            raw_whw = max(5.0, (wx2 - wx1) / 2.0)
            raw_whh = max(5.0, (wy2 - wy1) / 2.0)

            if self._last_orig_half is None:
                self._last_orig_half = (raw_hw, raw_hh)
                self._last_warped_half = (raw_whw, raw_whh)
                self._last_wx = wx
            else:
                self._last_orig_half = (0.7 * self._last_orig_half[0] + 0.3 * raw_hw,
                                        0.7 * self._last_orig_half[1] + 0.3 * raw_hh)
                self._last_warped_half = (0.7 * self._last_warped_half[0] + 0.3 * raw_whw,
                                          0.7 * self._last_warped_half[1] + 0.3 * raw_whh)
                self._last_wx = 0.7 * self._last_wx + 0.3 * wx

            self._last_conf = detection.confidence

            if self.frame_skip > 0:
                kf_pos = self.kf.get_position()
                pred_cy = float(np.clip(kf_pos * (LANE_H / self.lane_length_m), 0, LANE_H))
                pred_cx, pred_cy_orig = self._to_original(self._last_wx, pred_cy)
                result.point = (pred_cx, pred_cy_orig)
                result.position_m = kf_pos
                ohw, ohh = self._last_orig_half
                bx1 = float(np.clip(pred_cx - ohw, 0, frame.shape[1]))
                by1 = float(np.clip(pred_cy_orig - ohh, 0, frame.shape[0]))
                bx2 = float(np.clip(pred_cx + ohw, 0, frame.shape[1]))
                by2 = float(np.clip(pred_cy_orig + ohh, 0, frame.shape[0]))
                result.bbox = (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2)))
                whw, whh = self._last_warped_half
                wx1 = float(np.clip(self._last_wx - whw, 0, LANE_W))
                wy1 = float(np.clip(pred_cy - whh, 0, LANE_H))
                wx2 = float(np.clip(self._last_wx + whw, 0, LANE_W))
                wy2 = float(np.clip(pred_cy + whh, 0, LANE_H))
                result.warped_bbox = (int(round(wx1)), int(round(wy1)), int(round(wx2)), int(round(wy2)))
            else:
                result.point = (float(cx), float(cy))
                result.position_m = pos_m
                result.bbox = tuple(int(v) for v in detection.bbox)
                result.warped_bbox = (int(round(wx1)), int(round(wy1)), int(round(wx2)), int(round(wy2)))
        elif self.kf is not None:
            if should_detect:
                # Detector ran and missed
                self._active_track = False
                self._frames_since_detect = 0
                result.status = DetectionStatus.PREDICTED
                result.confidence = None
            else:
                # Intentionally skipped frame while actively tracking
                self._frames_since_detect += 1
                result.status = DetectionStatus.DETECTED
                result.confidence = self._last_conf

            pos_m, _ = self.kf.predict(timestamp)
            pred_cy = float(np.clip(pos_m * (LANE_H / self.lane_length_m), 0, LANE_H))
            target_wx = self._last_wx if self._last_wx is not None else (LANE_W / 2.0)
            pred_cx, pred_cy_orig = self._to_original(target_wx, pred_cy)
            result.point = (pred_cx, pred_cy_orig)
            result.position_m = pos_m

            if self._last_orig_half is not None:
                ohw, ohh = self._last_orig_half
                bx1 = float(np.clip(pred_cx - ohw, 0, frame.shape[1]))
                by1 = float(np.clip(pred_cy_orig - ohh, 0, frame.shape[0]))
                bx2 = float(np.clip(pred_cx + ohw, 0, frame.shape[1]))
                by2 = float(np.clip(pred_cy_orig + ohh, 0, frame.shape[0]))
                result.bbox = (int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2)))

            if self._last_warped_half is not None:
                whw, whh = self._last_warped_half
                wx1 = float(np.clip(target_wx - whw, 0, LANE_W))
                wy1 = float(np.clip(pred_cy - whh, 0, LANE_H))
                wx2 = float(np.clip(target_wx + whw, 0, LANE_W))
                wy2 = float(np.clip(pred_cy + whh, 0, LANE_H))
                result.warped_bbox = (int(round(wx1)), int(round(wy1)), int(round(wx2)), int(round(wy2)))

        result.speed_m_s = self.kf.get_speed() if self.kf is not None else 0.0
        self.frames_processed += 1
        self.last_timestamp = timestamp
        return result

    def summary(self, completed: bool):
        from swimmertracker.analysis.session import AnalysisSummary

        summary = AnalysisSummary(
            start_sec=self.start_sec,
            end_sec=self.last_timestamp,
            completed=completed,
            frames_processed=self.frames_processed,
            frames_detected=len(self.raw_positions),
            run_id=self.run_id,
        )
        ts, speeds = smooth_and_cap_speed(self.raw_positions, self.raw_timestamps)
        if len(speeds):
            summary.avg_speed_m_s = float(np.mean(speeds))
            summary.max_speed_m_s = float(np.max(speeds))
            summary.smoothed_times = [float(t - self.start_sec) for t in ts]
            summary.smoothed_speeds = [float(s) for s in speeds]
        return summary
