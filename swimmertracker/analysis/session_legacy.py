"""Legacy swimmer detection and tracking logic (warp lane before detection).

Each frame is warped into bird's-eye lane coordinates (256x1024), the detector runs on
the warped strip, and a 1D Kalman filter smooths position and speed along the lane.
"""
from typing import Callable, Optional

import cv2
import numpy as np

from swimmertracker.analysis.speed import smooth_and_cap_speed
from swimmertracker.detection.types import highest_confidence
from swimmertracker.tracking.kalman_filter import SwimmerKalmanFilter1D
from swimmertracker.video.lane_warp import DST_PTS, LANE_H, LANE_W, lane_homography, warp_lane


class LegacyAnalysisSession:
    """Runs legacy detection on the warped lane strip."""

    def __init__(self, detector, lane_coords, lane_length_m: float, confidence: float, start_sec: float,
                 run_id: int = 0, select: Optional[Callable] = None):
        self.run_id = run_id
        self.detector = detector
        self.detector.confidence = confidence
        self.select = select or highest_confidence
        self.lane_length_m = float(lane_length_m)
        self.start_sec = start_sec
        self.matrix = lane_homography(lane_coords)
        self.matrix_inv = cv2.getPerspectiveTransform(DST_PTS, np.array(lane_coords, dtype=np.float32))

        self.kf: Optional[SwimmerKalmanFilter1D] = None
        self.frames_processed = 0
        self.last_timestamp = start_sec
        self.raw_positions = []
        self.raw_timestamps = []

    def _to_original(self, x: float, y: float) -> tuple:
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), self.matrix_inv)[0][0]
        return float(pt[0]), float(pt[1])

    def process(self, frame: np.ndarray, timestamp: float):
        from swimmertracker.analysis.session import DetectionStatus, FrameResult

        warped = warp_lane(frame, self.matrix)
        detections = self.detector.detect(warped)
        detection = self.select(detections)
        result = FrameResult(timestamp=timestamp, frame=frame, warped=warped, status=DetectionStatus.SEARCHING,
                             run_id=self.run_id)

        if detection is not None:
            cx, cy = detection.center
            pos_m = cy * (self.lane_length_m / LANE_H)
            if self.kf is None:
                self.kf = SwimmerKalmanFilter1D(pos_m, timestamp)
            else:
                self.kf.predict(timestamp)
                self.kf.update(pos_m)
            self.raw_positions.append(pos_m)
            self.raw_timestamps.append(timestamp)
            result.status = DetectionStatus.DETECTED
            result.confidence = detection.confidence
            result.warped_bbox = tuple(int(v) for v in detection.bbox)
            corners = np.array([[[detection.x1, detection.y1], [detection.x2, detection.y1],
                                 [detection.x2, detection.y2], [detection.x1, detection.y2]]], dtype=np.float32)
            orig_pts = cv2.perspectiveTransform(corners, self.matrix_inv)[0]
            result.bbox = (int(round(orig_pts[:, 0].min())), int(round(orig_pts[:, 1].min())),
                           int(round(orig_pts[:, 0].max())), int(round(orig_pts[:, 1].max())))
            result.point = self._to_original(cx, cy)
            result.position_m = pos_m
        elif self.kf is not None:
            pos_m, _ = self.kf.predict(timestamp)
            pred_cy = float(np.clip(pos_m * (LANE_H / self.lane_length_m), 0, LANE_H))
            result.status = DetectionStatus.PREDICTED
            result.point = self._to_original(LANE_W / 2, pred_cy)
            result.position_m = pos_m

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
