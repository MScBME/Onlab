"""Per-frame swimmer analysis of one lane: warp -> detect -> Kalman -> position/speed.

UI-independent: produces plain data (FrameResult / AnalysisSummary); drawing is up to the caller.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.analysis.speed import smooth_and_cap_speed
from src.tracking.kalman_filter import SwimmerKalmanFilter1D
from src.video.lane_warp import DST_PTS, LANE_H, LANE_W, lane_homography, warp_lane


class DetectionStatus(Enum):
    DETECTED = "detected"
    PREDICTED = "predicted"  # no detection this frame, Kalman prediction (e.g. underwater)
    SEARCHING = "searching"  # nothing detected yet


@dataclass(frozen=True)
class AnalysisParams:
    video_path: Path
    lane_id: int
    lane_coords: tuple
    lane_length_m: float
    start_sec: float
    end_sec: float
    model_path: Path
    confidence: float
    run_id: int = 0  # lets consumers drop results that belong to an earlier run


@dataclass
class FrameResult:
    timestamp: float
    frame: np.ndarray
    warped: Optional[np.ndarray] = None
    status: Optional[DetectionStatus] = None  # None in preview (no analysis running)
    confidence: Optional[float] = None
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2) in warped-lane pixels
    point: Optional[tuple] = None  # swimmer point (detected or predicted) in original-frame pixels
    position_m: Optional[float] = None
    speed_m_s: Optional[float] = None
    fps: float = 0.0
    run_id: Optional[int] = None  # None in preview


@dataclass
class AnalysisSummary:
    start_sec: float
    end_sec: float  # timestamp of the last processed frame
    completed: bool  # reached the end of the range (False: stopped early)
    frames_processed: int
    frames_detected: int
    avg_speed_m_s: Optional[float] = None
    max_speed_m_s: Optional[float] = None
    smoothed_times: list = field(default_factory=list)  # seconds relative to start_sec
    smoothed_speeds: list = field(default_factory=list)
    run_id: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def detection_rate(self) -> float:
        return self.frames_detected / self.frames_processed if self.frames_processed else 0.0


def lane_preview(frame: np.ndarray, lane_coords) -> np.ndarray:
    return warp_lane(frame, lane_homography(lane_coords))


class AnalysisSession:
    def __init__(self, detector, lane_coords, lane_length_m: float, confidence: float, start_sec: float,
                 run_id: int = 0):
        self.run_id = run_id
        self.detector = detector
        self.detector.confidence = confidence
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

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        warped = warp_lane(frame, self.matrix)
        detection = self.detector.detect(warped)
        result = FrameResult(timestamp=timestamp, frame=frame, warped=warped, status=DetectionStatus.SEARCHING,
                             run_id=self.run_id)

        if detection:
            (cx, cy), bbox, conf = detection
            pos_m = cy * (self.lane_length_m / LANE_H)
            if self.kf is None:
                self.kf = SwimmerKalmanFilter1D(pos_m, timestamp)
            else:
                self.kf.predict(timestamp)
                self.kf.update(pos_m)
            self.raw_positions.append(pos_m)
            self.raw_timestamps.append(timestamp)
            result.status = DetectionStatus.DETECTED
            result.confidence = conf
            result.bbox = bbox
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

    def summary(self, completed: bool) -> AnalysisSummary:
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
