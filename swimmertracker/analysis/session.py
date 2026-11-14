"""Per frame swimmer analysis of one lane.

Each full frame is passed to the detector, detections overlapping the target lane are filtered and ranked by lane
overlap (then confidence), and a Kalman filter smooths position and speed along the lane.

The module is independent of the UI. It produces plain data in FrameResult and AnalysisSummary, drawing is up to
the caller.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from swimmertracker.analysis.speed import smooth_and_cap_speed
from swimmertracker.tracking.kalman_filter import SwimmerKalmanFilter1D
from swimmertracker.video.lane_warp import DST_PTS, LANE_H, LANE_W, lane_homography, warp_lane


class DetectionStatus(Enum):
    DETECTED = "detected"
    PREDICTED = "predicted"  # no detection in this frame, Kalman prediction, for example underwater
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
    frame_skip: int = 0  # number of frames to skip between detections (0 = every frame)


@dataclass
class FrameResult:
    timestamp: float
    frame: np.ndarray
    warped: Optional[np.ndarray] = None
    status: Optional[DetectionStatus] = None  # None in preview, when no analysis is running
    confidence: Optional[float] = None
    bbox: Optional[tuple] = None  # x1, y1, x2, y2 in original frame pixels
    warped_bbox: Optional[tuple] = None  # x1, y1, x2, y2 in warped lane pixels
    point: Optional[tuple] = None  # detected or predicted swimmer point in original frame pixels
    position_m: Optional[float] = None
    speed_m_s: Optional[float] = None
    fps: float = 0.0
    run_id: Optional[int] = None  # None in preview


@dataclass
class AnalysisSummary:
    start_sec: float
    end_sec: float  # timestamp of the last processed frame
    completed: bool  # False if the run stopped before the end of the range
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
    """Pick the detection that overlaps ``lane_hull`` the most, breaking ties by confidence."""
    best = None
    best_key = (0.0, -1.0)
    for det in detections:
        overlap = lane_overlap(det, lane_hull)
        if overlap > 0.0 and (overlap, det.confidence) > best_key:
            best = det
            best_key = (overlap, det.confidence)
    return best


class AnalysisSession:
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
        self._last_warped_half = None
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

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        warped = warp_lane(frame, self.matrix)
        result = FrameResult(timestamp=timestamp, frame=frame, warped=warped, status=DetectionStatus.SEARCHING,
                             run_id=self.run_id)

        # Run detector when searching or when the skip interval elapsed
        should_detect = self.frame_skip <= 0 or self.kf is None or self._frames_since_detect >= self.frame_skip
        detection = None
        if should_detect:
            detections = self.detector.detect(frame)
            if self.select is None:
                detection = select_most_in_lane(detections, self._lane_hull)
            else:
                detection = self.select([d for d in detections if lane_overlap(d, self._lane_hull) > 0.0])

        if detection is not None:
            self._frames_since_detect = 0
            cx, cy = detection.center
            _, wy = self._to_warped(cx, cy)
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
            result.bbox = tuple(int(v) for v in detection.bbox)
            corners = np.array([[[detection.x1, detection.y1], [detection.x2, detection.y1],
                                 [detection.x2, detection.y2], [detection.x1, detection.y2]]], dtype=np.float32)
            projected = cv2.perspectiveTransform(corners, self.matrix)[0]
            wx1 = float(np.clip(projected[:, 0].min(), 0, LANE_W))
            wy1 = float(np.clip(projected[:, 1].min(), 0, LANE_H))
            wx2 = float(np.clip(projected[:, 0].max(), 0, LANE_W))
            wy2 = float(np.clip(projected[:, 1].max(), 0, LANE_H))
            self._last_warped_half = (max(5.0, (wx2 - wx1) / 2), max(5.0, (wy2 - wy1) / 2))
            result.warped_bbox = (int(round(wx1)), int(round(wy1)), int(round(wx2)), int(round(wy2)))
            result.point = (float(cx), float(cy))
            result.position_m = pos_m
        elif self.kf is not None:
            if not should_detect:
                self._frames_since_detect += 1
            else:
                self._frames_since_detect = 0
            pos_m, _ = self.kf.predict(timestamp)
            pred_cy = float(np.clip(pos_m * (LANE_H / self.lane_length_m), 0, LANE_H))
            result.status = DetectionStatus.PREDICTED
            result.point = self._to_original(LANE_W / 2, pred_cy)
            result.position_m = pos_m
            if self._last_warped_half is not None:
                hw, hh = self._last_warped_half
                wx1 = float(np.clip(LANE_W / 2 - hw, 0, LANE_W))
                wy1 = float(np.clip(pred_cy - hh, 0, LANE_H))
                wx2 = float(np.clip(LANE_W / 2 + hw, 0, LANE_W))
                wy2 = float(np.clip(pred_cy + hh, 0, LANE_H))
                result.warped_bbox = (int(round(wx1)), int(round(wy1)), int(round(wx2)), int(round(wy2)))
                corners = np.array([[[wx1, wy1], [wx2, wy1], [wx2, wy2], [wx1, wy2]]], dtype=np.float32)
                orig_pts = cv2.perspectiveTransform(corners, self.matrix_inv)[0]
                result.bbox = (int(round(orig_pts[:, 0].min())), int(round(orig_pts[:, 1].min())),
                               int(round(orig_pts[:, 0].max())), int(round(orig_pts[:, 1].max())))

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
