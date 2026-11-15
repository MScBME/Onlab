"""Swimmer analysis session interface and routing.

Common dataclasses (AnalysisParams, FrameResult, AnalysisSummary) and helper functions.
Routes AnalysisSession instantiation to either LegacyAnalysisSession or FullFrameAnalysisSession
via a single conditional check.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from swimmertracker.analysis.session_fullframe import (
    FullFrameAnalysisSession,
    lane_overlap,
    select_most_in_lane,
)
from swimmertracker.analysis.session_legacy import LegacyAnalysisSession
from swimmertracker.video.lane_warp import lane_homography, warp_lane


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
    legacy: bool = False  # True: legacy warp-then-detect; False: new full-frame detection


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


def AnalysisSession(detector, lane_coords, lane_length_m: float, confidence: float, start_sec: float,
                    run_id: int = 0, select: Optional[Callable] = None, frame_skip: int = 0,
                    legacy: bool = False):
    """Factory creating the appropriate AnalysisSession.

    Single if condition:
    - legacy=True: runs LegacyAnalysisSession (warp before detect)
    - legacy=False: runs FullFrameAnalysisSession (detect on full frame, cut after)
    """
    if legacy:
        return LegacyAnalysisSession(detector, lane_coords, lane_length_m, confidence, start_sec,
                                     run_id=run_id, select=select)
    return FullFrameAnalysisSession(detector, lane_coords, lane_length_m, confidence, start_sec,
                                    run_id=run_id, select=select, frame_skip=frame_skip)
