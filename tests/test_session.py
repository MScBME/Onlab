import numpy as np
import pytest

from swimmertracker.analysis.session import AnalysisSession, DetectionStatus
from swimmertracker.detection.types import Detection
from swimmertracker.video.lane_warp import LANE_H

LANE = [[0, 774], [90, 724], [1487, 713], [1548, 761]]


def box(cy, conf=0.9):
    return Detection(100.0, cy - 20, 156.0, cy + 20, conf)


class ScriptedDetector:
    """Returns detections whose centre y moves along the lane. None means no detection."""

    def __init__(self, ys):
        self.ys = list(ys)
        self.confidence = None

    def detect(self, _warped):
        cy = self.ys.pop(0)
        return [] if cy is None else [box(cy)]


def test_statuses_positions_and_summary():
    fps = 30.0
    # swimmer moves 2 px/frame in warped space, disappears for 3 frames, reappears
    ys = [100 + 2 * i for i in range(30)] + [None] * 3 + [166 + 2 * i for i in range(30)]
    detector = ScriptedDetector(ys)
    session = AnalysisSession(detector, LANE, lane_length_m=25.0, confidence=0.35, start_sec=10.0)
    frame = np.zeros((1080, 1920, 3), np.uint8)

    results = [session.process(frame, 10.0 + i / fps) for i in range(len(ys))]

    assert detector.confidence == 0.35
    assert results[0].status == DetectionStatus.DETECTED
    assert results[0].position_m == pytest.approx(100 * 25.0 / LANE_H)
    assert results[0].bbox == (100, 80, 156, 120)
    assert results[0].warped.shape == (LANE_H, 256, 3)
    assert [r.status for r in results[30:33]] == [DetectionStatus.PREDICTED] * 3
    assert all(r.point is not None for r in results)

    summary = session.summary(completed=True)
    assert summary.frames_processed == 63 and summary.frames_detected == 60
    assert summary.detection_rate == pytest.approx(60 / 63)
    true_speed = 2 * fps * 25.0 / LANE_H  # ~1.46 m/s
    assert summary.max_speed_m_s == pytest.approx(true_speed, rel=0.15)
    assert summary.smoothed_times[0] == pytest.approx(0.0)


def test_searching_before_first_detection_and_short_runs():
    session = AnalysisSession(ScriptedDetector([None, None]), LANE, 25.0, 0.4, start_sec=0.0)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    result = session.process(frame, 0.0)
    assert result.status == DetectionStatus.SEARCHING and result.position_m is None
    session.process(frame, 0.033)
    summary = session.summary(completed=False)
    assert summary.avg_speed_m_s is None and summary.frames_detected == 0


class ManyDetector:
    confidence = None

    def detect(self, _warped):
        return [box(300, 0.6), box(500, 0.95), box(700, 0.7)]


def test_default_selection_is_the_most_confident_detection():
    session = AnalysisSession(ManyDetector(), LANE, 25.0, 0.4, start_sec=0.0)
    result = session.process(np.zeros((1080, 1920, 3), np.uint8), 0.0)
    assert result.confidence == 0.95
    assert result.position_m == pytest.approx(500 * 25.0 / LANE_H)


def test_selection_strategy_is_replaceable():
    def lowest_in_lane(detections):
        return min(detections, key=lambda d: d.center[1], default=None)

    session = AnalysisSession(ManyDetector(), LANE, 25.0, 0.4, start_sec=0.0, select=lowest_in_lane)
    result = session.process(np.zeros((1080, 1920, 3), np.uint8), 0.0)
    assert result.confidence == 0.6
    assert result.position_m == pytest.approx(300 * 25.0 / LANE_H)
