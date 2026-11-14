import cv2
import numpy as np
import pytest

from swimmertracker.analysis.session import AnalysisSession, DetectionStatus, lane_overlap
from swimmertracker.detection.types import Detection
from swimmertracker.video.lane_warp import DST_PTS, LANE_H, LANE_W

LANE = [[0, 774], [90, 724], [1487, 713], [1548, 761]]
MATRIX_INV = cv2.getPerspectiveTransform(DST_PTS, np.array(LANE, dtype=np.float32))


def box_at_warped(wy, conf=0.9, half_w=20.0, half_h=10.0):
    """Build an original-frame Detection centered at (LANE_W / 2, wy) in warped lane coordinates."""
    ox, oy = cv2.perspectiveTransform(np.array([[[LANE_W / 2, wy]]], dtype=np.float32), MATRIX_INV)[0][0]
    return Detection(float(ox - half_w), float(oy - half_h), float(ox + half_w), float(oy + half_h), conf)


class ScriptedDetector:
    """Returns detections whose centre moves along the lane. None means no detection."""

    def __init__(self, ys):
        self.ys = list(ys)
        self.confidence = None
        self.seen_shapes = []

    def detect(self, frame):
        self.seen_shapes.append(frame.shape)
        cy = self.ys.pop(0)
        return [] if cy is None else [box_at_warped(cy)]


def test_statuses_positions_and_summary():
    fps = 30.0
    # swimmer moves 2 px/frame along the warped lane axis, disappears for 3 frames, reappears
    ys = [100 + 2 * i for i in range(30)] + [None] * 3 + [166 + 2 * i for i in range(30)]
    detector = ScriptedDetector(ys)
    session = AnalysisSession(detector, LANE, lane_length_m=25.0, confidence=0.35, start_sec=10.0)
    frame = np.zeros((1080, 1920, 3), np.uint8)

    results = [session.process(frame, 10.0 + i / fps) for i in range(len(ys))]

    assert detector.confidence == 0.35
    assert detector.seen_shapes[0] == (1080, 1920, 3)
    expected_box = box_at_warped(100)
    assert results[0].status == DetectionStatus.DETECTED
    assert results[0].position_m == pytest.approx(100 * 25.0 / LANE_H, abs=1e-3)
    assert results[0].bbox == tuple(int(v) for v in expected_box.bbox)
    assert results[0].warped_bbox is not None and len(results[0].warped_bbox) == 4
    assert results[0].point == pytest.approx(expected_box.center)
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

    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, _frame):
        return self.detections


def test_default_selection_picks_most_overlapping_then_confidence():
    hull = cv2.convexHull(np.array(LANE, dtype=np.float32))
    outside = Detection(500.0, 100.0, 560.0, 140.0, 0.99)
    edge_splash = Detection(500.0, 690.0, 560.0, 740.0, 0.95)  # partially outside top lane border
    mostly_in = box_at_warped(500, conf=0.70)
    assert lane_overlap(outside, hull) == 0.0
    assert 0.0 < lane_overlap(edge_splash, hull) < lane_overlap(mostly_in, hull)

    session = AnalysisSession(ManyDetector([outside, edge_splash, mostly_in]), LANE, 25.0, 0.4, start_sec=0.0)
    result = session.process(np.zeros((1080, 1920, 3), np.uint8), 0.0)
    assert result.confidence == pytest.approx(0.70)
    assert result.position_m == pytest.approx(500 * 25.0 / LANE_H, abs=1e-3)

    # Same overlap -> higher confidence wins
    in_low = box_at_warped(300, conf=0.60, half_w=10.0, half_h=5.0)
    in_high = box_at_warped(700, conf=0.95, half_w=10.0, half_h=5.0)
    session2 = AnalysisSession(ManyDetector([outside, in_low, in_high]), LANE, 25.0, 0.4, start_sec=0.0)
    result2 = session2.process(np.zeros((1080, 1920, 3), np.uint8), 0.0)
    assert result2.confidence == pytest.approx(0.95)
    assert result2.position_m == pytest.approx(700 * 25.0 / LANE_H, abs=1e-3)


def test_ignores_detections_outside_the_lane():
    outside = Detection(200.0, 100.0, 260.0, 140.0, 0.99)
    session = AnalysisSession(ManyDetector([outside]), LANE, 25.0, 0.4, start_sec=0.0)
    result = session.process(np.zeros((1080, 1920, 3), np.uint8), 0.0)
    assert result.status == DetectionStatus.SEARCHING
    assert result.bbox is None and result.point is None


def test_selection_strategy_is_replaceable():
    def lowest_in_lane(detections):
        return min(detections, key=lambda d: d.center[1], default=None)

    outside_higher = Detection(500.0, 100.0, 560.0, 140.0, 0.99)
    in_lane_top = box_at_warped(300, conf=0.6, half_w=10.0, half_h=5.0)
    in_lane_bot = box_at_warped(700, conf=0.95, half_w=10.0, half_h=5.0)
    # In original frame image y, select the box with the smaller y coordinate.
    small_oy = min([in_lane_top, in_lane_bot], key=lambda d: d.center[1])
    session = AnalysisSession(ManyDetector([outside_higher, in_lane_top, in_lane_bot]), LANE, 25.0, 0.4,
                              start_sec=0.0, select=lowest_in_lane)
    result = session.process(np.zeros((1080, 1920, 3), np.uint8), 0.0)
    assert result.confidence == pytest.approx(small_oy.confidence)
