"""Lane geometry: corner-order convention and the camera-view -> bird's-eye warp.

Corner convention (matches every lane in data/videos.json and the training data):
clockwise on the image, the first edge (p0 -> p1) is one end of the lane (a short side),
and the lane end with the smaller x (then smaller y) comes first. DST_PTS maps that first
end to y = 0, so the lane length runs along the Y axis of the warped image.
"""
import cv2
import numpy as np

LANE_W = 256
LANE_H = 1024
DST_PTS = np.array([[0, 0], [LANE_W, 0], [LANE_W, LANE_H], [0, LANE_H]], dtype="float32")

MIN_LANE_AREA_PX = 100.0


def lane_homography(coords) -> np.ndarray:
    return cv2.getPerspectiveTransform(np.array(coords, dtype="float32"), DST_PTS)


def warp_lane(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame, matrix, (LANE_W, LANE_H))


def _signed_area2(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def normalize_corners(points) -> list:
    """Reorder 4 lane corners (given in any order) into the project's corner convention."""
    pts = np.array(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("A lane needs exactly 4 corners")

    # Cyclic order around the centroid, then clockwise on the image (positive shoelace with y down).
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(angles, kind="stable")]
    if _signed_area2(pts) < 0:
        pts = pts[::-1]

    edge_len = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    end_edges = (0, 2) if edge_len[0] + edge_len[2] <= edge_len[1] + edge_len[3] else (1, 3)

    def end_key(i):
        mid = (pts[i] + pts[(i + 1) % 4]) / 2
        return (mid[0], mid[1])

    start = min(end_edges, key=end_key)
    pts = np.roll(pts, -start, axis=0)
    return [[_as_number(x), _as_number(y)] for x, y in pts]


def _as_number(value: float):
    return int(value) if float(value).is_integer() else float(value)


def validate_corners(points, frame_size=None) -> list:
    """Return human-readable problems with a (normalized) lane quad; empty list means valid."""
    pts = np.array(points, dtype=np.float64)
    if pts.shape != (4, 2):
        return ["A lane needs exactly 4 corners."]

    errors = []
    if frame_size is not None:
        width, height = frame_size
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] > width) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] > height):
            errors.append("All corners must lie inside the video frame.")

    crosses = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        errors.append("Lane corners must form a convex quadrilateral.")
    elif abs(_signed_area2(pts)) / 2 < MIN_LANE_AREA_PX:
        errors.append("The lane area is too small.")
    return errors
