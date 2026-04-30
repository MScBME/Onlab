import cv2
import numpy as np

MAX_JUMP_M = 0.4
ROLLING_WINDOW = 51
SPEED_CAP_M_S = 2.5


def _clip_position_jumps(pos: np.ndarray, max_jump_m: float) -> np.ndarray:
    clipped = np.zeros_like(pos)
    clipped[0] = pos[0]
    for i in range(1, len(pos)):
        if abs(pos[i] - clipped[i - 1]) > max_jump_m:
            clipped[i] = clipped[i - 1]
        else:
            clipped[i] = pos[i]
    return clipped


def _cap_speeds(speeds: np.ndarray, cap: float) -> np.ndarray:
    capped = np.zeros_like(speeds)
    for i, v in enumerate(speeds):
        if v > cap:
            capped[i] = capped[i - 1] if i > 0 else 0.0
        else:
            capped[i] = v
    return capped


def smooth_and_cap_speed(
    positions_m,
    timestamps,
    window_size: int = ROLLING_WINDOW,
    max_jump_m: float = MAX_JUMP_M,
    speed_cap: float = SPEED_CAP_M_S,
):
    """Outlier-suppressed speed pipeline:
    jump-clip → smooth positions → |dy/dt| → cap → smooth speeds.
    """
    if len(positions_m) < window_size:
        return np.array([]), np.array([])

    pos = np.array(positions_m, dtype=np.float64)
    ts = np.array(timestamps, dtype=np.float64)

    clipped = _clip_position_jumps(pos, max_jump_m)

    kernel = np.ones(window_size) / window_size
    smooth_pos = np.convolve(clipped, kernel, mode="valid")
    smooth_ts = ts[window_size - 1:]

    dp = np.abs(np.diff(smooth_pos))
    dt = np.diff(smooth_ts)
    dt[dt == 0] = 1e-6

    speeds = _cap_speeds(dp / dt, speed_cap)
    if len(speeds) > window_size:
        speeds = np.convolve(speeds, kernel, mode="same")

    return smooth_ts[1:], speeds


def compute_speed(xs, ys, timestamps, homography, window_size: int = ROLLING_WINDOW):
    """Legacy CSRT entrypoint: pixel coords + homography → meters → smooth_and_cap_speed."""
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    points_pixel = np.stack((xs, ys), axis=-1).reshape(-1, 1, 2)
    points_real = cv2.perspectiveTransform(points_pixel, homography)
    real_xs = points_real[:, 0, 0]
    return smooth_and_cap_speed(real_xs, timestamps, window_size=window_size)
