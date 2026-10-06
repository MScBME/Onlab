import cv2
import numpy as np

from src.tracking.kalman_filter import SwimmerKalmanFilter1D

MAX_JUMP_M = 0.4
ROLLING_WINDOW = 11
SPEED_SMOOTH_WINDOW = 7
SPEED_CAP_M_S = 3.2


def filter_positions_kalman(positions_m, timestamps, process_noise_std=0.8, measurement_noise_std=0.15):
    """
    Offline Kalman-szűrés a mért pozíciók és időbélyegek listáján.
    Visszaadja a szűrt időbélyegeket, pozíciókat és a számított sebességeket.
    """
    if len(positions_m) == 0:
        return np.array([]), np.array([]), np.array([])

    pos = np.array(positions_m, dtype=np.float64)
    ts = np.array(timestamps, dtype=np.float64)

    kf = SwimmerKalmanFilter1D(
        initial_pos=pos[0],
        initial_time=ts[0],
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
    )

    filtered_positions = [pos[0]]
    filtered_speeds = [0.0]

    for i in range(1, len(pos)):
        kf.predict(ts[i])
        kf.update(pos[i])
        filtered_positions.append(kf.get_position())
        filtered_speeds.append(kf.get_speed())

    return ts, np.array(filtered_positions), np.array(filtered_speeds)


def smooth_and_cap_speed(
    positions_m,
    timestamps,
    window_size: int = ROLLING_WINDOW,
    speed_window_size: int = SPEED_SMOOTH_WINDOW,
    max_jump_m: float = MAX_JUMP_M,
    speed_cap: float = SPEED_CAP_M_S,
):
    """
    Kalman-alapú sebességszámítás kíméletes utólagos simítással.
    Megszünteti a szélső adatok elvesztését és a zaj pozitív integrálódását.
    """
    if len(positions_m) < 3:
        return np.array([]), np.array([])

    pos = np.array(positions_m, dtype=np.float64)
    ts = np.array(timestamps, dtype=np.float64)

    # 1. Kalman szűrés
    ts_out, _, speeds = filter_positions_kalman(pos, ts)

    # 2. Fizikai korlátok érvényesítése
    speeds = np.clip(speeds, 0.0, speed_cap)

    # 3. Kíméletes mozgóablakos simítás a karciklusok egyenletes megjelenítéséhez
    if speed_window_size > 1 and len(speeds) >= speed_window_size:
        kernel = np.ones(speed_window_size) / speed_window_size
        speeds = np.convolve(speeds, kernel, mode="same")

    return ts_out, speeds


def compute_speed(xs, ys, timestamps, homography, window_size: int = ROLLING_WINDOW):
    """CSRT kézi követő kompatibilitás."""
    if len(xs) == 0:
        return np.array([]), np.array([])

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    points_pixel = np.stack((xs, ys), axis=-1).reshape(-1, 1, 2)
    points_real = cv2.perspectiveTransform(points_pixel, homography)
    real_xs = points_real[:, 0, 0]
    return smooth_and_cap_speed(real_xs, timestamps, window_size=window_size)