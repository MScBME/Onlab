import numpy as np
import cv2

MAX_JUMP_M = 0.4
ROLLING_WINDOW = 61
SPEED_CAP_M_S = 3.5

def compute_speed(xs, ys, timestamps, H, window_size=ROLLING_WINDOW):
    if len(xs) < window_size:
        return np.array([]), np.array([])

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    ts = np.array(timestamps)

    points_pixel = np.stack((xs, ys), axis=-1).reshape(-1, 1, 2)
    points_real = cv2.perspectiveTransform(points_pixel, H)
    real_xs = points_real[:, 0, 0]

    clipped_xs = np.zeros_like(real_xs)
    clipped_xs[0] = real_xs[0]
    for i in range(1, len(real_xs)):
        if abs(real_xs[i] - clipped_xs[i-1]) > MAX_JUMP_M:
            clipped_xs[i] = clipped_xs[i-1]
        else:
            clipped_xs[i] = real_xs[i]

    smooth_xs = np.convolve(clipped_xs, np.ones(window_size)/window_size, mode='valid')
    smooth_ts = ts[window_size-1:]

    dx = np.diff(smooth_xs)
    dt = np.diff(smooth_ts)
    dt[dt == 0] = 1e-6
    
    speeds = np.abs(dx / dt)
    
    final_speeds = np.zeros_like(speeds)
    for i in range(len(speeds)):
        v = speeds[i]
        if v > SPEED_CAP_M_S:
            final_speeds[i] = final_speeds[i-1] if i > 0 else SPEED_CAP_M_S
        else:
            final_speeds[i] = v

    if len(final_speeds) > window_size:
        final_speeds = np.convolve(final_speeds, np.ones(window_size)//window_size, mode='same')

    return smooth_ts[1:], final_speeds