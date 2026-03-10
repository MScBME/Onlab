import numpy as np
import cv2

def compute_speed(xs, ys, timestamps, H):
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    ts = np.array(timestamps)

    points_pixel = np.stack((xs, ys), axis=-1).reshape(-1, 1, 2)
    points_real = cv2.perspectiveTransform(points_pixel, H)

    real_xs = points_real[:, 0, 0]

    dx = np.diff(real_xs)
    dt = np.diff(ts)

    distances = np.abs(dx)
    raw_speeds = distances / dt

    window_size = 15 
    kernel = np.ones(window_size) / window_size
    smoothed_speeds = np.convolve(raw_speeds, kernel, mode='valid')

    trim_edge = window_size - 1
    smoothed_ts = ts[1:][trim_edge:] 

    return smoothed_ts, smoothed_speeds