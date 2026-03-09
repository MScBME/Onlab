import numpy as np

def compute_speed(xs, ys, timestamps):
    xs = np.array(xs)
    ys = np.array(ys)
    ts = np.array(timestamps)

    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.diff(ts)

    distances = np.sqrt(dx**2 + dy**2)
    raw_speeds = distances / dt

    window_size = 15 
    kernel = np.ones(window_size) / window_size
    smoothed_speeds = np.convolve(raw_speeds, kernel, mode='valid')

    trim_edge = window_size - 1
    smoothed_ts = ts[1:][trim_edge:] 

    return smoothed_ts, smoothed_speeds