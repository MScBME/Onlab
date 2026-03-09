import numpy as np


def compute_speed(xs, timestamps):

    xs = np.array(xs)
    ts = np.array(timestamps)

    dx = np.diff(xs)
    dt = np.diff(ts)

    speeds = dx / dt

    return speeds