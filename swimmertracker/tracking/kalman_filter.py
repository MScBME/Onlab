import numpy as np


class SwimmerKalmanFilter1D:
    """1D Kalman filter that estimates the position and speed of the swimmer along the lane in real time.

    The state vector is x = [position in meters, velocity in m/s]^T. The filter takes the time step from the
    timestamps, handles swimming in both directions with positive and negative velocity, bridges underwater or
    occluded stretches by prediction alone, and rejects false jumps caused by splashes or the neighbouring lane.
    """

    def __init__(
        self,
        initial_pos: float,
        initial_time: float,
        process_noise_std: float = 0.8,
        measurement_noise_std: float = 0.15,
    ):
        self.x = np.array([float(initial_pos), 0.0], dtype=np.float64)
        # Uncertainty of the initial state
        self.P = np.diag([0.2**2, 1.0**2])
        self.last_time = float(initial_time)

        self.q_std = float(process_noise_std)         # standard deviation of the swimmer's acceleration in m/s^2
        self.r_var = float(measurement_noise_std**2)  # variance of the measurement in m^2
        self.missed_frames = 0

    def predict(self, current_time: float):
        """Predict the state after the time elapsed since the last call."""
        dt = float(current_time) - self.last_time
        self.last_time = float(current_time)

        if dt <= 0:
            return float(self.x[0]), float(self.x[1])

        # State transition matrix, p = p + v*dt and v = v
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)

        # Discrete process noise of the piecewise constant white acceleration model
        G = np.array([[0.5 * dt**2],
                      [dt]], dtype=np.float64)
        Q = (G @ G.T) * (self.q_std**2)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.missed_frames += 1

        return float(self.x[0]), float(self.x[1])

    def update(self, measured_pos: float, max_residual_m: float = 1.2):
        """Incorporate a measurement, with an adaptive check against jumps."""
        pred_pos = self.x[0]
        residual = abs(measured_pos - pred_pos)

        # If the detection suddenly jumps too far while the previous frames were tracked
        # steadily, the measurement is dropped as an error.
        if residual > max_residual_m and self.missed_frames < 8:
            return float(self.x[0]), float(self.x[1])

        H = np.array([[1.0, 0.0]], dtype=np.float64)
        y = measured_pos - pred_pos  # innovation
        S = (H @ self.P @ H.T)[0, 0] + self.r_var
        K = (self.P @ H.T) / S

        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ H) @ self.P
        self.missed_frames = 0

        return float(self.x[0]), float(self.x[1])

    def get_position(self) -> float:
        return float(self.x[0])

    def get_velocity(self) -> float:
        """Signed velocity in m/s."""
        return float(self.x[1])

    def get_speed(self) -> float:
        """Absolute value of the velocity in m/s."""
        return float(abs(self.x[1]))
