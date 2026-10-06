import numpy as np


class SwimmerKalmanFilter1D:
    """
    1D Kalman-szűrő az úszó sávirányú pozíciójának és sebességének valós idejű becslésére.
    Állapotvektor: x = [pozíció (m), sebesség (m/s)]^T

    Tulajdonságok:
    - Dinamikus időközök (dt) kezelése időbélyegek alapján.
    - Kétirányú úszás kezelése (pozitív és negatív sebességvektor).
    - Coasting: Víz alatti vagy kitakart fázisban tisztán predikcióval hidal át.
    - Outlier-szűrés: Kiszűri a fröccsenésekből vagy szomszédos sávból származó téves ugrásokat.
    """

    def __init__(
        self,
        initial_pos: float,
        initial_time: float,
        process_noise_std: float = 0.8,
        measurement_noise_std: float = 0.15,
    ):
        self.x = np.array([float(initial_pos), 0.0], dtype=np.float64)
        # Kezdeti állapot bizonytalansága
        self.P = np.diag([0.2**2, 1.0**2])
        self.last_time = float(initial_time)

        self.q_std = float(process_noise_std)        # Úszó gyorsulási varianciája (m/s^2)
        self.r_var = float(measurement_noise_std**2) # Mérés bizonytalansága (m^2)
        self.missed_frames = 0

    def predict(self, current_time: float):
        """Állapotbecslés az eltelt dt alapján."""
        dt = float(current_time) - self.last_time
        self.last_time = float(current_time)

        if dt <= 0:
            return float(self.x[0]), float(self.x[1])

        # Állapotátmenet mátrix: p = p + v*dt, v = v
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)

        # Diszkrét folyamatzaj (Piecewise Constant White Acceleration modell)
        G = np.array([[0.5 * dt**2],
                      [dt]], dtype=np.float64)
        Q = (G @ G.T) * (self.q_std**2)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.missed_frames += 1

        return float(self.x[0]), float(self.x[1])

    def update(self, measured_pos: float, max_residual_m: float = 1.2):
        """Mérés beépítése adaptív ugrásellenőrzéssel."""
        pred_pos = self.x[0]
        residual = abs(measured_pos - pred_pos)

        # Ha a detektálás hirtelen túl messzire ugrik, de a megelőző frame-eken
        # stabilan követtük, eldobjuk mérési hibaként (outlier).
        if residual > max_residual_m and self.missed_frames < 8:
            return float(self.x[0]), float(self.x[1])

        H = np.array([[1.0, 0.0]], dtype=np.float64)
        y = measured_pos - pred_pos  # Innováció
        S = (H @ self.P @ H.T)[0, 0] + self.r_var
        K = (self.P @ H.T) / S

        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ H) @ self.P
        self.missed_frames = 0

        return float(self.x[0]), float(self.x[1])

    def get_position(self) -> float:
        return float(self.x[0])

    def get_velocity(self) -> float:
        """Előjeles sebesség (m/s)."""
        return float(self.x[1])

    def get_speed(self) -> float:
        """Abszolút haladási sebesség (m/s)."""
        return float(abs(self.x[1]))