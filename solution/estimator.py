import math

from .model import BatteryModel


def _multiply_2x2(left, right):
    return [
        [left[0][0] * right[0][0] + left[0][1] * right[1][0], left[0][0] * right[0][1] + left[0][1] * right[1][1]],
        [left[1][0] * right[0][0] + left[1][1] * right[1][0], left[1][0] * right[0][1] + left[1][1] * right[1][1]],
    ]


def _transpose_2x2(matrix):
    return [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]


class BatteryEKF:
    q_soc = 2.5e-8
    q_vp = 1.0e-5
    measurement_variance = 4.0e-6

    def __init__(self, model=None, initial_soc=0.78, initial_polarization_v=0.0):
        self.model = model or BatteryModel()
        self.soc = float(initial_soc)
        self.polarization_v = float(initial_polarization_v)
        self.covariance = [[0.01, 0.0], [0.0, 0.0025]]

    def predict(self, current_a, dt_s, temp_c):
        if not all(math.isfinite(value) for value in (current_a, dt_s, temp_c)) or dt_s <= 0.0:
            raise ValueError("prediction inputs must be finite and dt_s must be positive")
        efficiency = 1.0 if current_a >= 0.0 else 0.995
        self.soc = min(
            1.0,
            max(0.0, self.soc - efficiency * current_a * dt_s / (3600.0 * self.model.capacity_ah)),
        )
        _, r1 = self.model.resistance_pair(temp_c)
        decay = math.exp(-dt_s / (r1 * self.model.c1))
        self.polarization_v = self.model.polarization_step(
            self.polarization_v, current_a, dt_s, temp_c
        )
        p = self.covariance
        self.covariance = [
            [p[0][0] + self.q_soc * dt_s, p[0][1] * decay],
            [p[1][0] * decay, decay * decay * p[1][1] + self.q_vp * (1.0 - decay * decay)],
        ]

    def update(self, current_a, measured_voltage_v, temp_c):
        if not all(math.isfinite(value) for value in (current_a, measured_voltage_v, temp_c)):
            raise ValueError("measurement inputs must be finite")
        predicted_voltage = self.model.terminal_voltage(
            self.soc, self.polarization_v, current_a, temp_c
        )
        residual = measured_voltage_v - predicted_voltage
        h0 = self.model.ocv_slope(self.soc)
        h1 = -1.0
        p = self.covariance
        ph0 = p[0][0] * h0 + p[0][1] * h1
        ph1 = p[1][0] * h0 + p[1][1] * h1
        innovation_variance = h0 * ph0 + h1 * ph1 + self.measurement_variance
        if innovation_variance <= 0.0 or not math.isfinite(innovation_variance):
            raise ValueError("invalid innovation variance")
        k0 = ph0 / innovation_variance
        k1 = ph1 / innovation_variance
        self.soc = min(1.0, max(0.0, self.soc + k0 * residual))
        self.polarization_v += k1 * residual

        identity_minus_kh = [
            [1.0 - k0 * h0, -k0 * h1],
            [-k1 * h0, 1.0 - k1 * h1],
        ]
        joseph = _multiply_2x2(
            _multiply_2x2(identity_minus_kh, p), _transpose_2x2(identity_minus_kh)
        )
        joseph[0][0] += self.measurement_variance * k0 * k0
        joseph[0][1] += self.measurement_variance * k0 * k1
        joseph[1][0] += self.measurement_variance * k1 * k0
        joseph[1][1] += self.measurement_variance * k1 * k1
        cross = 0.5 * (joseph[0][1] + joseph[1][0])
        self.covariance = [[joseph[0][0], cross], [cross, joseph[1][1]]]
        return predicted_voltage, residual

    def step(self, current_a, measured_voltage_v, temp_c, dt_s=None):
        if dt_s is not None:
            self.predict(current_a, dt_s, temp_c)
        predicted_voltage, residual = self.update(current_a, measured_voltage_v, temp_c)
        return {
            "soc": self.soc,
            "polarization_v": self.polarization_v,
            "voltage_estimate_v": predicted_voltage,
            "residual_v": residual,
        }
