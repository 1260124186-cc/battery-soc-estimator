from .model import BatteryModel


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
        efficiency = 0.995 if current_a >= 0.0 else 1.0
        self.soc = min(1.0, max(0.0, self.soc + efficiency * current_a * dt_s / self.model.capacity_ah))
        self.polarization_v = self.model.polarization_step(
            self.polarization_v, current_a, 1.0, temp_c
        )
        self.covariance[0][0] += self.q_soc
        self.covariance[1][1] += self.q_vp

    def update(self, current_a, measured_voltage_v, temp_c):
        predicted_voltage = self.model.terminal_voltage(
            self.soc, self.polarization_v, current_a, temp_c
        )
        residual = measured_voltage_v - predicted_voltage
        h = [self.model.ocv_slope(self.soc), 1.0]
        p = self.covariance
        innovation_variance = (
            h[0] * (p[0][0] * h[0] + p[0][1] * h[1])
            + h[1] * (p[1][0] * h[0] + p[1][1] * h[1])
            + self.measurement_variance
        )
        k0 = (p[0][0] * h[0] + p[0][1] * h[1]) / innovation_variance
        k1 = (p[1][0] * h[0] + p[1][1] * h[1]) / innovation_variance
        self.soc = min(1.0, max(0.0, self.soc + k0 * residual))
        self.polarization_v += k1 * residual
        self.covariance = [
            [p[0][0] * (1.0 - k0 * h[0]), p[0][1]],
            [p[1][0], p[1][1] * (1.0 - k1 * h[1])],
        ]
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
