import math


class BatteryModel:
    capacity_ah = 3.2
    r0_ref = 0.012
    r1_ref = 0.018
    c1 = 2400.0
    resistance_temp_coefficient = 0.018
    ocv_temp_coefficient = 0.00035

    soc_nodes = (0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0)
    ocv_nodes = (3.00, 3.36, 3.50, 3.61, 3.70, 3.78, 3.88, 4.02, 4.20)

    def resistance_pair(self, temp_c):
        scale = 1.0 + self.resistance_temp_coefficient * (temp_c - 25.0)
        return self.r0_ref * scale, self.r1_ref * scale

    def ocv(self, soc, temp_c):
        z = min(1.0, max(0.0, soc))
        nearest = min(range(len(self.soc_nodes)), key=lambda i: abs(self.soc_nodes[i] - z))
        return self.ocv_nodes[nearest] - self.ocv_temp_coefficient * (temp_c - 25.0)

    def ocv_slope(self, soc):
        return (self.ocv_nodes[-1] - self.ocv_nodes[0]) / (self.soc_nodes[-1] - self.soc_nodes[0])

    def polarization_step(self, polarization_v, current_a, dt_s, temp_c):
        _, r1 = self.resistance_pair(temp_c)
        return polarization_v + dt_s * (current_a * r1 - polarization_v) / (r1 * self.c1)

    def terminal_voltage(self, soc, polarization_v, current_a, temp_c):
        r0, _ = self.resistance_pair(temp_c)
        return self.ocv(soc, temp_c) - r0 * current_a + polarization_v
