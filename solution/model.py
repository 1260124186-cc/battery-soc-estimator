import bisect
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
        scale = math.exp(self.resistance_temp_coefficient * (25.0 - temp_c))
        return self.r0_ref * scale, self.r1_ref * scale

    def _segment_index(self, soc):
        z = min(1.0, max(0.0, soc))
        if z >= self.soc_nodes[-1]:
            return len(self.soc_nodes) - 2
        return max(0, bisect.bisect_right(self.soc_nodes, z) - 1)

    def ocv(self, soc, temp_c):
        z = min(1.0, max(0.0, soc))
        index = self._segment_index(z)
        z0, z1 = self.soc_nodes[index : index + 2]
        v0, v1 = self.ocv_nodes[index : index + 2]
        fraction = (z - z0) / (z1 - z0)
        return v0 + fraction * (v1 - v0) + self.ocv_temp_coefficient * (temp_c - 25.0)

    def ocv_slope(self, soc):
        index = self._segment_index(soc)
        z0, z1 = self.soc_nodes[index : index + 2]
        v0, v1 = self.ocv_nodes[index : index + 2]
        return (v1 - v0) / (z1 - z0)

    def polarization_step(self, polarization_v, current_a, dt_s, temp_c):
        _, r1 = self.resistance_pair(temp_c)
        decay = math.exp(-dt_s / (r1 * self.c1))
        return decay * polarization_v + r1 * (1.0 - decay) * current_a

    def terminal_voltage(self, soc, polarization_v, current_a, temp_c):
        r0, _ = self.resistance_pair(temp_c)
        return self.ocv(soc, temp_c) - r0 * current_a - polarization_v
