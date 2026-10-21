import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, "/workspace")


SOC_NODES = (0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0)
OCV_NODES = (3.00, 3.36, 3.50, 3.61, 3.70, 3.78, 3.88, 4.02, 4.20)
CAPACITY_AH = 3.2
R0_REF = 0.012
R1_REF = 0.018
C1 = 2400.0
Q_SOC = 2.5e-8
Q_VP = 1.0e-5
MEASUREMENT_VARIANCE = 4.0e-6


def close(actual, expected, tolerance=1e-10):
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def segment_index(soc):
    z = min(1.0, max(0.0, soc))
    for index in range(len(SOC_NODES) - 1):
        if z < SOC_NODES[index + 1]:
            return index
    return len(SOC_NODES) - 2


def reference_ocv(soc, temp_c):
    z = min(1.0, max(0.0, soc))
    index = segment_index(z)
    z0, z1 = SOC_NODES[index : index + 2]
    v0, v1 = OCV_NODES[index : index + 2]
    return v0 + (z - z0) * (v1 - v0) / (z1 - z0) + 0.00035 * (temp_c - 25.0)


def reference_slope(soc):
    index = segment_index(soc)
    return (OCV_NODES[index + 1] - OCV_NODES[index]) / (
        SOC_NODES[index + 1] - SOC_NODES[index]
    )


def resistance_pair(temp_c):
    scale = math.exp(0.018 * (25.0 - temp_c))
    return R0_REF * scale, R1_REF * scale


def reference_polarization(polarization_v, current_a, dt_s, temp_c):
    _, r1 = resistance_pair(temp_c)
    decay = math.exp(-dt_s / (r1 * C1))
    return decay * polarization_v + r1 * (1.0 - decay) * current_a


def reference_voltage(soc, polarization_v, current_a, temp_c):
    r0, _ = resistance_pair(temp_c)
    return reference_ocv(soc, temp_c) - r0 * current_a - polarization_v


def multiply(left, right):
    return [
        [left[0][0] * right[0][0] + left[0][1] * right[1][0], left[0][0] * right[0][1] + left[0][1] * right[1][1]],
        [left[1][0] * right[0][0] + left[1][1] * right[1][0], left[1][0] * right[0][1] + left[1][1] * right[1][1]],
    ]


def transpose(matrix):
    return [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]


def reference_predict(soc, polarization_v, covariance, current_a, dt_s, temp_c):
    efficiency = 1.0 if current_a >= 0.0 else 0.995
    soc = min(1.0, max(0.0, soc - efficiency * current_a * dt_s / (3600.0 * CAPACITY_AH)))
    polarization_v = reference_polarization(polarization_v, current_a, dt_s, temp_c)
    _, r1 = resistance_pair(temp_c)
    decay = math.exp(-dt_s / (r1 * C1))
    p = covariance
    covariance = [
        [p[0][0] + Q_SOC * dt_s, p[0][1] * decay],
        [p[1][0] * decay, decay * decay * p[1][1] + Q_VP * (1.0 - decay * decay)],
    ]
    return soc, polarization_v, covariance


def reference_update(soc, polarization_v, covariance, current_a, measured_voltage_v, temp_c):
    predicted_voltage = reference_voltage(soc, polarization_v, current_a, temp_c)
    residual = measured_voltage_v - predicted_voltage
    h0, h1 = reference_slope(soc), -1.0
    p = covariance
    ph0 = p[0][0] * h0 + p[0][1] * h1
    ph1 = p[1][0] * h0 + p[1][1] * h1
    innovation = h0 * ph0 + h1 * ph1 + MEASUREMENT_VARIANCE
    k0, k1 = ph0 / innovation, ph1 / innovation
    soc = min(1.0, max(0.0, soc + k0 * residual))
    polarization_v += k1 * residual
    a_matrix = [[1.0 - k0 * h0, -k0 * h1], [-k1 * h0, 1.0 - k1 * h1]]
    covariance = multiply(multiply(a_matrix, p), transpose(a_matrix))
    covariance[0][0] += MEASUREMENT_VARIANCE * k0 * k0
    covariance[0][1] += MEASUREMENT_VARIANCE * k0 * k1
    covariance[1][0] += MEASUREMENT_VARIANCE * k1 * k0
    covariance[1][1] += MEASUREMENT_VARIANCE * k1 * k1
    cross = 0.5 * (covariance[0][1] + covariance[1][0])
    covariance[0][1] = covariance[1][0] = cross
    return soc, polarization_v, covariance, predicted_voltage, residual


def model_checks():
    from battery_estimator.model import BatteryModel

    model = BatteryModel()
    expected_ocv = reference_ocv(0.275, 10.0)
    expected_slope = reference_slope(0.275)
    expected_resistance = resistance_pair(5.0)
    expected_polarization = reference_polarization(0.014, 1.7, 19.0, 5.0)
    expected_voltage = reference_voltage(0.275, expected_polarization, 1.7, 5.0)
    actual_resistance = model.resistance_pair(5.0)
    actual_polarization = model.polarization_step(0.014, 1.7, 19.0, 5.0)
    return all(
        (
            close(model.ocv(0.275, 10.0), expected_ocv),
            close(model.ocv_slope(0.275), expected_slope),
            close(actual_resistance[0], expected_resistance[0]),
            close(actual_resistance[1], expected_resistance[1]),
            close(actual_polarization, expected_polarization),
            close(model.terminal_voltage(0.275, actual_polarization, 1.7, 5.0), expected_voltage),
        )
    )


def prediction_checks():
    from battery_estimator.estimator import BatteryEKF

    estimator = BatteryEKF(initial_soc=0.6, initial_polarization_v=0.02)
    estimator.covariance = [[0.02, 0.003], [0.003, 0.004]]
    expected = reference_predict(0.6, 0.02, [[0.02, 0.003], [0.003, 0.004]], 1.7, 13.0, 8.0)
    estimator.predict(1.7, 13.0, 8.0)
    discharge_ok = (
        close(estimator.soc, expected[0])
        and close(estimator.polarization_v, expected[1])
        and all(close(estimator.covariance[i][j], expected[2][i][j]) for i in range(2) for j in range(2))
    )
    charging = BatteryEKF(initial_soc=0.4, initial_polarization_v=-0.01)
    expected_charge = reference_predict(0.4, -0.01, [[0.01, 0.0], [0.0, 0.0025]], -1.2, 600.0, 30.0)
    charging.predict(-1.2, 600.0, 30.0)
    charge_ok = close(charging.soc, expected_charge[0]) and close(charging.polarization_v, expected_charge[1])
    invalid_ok = False
    try:
        charging.predict(1.0, 0.0, 25.0)
    except ValueError:
        invalid_ok = True
    return discharge_ok and charge_ok and invalid_ok


def update_checks():
    from battery_estimator.estimator import BatteryEKF

    estimator = BatteryEKF(initial_soc=0.44, initial_polarization_v=0.015)
    estimator.covariance = [[0.008, 0.0012], [0.0012, 0.003]]
    measured = reference_voltage(0.44, 0.015, 0.8, 17.0) + 0.0123
    expected = reference_update(0.44, 0.015, [[0.008, 0.0012], [0.0012, 0.003]], 0.8, measured, 17.0)
    predicted, residual = estimator.update(0.8, measured, 17.0)
    determinant = estimator.covariance[0][0] * estimator.covariance[1][1] - estimator.covariance[0][1] ** 2
    return (
        close(estimator.soc, expected[0])
        and close(estimator.polarization_v, expected[1])
        and close(predicted, expected[3])
        and close(residual, expected[4])
        and all(close(estimator.covariance[i][j], expected[2][i][j]) for i in range(2) for j in range(2))
        and close(estimator.covariance[0][1], estimator.covariance[1][0])
        and determinant >= 0.0
    )


def reference_sequence(rows):
    soc, polarization_v = 0.78, 0.0
    covariance = [[0.01, 0.0], [0.0, 0.0025]]
    previous_timestamp = None
    results = []
    for timestamp, current, measured, temp in rows:
        if previous_timestamp is not None:
            soc, polarization_v, covariance = reference_predict(
                soc, polarization_v, covariance, current, timestamp - previous_timestamp, temp
            )
        soc, polarization_v, covariance, predicted, residual = reference_update(
            soc, polarization_v, covariance, current, measured, temp
        )
        results.append((timestamp, soc, polarization_v, predicted, residual))
        previous_timestamp = timestamp
    return results


def cli_checks():
    rows = [
        (0.0, 0.3, 3.854, 24.0),
        (4.0, 1.1, 3.842, 23.5),
        (15.0, 1.8, 3.821, 20.0),
        (22.0, -0.7, 3.868, 18.0),
        (47.0, 2.2, 3.787, 12.0),
    ]
    expected = reference_sequence(rows)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.csv"
        output_path = root / "output.csv"
        summary_path = root / "summary.json"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("timestamp_s", "current_a", "voltage_v", "temp_c"))
            writer.writerows(rows)
        process = subprocess.run(
            [sys.executable, "-m", "battery_estimator.cli", str(input_path), str(output_path), "--summary", str(summary_path)],
            cwd="/workspace",
            capture_output=True,
            text=True,
            timeout=20,
        )
        if process.returncode != 0 or not output_path.exists() or not summary_path.exists():
            return False
        with output_path.open(newline="", encoding="utf-8") as handle:
            actual_rows = list(csv.DictReader(handle))
            if handle.closed:
                pass
        if list(actual_rows[0].keys()) != ["timestamp_s", "soc", "polarization_v", "voltage_estimate_v", "residual_v"]:
            return False
        if len(actual_rows) != len(expected):
            return False
        for actual, wanted in zip(actual_rows, expected):
            values = [float(actual[name]) for name in ("timestamp_s", "soc", "polarization_v", "voltage_estimate_v", "residual_v")]
            if not all(close(value, target, 2e-10) for value, target in zip(values, wanted)):
                return False
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        residuals = [row[4] for row in expected]
        expected_summary = {
            "rows": len(expected),
            "final_soc": expected[-1][1],
            "rms_residual_v": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
            "max_abs_residual_v": max(abs(value) for value in residuals),
        }
        if summary.get("rows") != expected_summary["rows"]:
            return False
        if any(not close(float(summary.get(name)), expected_summary[name], 2e-10) for name in ("final_soc", "rms_residual_v", "max_abs_residual_v")):
            return False

        bad_path = root / "bad.csv"
        bad_path.write_text(
            "timestamp_s,current_a,voltage_v,temp_c\n0,0,3.8,25\n0,1,3.7,25\n",
            encoding="utf-8",
        )
        bad_process = subprocess.run(
            [sys.executable, "-m", "battery_estimator.cli", str(bad_path), str(root / "bad-output.csv"), "--summary", str(root / "bad-summary.json")],
            cwd="/workspace",
            capture_output=True,
            text=True,
            timeout=20,
        )
        return bad_process.returncode != 0


def main():
    groups = [
        ("model_checks", 0.25, model_checks),
        ("prediction_checks", 0.25, prediction_checks),
        ("update_checks", 0.30, update_checks),
        ("cli_checks", 0.20, cli_checks),
    ]
    score = 0.0
    for name, weight, check in groups:
        try:
            passed = bool(check())
        except Exception as exc:
            passed = False
            print(f"{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print(f"{name}: {'pass' if passed else 'fail'}", file=sys.stderr)
        if passed:
            score += weight
    print(f"{score:.3f}")


if __name__ == "__main__":
    main()
