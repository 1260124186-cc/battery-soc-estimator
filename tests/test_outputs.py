import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

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


def matmul(left, right):
    return [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]


def transpose(matrix):
    return [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]


def reference_predict(soc, polarization_v, covariance, current_a, dt_s, temp_c):
    efficiency = 1.0 if current_a >= 0.0 else 0.995
    soc = min(
        1.0,
        max(0.0, soc - efficiency * current_a * dt_s / (3600.0 * CAPACITY_AH)),
    )
    polarization_v = reference_polarization(
        polarization_v, current_a, dt_s, temp_c
    )
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
    covariance = matmul(matmul(a_matrix, p), transpose(a_matrix))
    covariance[0][0] += MEASUREMENT_VARIANCE * k0 * k0
    covariance[0][1] += MEASUREMENT_VARIANCE * k0 * k1
    covariance[1][0] += MEASUREMENT_VARIANCE * k1 * k0
    covariance[1][1] += MEASUREMENT_VARIANCE * k1 * k1
    cross = 0.5 * (covariance[0][1] + covariance[1][0])
    covariance[0][1] = covariance[1][0] = cross
    return soc, polarization_v, covariance, predicted_voltage, residual


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


def _model():
    from battery_estimator.model import BatteryModel
    return BatteryModel()


def _estimator(initial_soc=0.6, initial_polarization_v=0.02):
    from battery_estimator.estimator import BatteryEKF
    return BatteryEKF(initial_soc=initial_soc, initial_polarization_v=initial_polarization_v)


def _close(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def test_model_ocv_is_piecewise_linear():
    model = _model()
    assert _close(model.ocv(0.275, 25.0), reference_ocv(0.275, 25.0))


def test_model_ocv_temperature_correction_sign():
    model = _model()
    assert _close(model.ocv(0.275, 10.0), reference_ocv(0.275, 10.0))


def test_model_ocv_clamps_to_endpoints():
    model = _model()
    assert _close(model.ocv(-0.5, 25.0), reference_ocv(-0.5, 25.0))
    assert _close(model.ocv(1.5, 25.0), reference_ocv(1.5, 25.0))


def test_model_ocv_slope_uses_current_segment():
    model = _model()
    assert _close(model.ocv_slope(0.275), reference_slope(0.275))


def test_model_resistance_exponential_temperature_scaling():
    model = _model()
    actual = model.resistance_pair(5.0)
    expected = resistance_pair(5.0)
    assert _close(actual[0], expected[0]) and _close(actual[1], expected[1])


def test_model_polarization_rc_discrete_step():
    model = _model()
    assert _close(
        model.polarization_step(0.014, 1.7, 19.0, 5.0),
        reference_polarization(0.014, 1.7, 19.0, 5.0),
    )


def test_model_terminal_voltage_sign_convention():
    model = _model()
    vp = reference_polarization(0.014, 1.7, 19.0, 5.0)
    assert _close(
        model.terminal_voltage(0.275, vp, 1.7, 5.0),
        reference_voltage(0.275, vp, 1.7, 5.0),
    )


def test_predict_discharge_soc_and_ah_conversion():
    estimator = _estimator(0.6, 0.02)
    estimator.covariance = [[0.02, 0.003], [0.003, 0.004]]
    expected = reference_predict(0.6, 0.02, [[0.02, 0.003], [0.003, 0.004]], 1.7, 13.0, 8.0)
    estimator.predict(1.7, 13.0, 8.0)
    assert _close(estimator.soc, expected[0])


def test_predict_charge_uses_efficiency():
    estimator = _estimator(0.4, -0.01)
    estimator.covariance = [[0.01, 0.0], [0.0, 0.0025]]
    expected = reference_predict(0.4, -0.01, [[0.01, 0.0], [0.0, 0.0025]], -1.2, 600.0, 30.0)
    estimator.predict(-1.2, 600.0, 30.0)
    assert _close(estimator.soc, expected[0])


def test_predict_polarization_uses_actual_interval():
    estimator = _estimator(0.6, 0.02)
    estimator.covariance = [[0.02, 0.003], [0.003, 0.004]]
    expected = reference_predict(0.6, 0.02, [[0.02, 0.003], [0.003, 0.004]], 1.7, 13.0, 8.0)
    estimator.predict(1.7, 13.0, 8.0)
    assert _close(estimator.polarization_v, expected[1])


def test_predict_covariance_propagates_through_state_transition():
    estimator = _estimator(0.6, 0.02)
    estimator.covariance = [[0.02, 0.003], [0.003, 0.004]]
    expected = reference_predict(0.6, 0.02, [[0.02, 0.003], [0.003, 0.004]], 1.7, 13.0, 8.0)
    estimator.predict(1.7, 13.0, 8.0)
    assert all(
        _close(estimator.covariance[i][j], expected[2][i][j])
        for i in range(2)
        for j in range(2)
    )


def test_predict_rejects_nonpositive_interval():
    estimator = _estimator(0.4, -0.01)
    with pytest.raises(ValueError):
        estimator.predict(1.0, 0.0, 25.0)


def test_update_voltage_jacobian_and_residual():
    estimator = _estimator(0.44, 0.015)
    estimator.covariance = [[0.008, 0.0012], [0.0012, 0.003]]
    measured = reference_voltage(0.44, 0.015, 0.8, 17.0) + 0.0123
    expected = reference_update(0.44, 0.015, [[0.008, 0.0012], [0.0012, 0.003]], 0.8, measured, 17.0)
    predicted, residual = estimator.update(0.8, measured, 17.0)
    assert _close(predicted, expected[3]) and _close(residual, expected[4])


def test_update_soc_and_polarization():
    estimator = _estimator(0.44, 0.015)
    estimator.covariance = [[0.008, 0.0012], [0.0012, 0.003]]
    measured = reference_voltage(0.44, 0.015, 0.8, 17.0) + 0.0123
    expected = reference_update(0.44, 0.015, [[0.008, 0.0012], [0.0012, 0.003]], 0.8, measured, 17.0)
    estimator.update(0.8, measured, 17.0)
    assert _close(estimator.soc, expected[0]) and _close(estimator.polarization_v, expected[1])


def test_update_soc_is_clamped():
    estimator = _estimator(0.999, 0.0)
    estimator.covariance = [[0.008, 0.0012], [0.0012, 0.003]]
    measured = reference_voltage(0.999, 0.0, 2.0, 25.0) + 0.05
    estimator.update(2.0, measured, 25.0)
    assert 0.0 <= estimator.soc <= 1.0


def test_update_covariance_is_symmetric_joseph():
    estimator = _estimator(0.44, 0.015)
    estimator.covariance = [[0.008, 0.0012], [0.0012, 0.003]]
    measured = reference_voltage(0.44, 0.015, 0.8, 17.0) + 0.0123
    estimator.update(0.8, measured, 17.0)
    determinant = (
        estimator.covariance[0][0] * estimator.covariance[1][1]
        - estimator.covariance[0][1] ** 2
    )
    assert _close(estimator.covariance[0][1], estimator.covariance[1][0])
    assert determinant >= -1e-18


def _run_cli(rows, cwd="/workspace"):
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
            [
                sys.executable,
                "-m",
                "battery_estimator.cli",
                str(input_path),
                str(output_path),
                "--summary",
                str(summary_path),
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output_rows = None
        if output_path.exists():
            with output_path.open(newline="", encoding="utf-8") as handle:
                output_rows = list(csv.DictReader(handle))
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return process, output_rows, summary


def _sample_rows():
    return [
        (0.0, 0.3, 3.854, 24.0),
        (4.0, 1.1, 3.842, 23.5),
        (15.0, 1.8, 3.821, 20.0),
        (22.0, -0.7, 3.868, 18.0),
        (47.0, 2.2, 3.787, 12.0),
    ]


def test_cli_writes_expected_output_schema():
    process, actual_rows, _ = _run_cli(_sample_rows())
    assert process.returncode == 0
    assert actual_rows is not None and len(actual_rows) > 0
    assert list(actual_rows[0].keys()) == [
        "timestamp_s",
        "soc",
        "polarization_v",
        "voltage_estimate_v",
        "residual_v",
    ]


def test_cli_sequence_matches_reference():
    rows = _sample_rows()
    expected = reference_sequence(rows)
    process, actual_rows, _ = _run_cli(rows)
    assert process.returncode == 0
    assert actual_rows is not None
    assert len(actual_rows) == len(expected)
    for actual, wanted in zip(actual_rows, expected):
        values = [
            float(actual[name])
            for name in (
                "timestamp_s",
                "soc",
                "polarization_v",
                "voltage_estimate_v",
                "residual_v",
            )
        ]
        assert all(_close(value, target, rel_tol=1e-8) for value, target in zip(values, wanted))


def test_cli_summary_is_full_precision():
    rows = _sample_rows()
    expected = reference_sequence(rows)
    process, _, summary = _run_cli(rows)
    assert process.returncode == 0
    assert summary is not None
    residuals = [row[4] for row in expected]
    expected_summary = {
        "rows": len(expected),
        "final_soc": expected[-1][1],
        "rms_residual_v": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "max_abs_residual_v": max(abs(value) for value in residuals),
    }
    assert summary.get("rows") == expected_summary["rows"]
    for name in ("final_soc", "rms_residual_v", "max_abs_residual_v"):
        assert _close(float(summary.get(name)), expected_summary[name], rel_tol=1e-8)


def test_cli_rejects_duplicate_or_descending_timestamps():
    bad_rows = [
        (0.0, 0.3, 3.8, 25.0),
        (0.0, 1.1, 3.7, 25.0),
    ]
    process, output_rows, summary = _run_cli(bad_rows)
    assert process.returncode != 0
    assert output_rows is None
    assert summary is None


def test_cli_rejects_non_numeric_values():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bad_path = root / "bad.csv"
        bad_path.write_text(
            "timestamp_s,current_a,voltage_v,temp_c\n0,0,3.8,25\n1,abc,3.7,25\n",
            encoding="utf-8",
        )
        out_path = root / "out.csv"
        summary_path = root / "summary.json"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "battery_estimator.cli",
                str(bad_path),
                str(out_path),
                "--summary",
                str(summary_path),
            ],
            cwd="/workspace",
            capture_output=True,
            text=True,
            timeout=30,
        )
        output_exists = out_path.exists()
        summary_exists = summary_path.exists()
    assert process.returncode != 0
    assert not output_exists
    assert not summary_exists


def test_workspace_has_no_test_framework_hijack():
    """校验 /workspace 里没有劫持测试框架的同名模块或入口。"""
    forbidden = (
        "pytest.py",
        "pytest",
        "conftest.py",
        "sitecustomize.py",
        "usercustomize.py",
        "_pytest",
        "_pytest.py",
    )
    found = [
        name for name in forbidden if os.path.exists(os.path.join("/workspace", name))
    ]
    assert not found, f"/workspace 存在测试框架劫持文件: {found}"
