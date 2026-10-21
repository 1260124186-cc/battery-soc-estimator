import argparse
import csv
import json
import math
from pathlib import Path

from .estimator import BatteryEKF


FIELD_NAMES = ("timestamp_s", "current_a", "voltage_v", "temp_c")
OUTPUT_FIELDS = (
    "timestamp_s",
    "soc",
    "polarization_v",
    "voltage_estimate_v",
    "residual_v",
)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in FIELD_NAMES):
            raise ValueError("input columns are incomplete")
        rows = []
        for raw in reader:
            row = {name: float(raw[name]) for name in FIELD_NAMES}
            rows.append(row)
    return sorted(rows, key=lambda row: row["timestamp_s"])


def run(input_path, output_path, summary_path):
    rows = read_rows(input_path)
    estimator = BatteryEKF()
    results = []
    previous_timestamp = None
    for row in rows:
        dt_s = 1.0 if previous_timestamp is None else row["timestamp_s"] - previous_timestamp
        state = estimator.step(row["current_a"], row["voltage_v"], row["temp_c"], dt_s)
        results.append({"timestamp_s": row["timestamp_s"], **state})
        previous_timestamp = row["timestamp_s"]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    residuals = [row["residual_v"] for row in results]
    summary = {
        "rows": len(results),
        "final_soc": round(results[-1]["soc"], 4),
        "rms_residual_v": round(math.sqrt(sum(x * x for x in residuals) / len(residuals)), 4),
        "max_abs_residual_v": round(max(abs(x) for x in residuals), 4),
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    run(args.input, args.output, args.summary)


if __name__ == "__main__":
    main()
