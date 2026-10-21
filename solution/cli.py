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
        previous_timestamp = None
        for raw in reader:
            try:
                row = {name: float(raw[name]) for name in FIELD_NAMES}
            except (TypeError, ValueError) as exc:
                raise ValueError("input contains a non-numeric value") from exc
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError("input contains a non-finite value")
            if previous_timestamp is not None and row["timestamp_s"] <= previous_timestamp:
                raise ValueError("timestamps must be strictly increasing")
            rows.append(row)
            previous_timestamp = row["timestamp_s"]
    if not rows:
        raise ValueError("input contains no data rows")
    return rows


def run(input_path, output_path, summary_path):
    rows = read_rows(input_path)
    estimator = BatteryEKF()
    results = []
    previous_timestamp = None
    for row in rows:
        dt_s = None if previous_timestamp is None else row["timestamp_s"] - previous_timestamp
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
        "final_soc": results[-1]["soc"],
        "rms_residual_v": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "max_abs_residual_v": max(abs(value) for value in residuals),
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.summary)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
