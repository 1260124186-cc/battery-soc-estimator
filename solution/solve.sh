#!/bin/bash
set -euo pipefail

install -m 0644 /solution/model.py /workspace/battery_estimator/model.py
install -m 0644 /solution/estimator.py /workspace/battery_estimator/estimator.py
install -m 0644 /solution/cli.py /workspace/battery_estimator/cli.py
