#!/bin/bash
set -u

reward_file=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$reward_file"
trap 'echo 0 > "$reward_file"; exit 130' INT TERM HUP

if ! python -m pip install --user --no-index --find-links /tests/wheels --require-hashes -r /tests/requirements.txt --quiet; then
  echo 0 > "$reward_file"
  exit 0
fi

cd /workspace
if python -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -q; then
  echo 1 > "$reward_file"
else
  echo 0 > "$reward_file"
fi

trap - INT TERM HUP
exit 0
