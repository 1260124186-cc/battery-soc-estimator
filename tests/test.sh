#!/bin/bash
set -u

reward_file=/logs/verifier/reward.txt
mkdir -p /logs/verifier

write_reward() {
    chmod 0644 "$reward_file" 2>/dev/null || true
    printf '%s\n' "$1" > "$reward_file"
    chmod 0444 "$reward_file"
}

write_reward 0
trap 'write_reward 0; exit 130' INT TERM HUP

score="$(cd /workspace && python /tests/check.py)"
status=$?
if [ "$status" -eq 0 ] && printf '%s' "$score" | grep -Eq '^(0(\.[0-9]+)?|1(\.0+)?)$'; then
    write_reward "$score"
else
    write_reward 0
fi

trap - INT TERM HUP
exit 0
