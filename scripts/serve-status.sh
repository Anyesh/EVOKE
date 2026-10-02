#!/usr/bin/env bash
# Prints the /healthz answer and exits 0 when the server answers, 1 otherwise.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/serve-lib.sh"

if pid_alive; then
  echo "pid $(cat "$PID_FILE"), log $LOG_FILE"
else
  echo "no running process recorded in $PID_FILE"
fi
if body="$(healthy)"; then
  echo "$body"
  exit 0
fi
echo "no answer from $HEALTH_URL"
exit 1
