#!/usr/bin/env bash
# Stops the server started by serve-start.sh: SIGTERM to its process group, then SIGKILL after EVOKE_STOP_TIMEOUT seconds.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/serve-lib.sh"

if ! pid_alive; then
  command rm -f "$PID_FILE"
  echo "EVOKE is not running"
  exit 0
fi
pid="$(cat "$PID_FILE")"
kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
deadline=$((SECONDS + ${EVOKE_STOP_TIMEOUT:-30}))
while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then
  echo "EVOKE did not stop in time; sending SIGKILL" >&2
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
fi
command rm -f "$PID_FILE"
echo "EVOKE stopped"
