#!/usr/bin/env bash
# Starts the EVOKE server in the background. Settings are the EVOKE_* variables (see OPERATING.md);
# EVOKE_STATE_DIR holds the pid file and server.log, EVOKE_START_TIMEOUT is the seconds to wait for /healthz.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/serve-lib.sh"

mkdir -p "$STATE_DIR"
if pid_alive; then
  echo "EVOKE is already running (pid $(cat "$PID_FILE")); run serve-stop.sh first" >&2
  exit 1
fi
if healthy >/dev/null; then
  echo "something already answers $HEALTH_URL; refusing to start a second server" >&2
  exit 1
fi

cd "$REPO_ROOT"
# shellcheck disable=SC2086
setsid nohup ${EVOKE_SERVE_CMD:-uv run python scripts/evoke_serve.py} >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

deadline=$((SECONDS + ${EVOKE_START_TIMEOUT:-120}))
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! pid_alive; then
    break
  fi
  if healthy >/dev/null; then
    echo "EVOKE is up (pid $(cat "$PID_FILE")), log $LOG_FILE"
    exit 0
  fi
  sleep 2
done
echo "EVOKE did not answer $HEALTH_URL; last log lines:" >&2
tail -n 30 "$LOG_FILE" >&2
exit 1
