REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${EVOKE_STATE_DIR:-$REPO_ROOT/.evoke-state}"
PID_FILE="$STATE_DIR/evoke.pid"
LOG_FILE="$STATE_DIR/server.log"
HEALTH_HOST="${EVOKE_HEALTH_HOST:-${EVOKE_HOST:-127.0.0.1}}"
HEALTH_URL="http://$HEALTH_HOST:${EVOKE_PORT:-8000}/healthz"

pid_alive() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

healthy() {
  curl -fsS -m 5 "$HEALTH_URL" 2>/dev/null
}
