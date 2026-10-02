import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

STUB = """
import http.server, os
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(os.environ["EVOKE_PORT"])), H).serve_forever()
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    return {
        **os.environ,
        "EVOKE_STATE_DIR": str(tmp_path / "state"),
        "EVOKE_PORT": str(_free_port()),
        "EVOKE_SERVE_CMD": f"{sys.executable} {stub}",
        "EVOKE_START_TIMEOUT": "20",
        "EVOKE_STOP_TIMEOUT": "10",
    }


def run(script, env):
    return subprocess.run(
        ["bash", str(SCRIPTS / script)], env=env, capture_output=True, text=True, timeout=60
    )


def test_start_status_double_start_stop_cycle(env, tmp_path):
    try:
        started = run("serve-start.sh", env)
        assert started.returncode == 0, started.stdout + started.stderr
        assert (tmp_path / "state" / "evoke.pid").exists()

        status = run("serve-status.sh", env)
        assert status.returncode == 0
        assert '"status":"ok"' in status.stdout

        again = run("serve-start.sh", env)
        assert again.returncode == 1
        assert "already running" in again.stdout + again.stderr

        stopped = run("serve-stop.sh", env)
        assert stopped.returncode == 0
        assert not (tmp_path / "state" / "evoke.pid").exists()

        assert run("serve-status.sh", env).returncode == 1
    finally:
        run("serve-stop.sh", env)


def test_stop_with_nothing_running_succeeds(env):
    assert run("serve-stop.sh", env).returncode == 0


def test_start_fails_with_log_tail_when_the_server_never_answers(env, tmp_path):
    env = {**env, "EVOKE_SERVE_CMD": "sh -c 'echo boom; sleep 30'", "EVOKE_START_TIMEOUT": "3"}
    try:
        result = run("serve-start.sh", env)
        assert result.returncode == 1
        assert "boom" in result.stdout + result.stderr
    finally:
        run("serve-stop.sh", env)
