import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evoke_serve.py"


def test_help_prints_environment_and_exits_zero():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0
    assert "EVOKE_MODEL_PATH" in out.stdout


def test_missing_model_path_fails_with_message(monkeypatch):
    env = {"PATH": ""}
    out = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=60, env=env
    )
    assert out.returncode == 1
    assert "EVOKE_MODEL_PATH" in out.stdout
