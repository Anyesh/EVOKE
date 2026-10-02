import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "window_b"))

import batch


def test_stop_at_none_means_no_deadline():
    assert batch.parse_stop("none") == float("inf")


def test_stop_at_clock_time_is_in_the_future():
    import time

    assert batch.parse_stop("07:30") > time.time()


def _args(tmp_path, run_cmd):
    return argparse.Namespace(
        runs=str(tmp_path),
        stop_cmd="true",
        start_cmd="true",
        run_cmd=run_cmd,
        base_url="http://x",
        health_urls={"E": "http://x/healthz", "C": "http://x/health"},
    )


def test_runner_failure_with_dead_server_is_an_interruption_not_an_infra_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "wait_healthy", lambda url: True)
    monkeypatch.setattr(batch, "server_alive", lambda url: False)
    result = batch.run_arm(_args(tmp_path, "exit 1"), "t", "E")
    assert result["interrupted"] is True


def test_runner_failure_with_live_server_is_an_infra_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "wait_healthy", lambda url: True)
    monkeypatch.setattr(batch, "server_alive", lambda url: True)
    result = batch.run_arm(_args(tmp_path, "exit 1"), "t", "E")
    assert result["infra_failure"] is True
    assert not result.get("interrupted")


def test_exit_75_is_a_tempfail_with_the_reason_from_the_result_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "wait_healthy", lambda url: True)
    monkeypatch.setattr(batch, "server_alive", lambda url: True)
    cmd = 'echo \'{{"tempfail": true, "infra_reason": "insufficient_disk"}}\' > {result}; exit 75'
    result = batch.run_arm(_args(tmp_path, cmd), "t", "E")
    assert result["tempfail"] is True
    assert result["reason"] == "insufficient_disk"


def test_exit_75_without_a_result_file_is_still_a_tempfail(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "wait_healthy", lambda url: True)
    monkeypatch.setattr(batch, "server_alive", lambda url: True)
    result = batch.run_arm(_args(tmp_path, "exit 75"), "t", "E")
    assert result["tempfail"] is True
    assert result["reason"] == "exit 75"


def test_tempfail_result_file_with_exit_zero_is_a_tempfail(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "wait_healthy", lambda url: True)
    monkeypatch.setattr(batch, "server_alive", lambda url: True)
    cmd = (
        'echo \'{{"tempfail": true, "infra_reason": "docker_unavailable"}}\' > {result}'
    )
    result = batch.run_arm(_args(tmp_path, cmd), "t", "E")
    assert result["tempfail"] is True
    assert result["reason"] == "docker_unavailable"


def test_tempfail_never_consumes_an_attempt_and_the_batch_stops(tmp_path, monkeypatch):
    perm = tmp_path / "perm.json"
    perm.write_text('{"order": ["a", "b"]}')
    state = tmp_path / "state.json"
    monkeypatch.setattr(
        batch,
        "run_arm",
        lambda args, task, arm: {"tempfail": True, "reason": "insufficient_disk"},
    )
    monkeypatch.setattr(batch, "sh", lambda template, **kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch.py",
            "--perm",
            str(perm),
            "--state",
            str(state),
            "--runs",
            str(tmp_path),
            "--stop-at",
            "none",
            "--start-cmd",
            "x",
            "--stop-cmd",
            "y",
            "--run-cmd",
            "z",
            "--base-url",
            "http://x",
            "--health-url",
            "E=http://x",
            "--health-url",
            "C=http://x",
        ],
    )
    assert batch.main() == 75
    cells = batch.load(state)["tasks"]["a"]
    assert cells["E"]["status"] == "pending"
    assert cells["E"]["attempts"] == 0
    assert "b" not in batch.load(state)["tasks"]
