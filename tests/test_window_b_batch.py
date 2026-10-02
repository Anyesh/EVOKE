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
