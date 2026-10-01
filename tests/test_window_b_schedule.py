import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "window_b"))

from schedule import (
    arm_order,
    estimate_arm_seconds,
    new_state,
    next_job,
    permutation,
    record,
    recover,
    start,
)

IDS = [f"t{i}" for i in range(10)]


def test_permutation_is_seeded_and_input_order_independent():
    a = permutation(IDS, 20260927)
    assert a == permutation(list(reversed(IDS)), 20260927)
    assert sorted(a) == sorted(IDS)
    assert a != permutation(IDS, 1)


def test_arm_order_alternates_by_position():
    assert arm_order(0) == ("E", "C")
    assert arm_order(1) == ("C", "E")


def ok(secs=600):
    return {"resolved": True, "infra_failure": False, "seconds": secs}


def infra():
    return {"resolved": None, "infra_failure": True, "seconds": 30}


def test_pair_runs_back_to_back_in_counterbalanced_order():
    perm = ["a", "b"]
    s = new_state()
    assert next_job(s, perm, 0, 10**6) == ("a", "E")
    start(s, "a", "E")
    record(s, "a", "E", ok())
    assert next_job(s, perm, 0, 10**6) == ("a", "C")
    start(s, "a", "C")
    record(s, "a", "C", ok())
    assert next_job(s, perm, 0, 10**6) == ("b", "C")


def test_new_pair_refused_when_it_cannot_finish_before_stop():
    s = new_state()
    assert next_job(s, ["a"], now=0, stop_at=1000, arm_seconds=600) is None
    assert next_job(s, ["a"], now=0, stop_at=1500, arm_seconds=600) == ("a", "E")


def test_second_arm_of_split_pair_needs_only_one_arm_of_time():
    s = new_state()
    start(s, "a", "E")
    record(s, "a", "E", ok())
    assert next_job(s, ["a"], now=0, stop_at=700, arm_seconds=600) == ("a", "C")


def test_interrupted_arm_reruns_without_consuming_a_retry():
    s = new_state()
    start(s, "a", "E")
    recover(s)
    assert s["tasks"]["a"]["E"]["status"] == "pending"
    assert s["tasks"]["a"]["E"]["attempts"] == 0
    assert s["tasks"]["a"]["E"]["interrupted"] == 1


def test_infra_failure_retried_once_then_task_excluded_from_both_arms():
    s = new_state()
    start(s, "a", "E")
    record(s, "a", "E", infra())
    assert next_job(s, ["a", "b"], 0, 10**6) == ("a", "E")
    start(s, "a", "E")
    record(s, "a", "E", infra())
    assert s["tasks"]["a"]["E"]["status"] == "failed"
    assert next_job(s, ["a", "b"], 0, 10**6) == ("b", "C")


def test_limit_caps_tasks_considered():
    s = new_state()
    for arm in ("E", "C"):
        start(s, "a", arm)
        record(s, "a", arm, ok())
    assert next_job(s, ["a", "b"], 0, 10**6, limit=1) is None


def test_estimate_uses_median_of_observed_and_defaults_when_empty():
    s = new_state()
    assert estimate_arm_seconds(s) == 720
    for task, secs in (("x", 100), ("y", 300), ("z", 200)):
        start(s, task, "E")
        record(s, task, "E", ok(secs))
    assert estimate_arm_seconds(s) == 200 + 120
