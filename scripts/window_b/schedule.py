import random
import statistics

ARMS = ("E", "C")
MAX_ATTEMPTS = 2
DEFAULT_ARM_SECONDS = 600
SWAP_SECONDS = 120


def permutation(ids, seed):
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def arm_order(index):
    return ARMS if index % 2 == 0 else ARMS[::-1]


def new_state():
    return {"tasks": {}}


def _cell(state, task, arm):
    arms = state["tasks"].setdefault(task, {})
    return arms.setdefault(
        arm, {"status": "pending", "attempts": 0, "interrupted": 0, "result": None}
    )


def start(state, task, arm):
    _cell(state, task, arm)["status"] = "running"


def record(state, task, arm, result):
    cell = _cell(state, task, arm)
    cell["result"] = result
    if result.get("infra_failure"):
        cell["attempts"] += 1
        cell["status"] = "failed" if cell["attempts"] >= MAX_ATTEMPTS else "pending"
    else:
        cell["status"] = "done"


def recover(state):
    # a run cut off by power loss is not an infrastructure failure, so it keeps its retry
    for arms in state["tasks"].values():
        for cell in arms.values():
            if cell["status"] == "running":
                cell["status"] = "pending"
                cell["interrupted"] += 1


def estimate_arm_seconds(state):
    seen = [
        cell["result"]["seconds"]
        for arms in state["tasks"].values()
        for cell in arms.values()
        if cell["status"] == "done" and cell["result"].get("seconds")
    ]
    base = statistics.median(seen) if seen else DEFAULT_ARM_SECONDS
    return base + SWAP_SECONDS


def _excluded(state, task):
    arms = state["tasks"].get(task, {})
    return any(c["status"] == "failed" for c in arms.values())


def next_job(state, perm, now, stop_at, arm_seconds=None, limit=None):
    arm_seconds = arm_seconds if arm_seconds is not None else estimate_arm_seconds(state)
    for index, task in enumerate(perm[:limit]):
        if _excluded(state, task):
            continue
        remaining = [
            arm for arm in arm_order(index) if _cell(state, task, arm)["status"] != "done"
        ]
        if not remaining:
            continue
        if now + len(remaining) * arm_seconds > stop_at:
            return None
        return task, remaining[0]
    return None
