"""Direct OpenAI-API demo of EVOKE eviction + recovery in a long session.

Bypasses opencode (whose chat-template formatting drifts from ours and
triggers session resets). Drives the server with a controlled multi-turn
conversation that monotonically grows the prefix, so the server's prefix
cache always hits and the EvokeManager underneath has a chance to evict
under budget pressure and recover relevant blocks for the next query.

Plants a fact early, fills the session with unrelated filler turns to push
past the budget, then probes whether the fact still surfaces.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import urllib.request
import urllib.error
import json

BASE = os.environ.get("EVOKE_SERVER", "http://HOST:8000")
MODEL = os.environ.get("EVOKE_MODEL_NAME", "qwen35")

PASSKEY = "4242"
FACT = (
    f"Important system note: my favorite number is {PASSKEY}. "
    "Please remember this for the rest of our conversation."
)
PROBE = "Earlier in this conversation I told you my favorite number. What was it? Reply with just the number."

FILLERS = [
    "Explain the concept of an algorithm in one short paragraph.",
    "Explain the difference between recursion and iteration in one paragraph.",
    "Describe polymorphism in object-oriented programming in one paragraph.",
    "Explain Big-O notation briefly.",
    "Describe what a hash table is and one common use case.",
    "Explain garbage collection in one paragraph.",
    "Describe the publish-subscribe pattern.",
    "Explain what a closure is in programming.",
    "Describe the difference between TCP and UDP.",
    "Explain the model-view-controller pattern briefly.",
    "Describe what a binary search tree is.",
    "Explain how a relational database index works.",
]


def post(messages: list[dict[str, Any]], max_tokens: int = 1024) -> dict[str, Any]:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def health() -> dict[str, Any]:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as resp:
        return json.loads(resp.read())


def reset() -> None:
    req = urllib.request.Request(f"{BASE}/admin/reset", method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def main() -> int:
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _YELLOW = "\033[33m"
    _GREEN = "\033[32m"
    _CYAN = "\033[36m"
    _PASS_BG = "\033[42m\033[30m"
    _RESET = "\033[0m"

    def _fmt_counter(name: str, value: int, fired_color: str) -> str:
        text = f"{name}={value:<3}"
        if value == 0:
            return f"{_DIM}{text}{_RESET}"
        return f"{fired_color}{_BOLD}{text}{_RESET}"

    def _row(label_color: str, label: str, h: dict[str, Any], suffix: str = "") -> str:
        return (
            f"  {label_color}{_BOLD}{label}{_RESET} "
            f"{h['active_tokens']:>5}/{h['budget']:<5} "
            f"({h['budget_utilization']:.2f})  "
            f"{_fmt_counter('ev', h['total_evictions'], _YELLOW)}  "
            f"{_fmt_counter('rec', h['total_recoveries'], _GREEN)}"
            f"{suffix}"
        )

    reset()
    h = health()
    print(
        f"{_DIM}server: budget={h['budget']} n_ctx={h['n_ctx']} "
        f"kv_block={h['kv_block_primitives']}{_RESET}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": FACT},
    ]
    t0 = time.perf_counter()
    resp = post(messages, max_tokens=128)
    content = resp["choices"][0]["message"].get("content") or ""
    messages.append({"role": "assistant", "content": content})
    h = health()
    prev_ev, prev_rec = h["total_evictions"], h["total_recoveries"]
    print(_row(_YELLOW, "fact turn:", h, f"  asst={content[:40]!r}"))

    for i, q in enumerate(FILLERS):
        messages.append({"role": "user", "content": q})
        resp = post(messages, max_tokens=256)
        content = resp["choices"][0]["message"].get("content") or ""
        messages.append({"role": "assistant", "content": content})
        h = health()
        suffix = f"  asst={content[:40]!r}"
        if h["total_evictions"] > prev_ev and prev_ev == 0:
            suffix += f"  {_YELLOW}{_BOLD}<- eviction fires{_RESET}"
        if h["total_recoveries"] > prev_rec and prev_rec == 0:
            suffix += f"  {_GREEN}{_BOLD}<- kv_restore fires{_RESET}"
        print(_row("", f"filler {i:>2}:", h, suffix))
        prev_ev, prev_rec = h["total_evictions"], h["total_recoveries"]

    messages.append({"role": "user", "content": PROBE})
    resp = post(messages, max_tokens=128)
    answer = resp["choices"][0]["message"].get("content") or ""
    h = health()
    total = time.perf_counter() - t0
    print(_row(_CYAN, "PROBE:    ", h))
    print()
    print(f"  question: {PROBE!r}")
    if PASSKEY in answer:
        print(f"  answer:   {_GREEN}{_BOLD}{answer!r}{_RESET}")
    else:
        print(f"  answer:   {answer!r}")
    print(f"  elapsed:  {total:.1f}s")
    print()
    if PASSKEY in answer:
        print(
            f"  {_PASS_BG}{_BOLD} PASS {_RESET} "
            f"{_GREEN}{_BOLD}passkey survived through "
            f"{h['total_evictions']} evictions, "
            f"{h['total_recoveries']} recoveries{_RESET}"
        )
    else:
        print(f"  \033[41m\033[37m\033[1m FAIL {_RESET} passkey lost the session")
    return 0 if PASSKEY in answer else 1


if __name__ == "__main__":
    raise SystemExit(main())
