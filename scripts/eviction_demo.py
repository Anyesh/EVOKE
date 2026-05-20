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
    reset()
    h = health()
    print(
        f"server: budget={h['budget']} n_ctx={h['n_ctx']} kv_block={h['kv_block_primitives']}"
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
    print(
        f"  fact turn: {h['active_tokens']:>5}/{h['budget']:<5} "
        f"({h['budget_utilization']:.2f})  ev={h['total_evictions']:<3} "
        f"rec={h['total_recoveries']:<3}  asst={content[:40]!r}"
    )

    for i, q in enumerate(FILLERS):
        messages.append({"role": "user", "content": q})
        resp = post(messages, max_tokens=256)
        content = resp["choices"][0]["message"].get("content") or ""
        messages.append({"role": "assistant", "content": content})
        h = health()
        print(
            f"  filler {i:>2}:  {h['active_tokens']:>5}/{h['budget']:<5} "
            f"({h['budget_utilization']:.2f})  ev={h['total_evictions']:<3} "
            f"rec={h['total_recoveries']:<3}  asst={content[:40]!r}"
        )

    messages.append({"role": "user", "content": PROBE})
    resp = post(messages, max_tokens=128)
    answer = resp["choices"][0]["message"].get("content") or ""
    h = health()
    total = time.perf_counter() - t0
    print(
        f"  PROBE:     {h['active_tokens']:>5}/{h['budget']:<5} "
        f"({h['budget_utilization']:.2f})  ev={h['total_evictions']:<3} "
        f"rec={h['total_recoveries']:<3}"
    )
    print(f"\n  question: {PROBE!r}")
    print(f"  answer:   {answer!r}")
    print(f"  elapsed:  {total:.1f}s")
    print(
        f"\n  {'PASS' if PASSKEY in answer else 'FAIL'}: passkey {'survived' if PASSKEY in answer else 'lost'} the session"
    )
    return 0 if PASSKEY in answer else 1


if __name__ == "__main__":
    raise SystemExit(main())
