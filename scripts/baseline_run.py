from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE = os.environ.get("EVOKE_SERVER", "http://HOST:8000")
MODEL_NAME = os.environ.get("EVOKE_MODEL_NAME", "qwen25")
PASSKEY = "4242"
FACT = (
    f"Important system note: my favorite number is {PASSKEY}. "
    "Please remember this for the rest of our conversation."
)
PROBE = (
    "Earlier in this conversation I told you my favorite number. "
    "What was it? Reply with just the number."
)
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


def post(messages):
    body = json.dumps(
        {"model": MODEL_NAME, "messages": messages, "max_tokens": 256}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def health():
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as resp:
        return json.loads(resp.read())


def reset():
    req = urllib.request.Request(f"{BASE}/admin/reset", method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def main():
    reset()
    h = health()
    print(f"budget={h['budget']} n_ctx={h['n_ctx']}")
    messages = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": FACT},
    ]
    t0 = time.perf_counter()
    r = post(messages)
    messages.append(
        {
            "role": "assistant",
            "content": r["choices"][0]["message"].get("content") or "",
        }
    )
    h = health()
    print(
        f"  fact: active={h['active_tokens']} ev={h['total_evictions']}"
        f" rec={h['total_recoveries']}"
    )
    for i, q in enumerate(FILLERS):
        messages.append({"role": "user", "content": q})
        r = post(messages)
        messages.append(
            {
                "role": "assistant",
                "content": r["choices"][0]["message"].get("content") or "",
            }
        )
        h = health()
        print(
            f"  q{i:>2}:  active={h['active_tokens']} ev={h['total_evictions']}"
            f" rec={h['total_recoveries']}"
        )
    messages.append({"role": "user", "content": PROBE})
    r = post(messages)
    answer = r["choices"][0]["message"].get("content") or ""
    h = health()
    elapsed = time.perf_counter() - t0
    probe_ok = PASSKEY in answer
    print(
        f"RESULT budget={h['budget']} "
        f"probe={'PASS' if probe_ok else 'FAIL'} "
        f"ev={h['total_evictions']} rec={h['total_recoveries']} "
        f"active={h['active_tokens']} elapsed={elapsed:.1f}s"
    )
    print(f"answer: {answer.strip()[:80]!r}")
    return 0 if probe_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
