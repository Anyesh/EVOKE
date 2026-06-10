"""Agent-codebase demo driver for the EVOKE live server (arm-agnostic).

Simulates an agent reading the Tasklet webapp (scripts/demo_webapp/) file by file
over a session that exceeds the KV budget, then asking for a value that lives only
in the first file read (config.py: MAX_TODOS_PER_USER=17, SESSION_TIMEOUT_MINUTES=45).

The same trace is run against three server configs to build the budget-vs-decode
contrast (the server policy, not this script, sets the arm):
  no_eviction        -- cheap decode but active_tokens blows past the budget
  evict, no recovery -- stays in budget but re-decodes re-sent evicted content
  EVOKE identity     -- stays in budget AND splices re-referenced blocks recompute-free

Reports peak active_tokens (budget compliance), cumulative tokens actually decoded
(recompute cost), recovery counts, and answer correctness. Correctness is held equal
across arms (the client re-sends everything, so nobody loses content); the evidence
is the budget x decode contrast.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ.get("EVOKE_SERVER") or sys.exit(
    "set EVOKE_SERVER to http://host:port"
)
MODEL = os.environ.get("EVOKE_MODEL_NAME", "qwen3-14b")
ARM = os.environ.get("EVOKE_DEMO_ARM", "?")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(HERE, "demo_webapp")

# config.py first so it is the oldest block and the prime eviction target; the
# probe asks for values defined only there.
FILE_ORDER = ["config.py", "models.py", "storage.py", "app.py", "README.md"]
PROBE = (
    "Looking back at the Tasklet codebase you read earlier, what is the numeric "
    "value of MAX_TODOS_PER_USER and of SESSION_TIMEOUT_MINUTES, and which file "
    "defines them? Answer in one sentence."
)
EXPECT = ("17", "45")


def post(messages: list[dict], max_tokens: int) -> dict:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def health() -> dict:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as resp:
        return json.loads(resp.read())


def reset() -> None:
    urllib.request.urlopen(
        urllib.request.Request(f"{BASE}/admin/reset", method="POST"), timeout=10
    ).read()


def read_file(rel: str) -> str:
    with open(os.path.join(PROJECT, rel), encoding="utf-8") as fh:
        return fh.read()


def main() -> int:
    reset()
    h = health()
    print(
        f"[{ARM}] budget={h['budget']} n_ctx={h['n_ctx']} kv_block={h['kv_block_primitives']}"
    )
    messages: list[dict] = [
        {
            "role": "system",
            "content": "You are a coding assistant reviewing a repository. Acknowledge each file in a few words.",
        }
    ]
    peak_active = 0
    for rel in FILE_ORDER:
        content = read_file(rel)
        messages.append(
            {
                "role": "user",
                "content": f"Here is `{rel}` from the Tasklet repo:\n```\n{content}\n```",
            }
        )
        resp = post(messages, max_tokens=16)
        ack = resp["choices"][0]["message"].get("content") or ""
        messages.append({"role": "assistant", "content": ack})
        h = health()
        peak_active = max(peak_active, h["active_tokens"])
        print(
            f"  read {rel:12s} active={h['active_tokens']:>5}/{h['budget']:<5} "
            f"ev={h['total_evictions']:<3} rec={h['total_recoveries']:<3} "
            f"id_rec={h['identity_recovered']:<3} id_miss={h['identity_mismatch']:<3}"
        )

    messages.append({"role": "user", "content": PROBE})
    resp = post(messages, max_tokens=2048)
    answer = resp["choices"][0]["message"].get("content") or ""
    h = health()
    # Server-side peak (high-water mark before any trim) captures the transient
    # during-turn full-working-set spike; the client only sees post-turn /health.
    peak_active = max(peak_active, h.get("peak_active", 0), h["active_tokens"])
    correct = all(tok in answer for tok in EXPECT)
    decoded = h["total_new_decoded"]
    seen = h["total_prompt_tokens"]
    safe = answer.encode("ascii", "replace").decode("ascii")

    print(
        f"\n[{ARM}] RESULT  peak_active={peak_active}/{h['budget']}  "
        f"over_budget={peak_active > h['budget']}  decoded={decoded}  prompt_seen={seen}  "
        f"evictions={h['total_evictions']}  recoveries={h['total_recoveries']}  "
        f"identity_recovered={h['identity_recovered']}  identity_mismatch={h['identity_mismatch']}"
    )
    print(f"[{ARM}] answer: {safe[:280]!r}")
    print(f"[{ARM}] facts_present(17 & 45)={correct}")
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
