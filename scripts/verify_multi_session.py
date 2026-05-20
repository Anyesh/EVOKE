"""Verify multi-session server: two distinct sessions, swap-correct state.

Drives the server with two different X-EVOKE-Session ids interleaved.
Each session plants its own fact and probes it later, verifying that
swapping engine state between sessions preserves each session's KV cache
identity (no cross-contamination of facts, no session eats the other's
context).

Required env: EVOKE_SERVER (e.g. http://192.0.2.10:8000). Server must
already be running.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("EVOKE_SERVER") or sys.exit("set EVOKE_SERVER")
MODEL = os.environ.get("EVOKE_MODEL_NAME", "qwen25")


def post(
    messages: list[dict], session_id: str, max_tokens: int = 64
) -> tuple[str, dict]:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-EVOKE-Session": session_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read())
    content = payload["choices"][0]["message"].get("content") or ""
    return content, payload


def list_sessions() -> dict:
    with urllib.request.urlopen(f"{BASE}/v1/sessions", timeout=10) as resp:
        return json.loads(resp.read())


def health(session_id: str) -> dict:
    req = urllib.request.Request(
        f"{BASE}/health",
        headers={"X-EVOKE-Session": session_id},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> int:
    print("Initial pool state:")
    print(json.dumps(list_sessions(), indent=2))

    # Session ALPHA: planted code "AURORA-7"
    msgs_a: list[dict] = [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": "Important: remember the codeword AURORA-7. Confirm.",
        },
    ]
    resp_a, _ = post(msgs_a, session_id="alpha")
    print(f"\n[alpha] turn 1: {resp_a[:80]!r}")
    msgs_a.append({"role": "assistant", "content": resp_a})

    # Session BETA: planted code "BOREAS-3"
    msgs_b: list[dict] = [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": "Important: remember the codeword BOREAS-3. Confirm.",
        },
    ]
    resp_b, _ = post(msgs_b, session_id="beta")
    print(f"[beta]  turn 1: {resp_b[:80]!r}")
    msgs_b.append({"role": "assistant", "content": resp_b})

    print(f"\nAfter both first turns: {list_sessions()}")

    # Alpha asks for its own codeword. State should swap back to alpha.
    msgs_a.append(
        {
            "role": "user",
            "content": "What was the codeword I gave you? Reply with just the codeword.",
        }
    )
    resp_a2, _ = post(msgs_a, session_id="alpha", max_tokens=24)
    print(f"\n[alpha] probe: {resp_a2!r}")

    # Beta asks for ITS codeword.
    msgs_b.append(
        {
            "role": "user",
            "content": "What was the codeword I gave you? Reply with just the codeword.",
        }
    )
    resp_b2, _ = post(msgs_b, session_id="beta", max_tokens=24)
    print(f"[beta]  probe: {resp_b2!r}")

    alpha_ok = "AURORA-7" in resp_a2 or "AURORA-7" in resp_a2.upper()
    beta_ok = "BOREAS-3" in resp_b2 or "BOREAS-3" in resp_b2.upper()
    alpha_clean = "BOREAS" not in resp_a2.upper()
    beta_clean = "AURORA" not in resp_b2.upper()

    print(f"\nResults:")
    print(f"  alpha recalls AURORA-7:  {'PASS' if alpha_ok else 'FAIL'}")
    print(f"  beta recalls BOREAS-3:   {'PASS' if beta_ok else 'FAIL'}")
    print(f"  alpha has no BOREAS leak: {'PASS' if alpha_clean else 'FAIL'}")
    print(f"  beta has no AURORA leak:  {'PASS' if beta_clean else 'FAIL'}")

    pool = list_sessions()
    print(f"\nFinal pool state:\n{json.dumps(pool, indent=2)}")

    if all([alpha_ok, beta_ok, alpha_clean, beta_clean]):
        print("\nPASS: multi-session swap preserves per-session KV identity")
        return 0
    print("\nFAIL: cross-contamination or recall failure")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
