"""Concurrency / PCIe-bandwidth measurement for EVOKE.

Drives N parallel sessions against the same evoke_serve instance. Each
session runs an identical multi-turn workload designed to overflow the
KV budget and trigger evictions plus smart-recovery. Reports per-session
wall-clock, per-turn latency distribution, and total throughput as N
scales, so the paper can replace the reviewer's PCIe-saturation worry
with a measurement.

Required env:
  EVOKE_SERVER              http://host:port (must already be running)
  EVOKE_MODEL_NAME          logical model id (default qwen25)

Optional env:
  EVOKE_CONCURRENCY         comma list of N values, default 1,4,8,16
  EVOKE_TURNS               turns per session, default 14
  EVOKE_CONCURRENCY_JSON    output JSON path
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = os.environ.get("EVOKE_SERVER") or sys.exit("set EVOKE_SERVER")
MODEL = os.environ.get("EVOKE_MODEL_NAME", "qwen25")
LEVELS = [int(x) for x in os.environ.get("EVOKE_CONCURRENCY", "1,4,8,16").split(",")]
TURNS = int(os.environ.get("EVOKE_TURNS", "14"))
ROUNDS = int(os.environ.get("EVOKE_ROUNDS", "1"))
OUT_JSON = os.environ.get("EVOKE_CONCURRENCY_JSON")

PLANT_TURN = (
    "Quick context for later: the maximum retry limit set in config.py is 17 "
    "attempts. Now to your question:"
)

FILLERS = [
    "Explain how a transformer's attention mechanism scales with sequence length.",
    "Describe the difference between encoder-decoder and decoder-only architectures.",
    "What is the rotary position embedding scheme and why does it matter for long context?",
    "Outline how flash attention reduces the memory footprint of softmax-then-matmul.",
    "Describe the role of layer normalization in stabilizing training of deep transformers.",
    "What is mixture-of-experts and how does it relate to compute budget per token?",
    "Compare KV cache quantization at 8 bits versus 4 bits in terms of generation quality.",
    "Explain the trade-off between greedy sampling and nucleus sampling.",
    "What does grouped-query attention buy in terms of KV memory savings?",
    "Describe how speculative decoding accelerates inference.",
    "What is the role of attention sinks in long-context decoding?",
    "Explain how prefill differs from decode in cost structure.",
    "What is RAG and where does it shine versus long-context retention?",
]

PROBE = "What was the maximum retry limit I mentioned earlier?"


def _post(messages, session_id, max_tokens=64, timeout=300):
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def session_run(session_id: str) -> dict:
    history: list[dict] = []
    per_turn_s: list[float] = []
    fillers = FILLERS[: max(0, TURNS - 2)]
    transcript = [PLANT_TURN] + fillers + [PROBE]
    started = time.perf_counter()
    for msg in transcript:
        history.append({"role": "user", "content": msg})
        t0 = time.perf_counter()
        try:
            resp = _post(history, session_id, max_tokens=64)
        except urllib.error.URLError as exc:
            return {
                "session_id": session_id,
                "error": str(exc),
                "elapsed_to_error_s": time.perf_counter() - started,
                "per_turn_s": per_turn_s,
            }
        t1 = time.perf_counter()
        per_turn_s.append(t1 - t0)
        content = resp["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": content})
    final_answer = history[-1]["content"]
    return {
        "session_id": session_id,
        "wall_clock_s": time.perf_counter() - started,
        "per_turn_s": per_turn_s,
        "final_answer_preview": final_answer.strip().replace("\n", " ")[:120],
        "probe_ok": "17" in final_answer,
    }


def _pct(samples: list[float], p: float) -> float:
    if not samples:
        return float("nan")
    s = sorted(samples)
    idx = min(len(s) - 1, int(p * (len(s) - 1)))
    return s[idx]


def run_level(n: int) -> dict:
    print(f"=== concurrency N={n} (rounds={ROUNDS}) ===", flush=True)
    t0 = time.perf_counter()
    results: list[dict] = []
    round_walls: list[float] = []
    for r_idx in range(ROUNDS):
        r_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {
                ex.submit(session_run, f"cb_{n}_r{r_idx}_{i}"): i for i in range(n)
            }
            for fut in as_completed(futures):
                rec = fut.result()
                rec["round"] = r_idx
                results.append(rec)
        round_walls.append(time.perf_counter() - r_t0)
        print(f"  round {r_idx + 1}/{ROUNDS}: {round_walls[-1]:.2f}s", flush=True)
    wall = time.perf_counter() - t0

    successful = [r for r in results if "error" not in r]
    wallclocks = [r["wall_clock_s"] for r in successful]
    all_turns = [t for r in successful for t in r["per_turn_s"]]
    probes_ok = sum(1 for r in successful if r.get("probe_ok"))

    if wallclocks:
        wc_p50 = statistics.median(wallclocks)
        wc_min = min(wallclocks)
        wc_max = max(wallclocks)
    else:
        wc_p50 = wc_min = wc_max = float("nan")
    turn_p50 = _pct(all_turns, 0.50)
    turn_p95 = _pct(all_turns, 0.95)
    turn_p99 = _pct(all_turns, 0.99)
    turn_p999 = _pct(all_turns, 0.999)
    turn_max = max(all_turns) if all_turns else float("nan")

    print(
        f"  sessions ok: {len(successful)}/{n * ROUNDS}; "
        f"probe_ok: {probes_ok}/{len(successful)}; "
        f"total wall: {wall:.2f}s",
        flush=True,
    )
    print(
        f"  per-session wallclock min/p50/max: "
        f"{wc_min:.2f}s / {wc_p50:.2f}s / {wc_max:.2f}s",
        flush=True,
    )
    print(
        f"  per-turn latency n={len(all_turns)} p50/p95/p99/p999/max: "
        f"{turn_p50:.3f}s / {turn_p95:.3f}s / {turn_p99:.3f}s / "
        f"{turn_p999:.3f}s / {turn_max:.3f}s",
        flush=True,
    )
    return {
        "N": n,
        "rounds": ROUNDS,
        "total_wall_s": wall,
        "round_walls_s": round_walls,
        "n_successful": len(successful),
        "n_turn_samples": len(all_turns),
        "probes_ok": probes_ok,
        "wallclock_min_s": wc_min,
        "wallclock_p50_s": wc_p50,
        "wallclock_max_s": wc_max,
        "turn_p50_s": turn_p50,
        "turn_p95_s": turn_p95,
        "turn_p99_s": turn_p99,
        "turn_p999_s": turn_p999,
        "turn_max_s": turn_max,
        "sessions": results,
    }


def main() -> int:
    levels = []
    for n in LEVELS:
        levels.append(run_level(n))
    out = {"levels": levels, "model": MODEL, "turns_per_session": TURNS}
    if OUT_JSON:
        Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print(f"results JSON: {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
