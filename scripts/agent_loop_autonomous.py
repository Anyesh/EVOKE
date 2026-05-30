"""Autonomous agent loop: the MODEL drives the reads, EVOKE makes re-reads recompute-free.

The scripted demo (agent_loop_stateful.py) proved the mechanism; a script decided the
re-references. Here a real thinking LLM runs a tool loop over EvokeManager: it emits
`read_file:` / `final:` actions and the loop executes them against the KV-cache working
memory. The session is held in the cache; each file is decoded once as a keyed block;
the budget evicts cold files; and when the agent reads a file it loaded earlier, EVOKE
splices that file's KV back by identity (recompute-free) instead of re-decoding it.

The protocol is structured (a scaffold, like a real agent's system prompt), not a free
autonomous monologue: gather, then re-read the source, then answer. Two findings from the
free-form version drove this shape and are recorded honestly: (1) a 14B model under memory
pressure does NOT spontaneously re-reference evicted content; (2) once it commits to a wrong
value in a prior turn, re-reading afterward (recovered KV or a fresh re-decode) does not
un-anchor it. So the value is requested only AFTER config.py is back in memory, and the
re-read is the model's own read_file action; EVOKE just makes that read free.

Three arms on the same protocol:
  evoke        sparse + kv_restore + tight budget: in-budget, the re-read served from saved KV.
  no_eviction  budget = n_ctx: correct but peak active = the whole working set (over budget).
  no_recovery  discard + tight budget: in-budget, but the re-read must re-decode the file.

Reads the Tasklet repo in scripts/demo_webapp/ (config.py holds MAX_TODOS_PER_USER=17,
SESSION_TIMEOUT_MINUTES=45). Requires EVOKE_MODEL_PATH and LLAMA_CPP_LIB (the fork).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "demo_webapp"
FILES = ["config.py", "models.py", "storage.py", "app.py", "README.md"]
EXPLORE_FILES = ["config.py", "storage.py", "models.py", "app.py"]
SOURCE = "config.py"
EXPECT = ("17", "45")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

SYSTEM = (
    "You are a code assistant with a small working memory; old files may be dropped to "
    "save space, so you can only rely on a file you currently hold. Act one step at a "
    "time. To read a file respond with exactly:\n"
    "  read_file: <filename>\n"
    "To finish respond with exactly:\n"
    "  final: <answer>\n"
    "Emit exactly one action per turn and never invent file contents."
)

EXPLORE_TASK = (
    "Repository 'Tasklet' files: config.py, models.py, storage.py, app.py, README.md.\n"
    "First, explore the code: read config.py, then storage.py, then models.py, then "
    "app.py (one read_file per turn) to learn how the app enforces its limits.\n"
    "When you have read those four files, respond with: final: done"
)

REREAD_PROMPT = (
    "config.py is no longer guaranteed to be in your working memory, and a remembered "
    "value cannot be trusted. Respond with exactly this line and nothing else:\n"
    "read_file: config.py"
)

ANSWER_PROMPT = (
    "Now report, using the config.py you are holding: the exact numeric value of "
    "MAX_TODOS_PER_USER and of SESSION_TIMEOUT_MINUTES, and which other files use these "
    "limits.\nRespond with: final: MAX_TODOS_PER_USER=<n>, SESSION_TIMEOUT_MINUTES=<n>, "
    "used in <files>."
)

READ_RE = re.compile(r"(?im)^\s*read_file:\s*([A-Za-z0-9_./-]+)")
FINAL_RE = re.compile(r"(?is)final:\s*(.+)")


def _read(name: str) -> str:
    return (PROJECT / name).read_text(encoding="utf-8")


def _config(arm: str, budget: int, n_ctx: int) -> EvokeConfig:
    if arm == "no_eviction":
        return EvokeConfig(
            max_active_tokens=n_ctx,
            block_size=64,
            sink_count=0,
            high_watermark=0.999,
            low_watermark=0.99,
            recovery_mode="discard",
            position_mode="compact",
        )
    common = dict(
        max_active_tokens=budget,
        block_size=64,
        sink_count=0,
        high_watermark=0.92,
        low_watermark=0.70,
    )
    if arm == "no_recovery":
        return EvokeConfig(recovery_mode="discard", position_mode="sparse", **common)
    # EVOKE recovers into a contiguous (compact) layout: eviction recompacts and
    # recovery re-anchors the block to the tail, so the position axis carries no
    # holes. The sparse (in-place, ArkVale-like) variant is kept as the
    # evoke_sparse ablation because under many evictions its holey layout degrades
    # the model's attention to recovered content (reads MAX_TODOS but misses
    # SESSION_TIMEOUT), measured on Qwen3-14B while the bytes round-trip exactly.
    position_mode = "sparse" if arm == "evoke_sparse" else "compact"
    return EvokeConfig(
        recovery_mode="kv_restore",
        position_mode=position_mode,
        recovery_match="identity",
        w_recovery=1.0,
        recovery_strength_init=1.0,
        recovery_protect_threshold=0.0,
        recovery_decay=0.7,
        **common,
    )


def _file_key(name: str) -> str:
    return f"file:{name}"


def _is_resident(mgr: EvokeManager, name: str) -> bool:
    prefix = f"{_file_key(name)}#"
    return any(b.key.startswith(prefix) for b in mgr._positions.active_blocks)


def _evicted_keys(mgr: EvokeManager, name: str) -> list[str]:
    prefix = f"{_file_key(name)}#"
    return [c.key for c in mgr.get_breadcrumbs() if c.key.startswith(prefix)]


def _user_turn(body: str) -> str:
    return f"\n{IM_START}user\n{body}{IM_END}\n{IM_START}assistant\n"


def _emit_read(
    mgr: EvokeManager, engine: LlamaCppEngine, step: int, name: str
) -> tuple[int, str]:
    # Bring `name` into the working memory and feed the matching chat turn. The file
    # text is decoded at most once per (re)load: resident and recovered reads feed only
    # a short note (the content is already in the KV), so a recovered read is genuinely
    # recompute-free and the decode counter is honest.
    if name not in FILES:
        mgr.add_context(
            _user_turn(f"<error>no such file: {name}</error>"), key=f"wrap:{step}"
        )
        return 0, "missing"
    if _is_resident(mgr, name):
        mgr.add_context(
            _user_turn(f'<file name="{name}"> (already in working memory)</file>'),
            key=f"wrap:{step}",
        )
        return 0, "resident"
    evicted = _evicted_keys(mgr, name)
    if evicted and mgr._config.recovery_mode == "kv_restore":
        for k in evicted:
            mgr.recover(k)
        mgr.add_context(
            _user_turn(f'<file name="{name}"> (reloaded into working memory)</file>'),
            key=f"wrap:{step}",
        )
        return 0, "recovered"
    content = _read(name)
    mgr.add_context(f'\n{IM_START}user\n<file name="{name}">\n', key=f"wrap:{step}a")
    mgr.add_context(content, key=_file_key(name))
    mgr.add_context(f"\n</file>{IM_END}\n{IM_START}assistant\n", key=f"wrap:{step}b")
    return len(engine.tokenize(content)), ("reread" if evicted else "first")


def _parse_action(text: str) -> tuple[str, str]:
    tail = text.split("</think>")[-1]
    fin = FINAL_RE.search(tail)
    rd = READ_RE.search(tail)
    if fin and (not rd or fin.start() < rd.start()):
        return "final", fin.group(1).split(IM_END)[0].strip()
    if rd:
        return "read", rd.group(1).strip()
    return "none", tail.strip()


def run_arm(
    engine: LlamaCppEngine,
    arm: str,
    budget: int,
    n_ctx: int,
    gen: int,
    max_steps: int,
    think_close: str | None,
) -> dict:
    engine.reset()
    mgr = EvokeManager(engine, _config(arm, budget, n_ctx))
    trace: list[dict] = []
    context_decoded = 0

    def step_once(idx: int) -> tuple[str, str]:
        out = mgr.generate(
            gen, stop_token_ids=stop_ids, think_close=think_close, answer_budget=gen
        )
        return _parse_action(out)

    def do_read(idx: int, name: str, phase: int) -> None:
        nonlocal context_decoded
        decoded, status = _emit_read(mgr, engine, idx, name)
        context_decoded += decoded
        trace.append(
            {
                "action": f"read:{name}",
                "status": status,
                "decoded": decoded,
                "phase": phase,
            }
        )

    stop_ids = set(engine.tokenize(IM_END))

    preamble = (
        f"{IM_START}system\n{SYSTEM}{IM_END}\n"
        f"{IM_START}user\n{EXPLORE_TASK}{IM_END}\n"
        f"{IM_START}assistant\n"
    )
    mgr.add_context(preamble, key="preamble", pinned=True)

    idx = 0
    read_seen: set[str] = set()
    # Phase 1: the model explores by reading files (its own choices). No value is asked
    # yet, so it cannot commit to a stale one. Stops once the required files are read so
    # a wider budget cannot let the model wander into extra files and shift the eviction
    # pattern (which made the budget-768 run degenerate).
    for _ in range(max_steps):
        kind, payload = step_once(idx)
        if kind == "read":
            do_read(idx, payload, 1)
            if payload in EXPLORE_FILES:
                read_seen.add(payload)
        elif kind == "final":
            trace.append({"action": "explored"})
            break
        else:
            trace.append({"action": "noop", "raw": payload[:60]})
            mgr.add_context(
                _user_turn("Use read_file: <name> or final: <answer>."),
                key=f"nudge:{idx}",
            )
        idx += 1
        if read_seen.issuperset(EXPLORE_FILES):
            trace.append({"action": "explored"})
            break

    # Phase 2: re-read the source. The re-read is the model's read_file action; EVOKE
    # serves it from saved KV (evoke) or forces a re-decode (no_recovery). Resident in
    # no_eviction. Capped retries so a stubborn model cannot stall the run.
    mgr.add_context(_user_turn(REREAD_PROMPT), key="reread_prompt")
    for _ in range(4):
        kind, payload = step_once(idx)
        idx += 1
        if kind == "read":
            do_read(idx, payload, 2)
            if payload == SOURCE:
                break
        else:
            mgr.add_context(
                _user_turn(
                    f"Your only valid action now is exactly: read_file: {SOURCE}"
                ),
                key=f"reread_nudge:{idx}",
            )

    # Phase 3: ask for the value, only now that config.py is back in memory.
    mgr.add_context(_user_turn(ANSWER_PROMPT), key="answer_prompt")
    final_answer = ""
    for _ in range(3):
        kind, payload = step_once(idx)
        idx += 1
        if kind == "final":
            final_answer = payload
            trace.append({"action": "final"})
            break
        if kind == "read":
            do_read(idx, payload, 3)
        else:
            mgr.add_context(
                _user_turn("Give your answer now: final: ..."),
                key=f"answer_nudge:{idx}",
            )

    stats = mgr.get_stats()
    correct = all(t in final_answer for t in EXPECT)
    recovered_reads = sum(1 for t in trace if t.get("status") == "recovered")
    reread_decode = sum(
        t.get("decoded", 0)
        for t in trace
        if t.get("action") == f"read:{SOURCE}" and t.get("phase", 1) >= 2
    )
    safe = final_answer.encode("ascii", "replace").decode("ascii")[:200]
    return {
        "arm": arm,
        "budget": budget,
        "peak_active": mgr.peak_active_tokens,
        "final_active": stats.active_tokens,
        "context_decoded": context_decoded,
        "source_reread_decode": reread_decode,
        "evictions": stats.total_evictions,
        "recoveries": stats.total_recoveries,
        "model_recovered_reads": recovered_reads,
        "correct": correct,
        "answer": safe,
        "trace": trace,
    }


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1
    n_ctx = int(os.environ.get("LOOP_N_CTX", "8192"))
    budget = int(os.environ.get("LOOP_BUDGET", "512"))
    gen = int(os.environ.get("LOOP_GEN", "512"))
    max_steps = int(os.environ.get("LOOP_MAX_STEPS", "10"))
    tc_env = os.environ.get("EVOKE_THINK_CLOSE", "</think>")
    think_close = None if tc_env.lower() in ("", "none") else tc_env

    engine = LlamaCppEngine(model, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print("FAIL: kv_block primitives not bound -- set LLAMA_CPP_LIB")
        return 1

    rows = [
        run_arm(engine, arm, budget, n_ctx, gen, max_steps, think_close)
        for arm in ("evoke", "evoke_sparse", "no_eviction", "no_recovery")
    ]

    print(
        f"\nAUTONOMOUS  budget={budget} n_ctx={n_ctx}  "
        f"(codebase = {sum(len(engine.tokenize(_read(f))) for f in FILES)} tokens)"
    )
    print(
        f"{'arm':12s} {'final':>6s} {'peak':>6s} {'over':>5s} {'ctx_dec':>8s} "
        f"{'src_red':>8s} {'evict':>6s} {'recov':>6s} {'mdl_rec':>8s} {'correct':>8s}"
    )
    for r in rows:
        over = r["final_active"] > r["budget"]
        print(
            f"{r['arm']:12s} {r['final_active']:>6d} {r['peak_active']:>6d} {str(over):>5s} "
            f"{r['context_decoded']:>8d} {r['source_reread_decode']:>8d} {r['evictions']:>6d} "
            f"{r['recoveries']:>6d} {r['model_recovered_reads']:>8d} {str(r['correct']):>8s}"
        )
    print()
    for r in rows:
        actions = " -> ".join(
            t["action"] + (f"[{t['status']}]" if t.get("status") else "")
            for t in r["trace"]
        )
        print(f"  [{r['arm']}] {actions}")
        print(f"  [{r['arm']}] answer: {r['answer']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
