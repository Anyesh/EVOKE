"""Scale demonstration: EVOKE keeps a session alive past the context wall.

Everything proven so far is toy-scale (a 512-768 budget over ~1.7K tokens that would fit
VRAM anyway, so eviction pressure was artificial). This shows EVOKE doing its actual job:
a long session whose accumulated context (~60K tokens) far exceeds both the KV budget and
n_ctx, so the session CANNOT be held verbatim. The agent reads a large knowledge base
section by section, then answers questions that each require re-referencing a specific
earlier section.

Three arms over the same workload:
  evoke        tight budget + kv_restore: evicts cold sections during the read (staying within
               budget and n_ctx) and splices a needed section's KV back recompute-free on the
               re-reference. Completes the whole corpus.
  no_eviction  no budget: KV grows with the corpus and runs into n_ctx; the session dies
               mid-read at the wall. This is the failure EVOKE exists to prevent.
  no_recovery  tight budget, discard: stays within budget and n_ctx by evicting, but every
               re-reference must re-decode the section from source (the re-prefill tax).

Metrics per arm: whether it completed, where it hit the n_ctx wall, peak KV held (the VRAM
proxy), total tokens decoded (the compute proxy), recoveries, and answer correctness on the
re-reference questions.

Requires EVOKE_MODEL_PATH and LLAMA_CPP_LIB (the fork build).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

ENTITIES = [
    "Helios",
    "Borealis",
    "Cinder",
    "Dorado",
    "Everest",
    "Fathom",
    "Granite",
    "Halcyon",
    "Ironwood",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Marrow",
    "Nimbus",
    "Obsidian",
    "Pinnacle",
    "Quill",
    "Rampart",
    "Solstice",
    "Tundra",
    "Umbra",
    "Verdant",
    "Wraith",
    "Xenon",
    "Yarrow",
    "Zephyr",
]
FILLER = (
    "The subsystem coordinates background reconciliation tasks across the regional fleet, "
    "buffering work items and flushing them on a fixed cadence so downstream consumers see a "
    "stable ordering. Operators tune it through the deployment manifest and observe it through "
    "the standard telemetry pipeline; the defaults are chosen for a mid-sized cluster. "
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM = "You are a configuration lookup assistant. Answer each question with only the number."


def _section(i: int, tokens_each: int, engine: LlamaCppEngine) -> tuple[str, str, int]:
    # A self-contained knowledge-base section with one retrievable fact: the
    # service-id number assigned to a named module. The value is unguessable so a
    # parametric prior cannot supply it; the filler pads the section to length.
    name = f"{ENTITIES[i % len(ENTITIES)]}-{i:03d}"
    value = 100000 + i * 7
    head = (
        f"## Module {name}\n"
        f"Module {name} is documented in this section of the operations handbook.\n"
    )
    fact = f"FACT: the service-id of module {name} is {value}.\n"
    body = head + fact + FILLER
    while len(engine.tokenize(body)) < tokens_each:
        body += FILLER
    return body, name, value


def _config(arm: str, budget: int, n_ctx: int) -> EvokeConfig:
    if arm == "no_eviction":
        return EvokeConfig(
            max_active_tokens=n_ctx * 100,
            block_size=128,
            sink_count=0,
            high_watermark=0.999,
            low_watermark=0.99,
            recovery_mode="discard",
            position_mode="compact",
        )
    common = dict(
        max_active_tokens=budget,
        block_size=128,
        sink_count=0,
        high_watermark=0.92,
        low_watermark=0.70,
    )
    if arm == "no_recovery":
        return EvokeConfig(recovery_mode="discard", position_mode="compact", **common)
    return EvokeConfig(
        recovery_mode="kv_restore",
        position_mode="compact",
        recovery_match="identity",
        w_recovery=1.0,
        recovery_strength_init=1.0,
        recovery_decay=0.7,
        **common,
    )


def _section_key(name: str) -> str:
    return f"sec:{name}"


def _resident(mgr: EvokeManager, name: str) -> bool:
    p = f"{_section_key(name)}#"
    return any(b.key.startswith(p) for b in mgr._positions.active_blocks)


def _evicted(mgr: EvokeManager, name: str) -> list[str]:
    p = f"{_section_key(name)}#"
    return [c.key for c in mgr.get_breadcrumbs() if c.key.startswith(p)]


def run_arm(
    engine: LlamaCppEngine,
    arm: str,
    budget: int,
    n_ctx: int,
    sections: list[tuple[str, str, int]],
    questions: list[int],
) -> dict:
    engine.reset()
    mgr = EvokeManager(engine, _config(arm, budget, n_ctx))
    mgr.add_context(f"{IM_START}system\n{SYSTEM}{IM_END}\n", key="sys", pinned=True)
    q_stop = set(engine.tokenize(IM_END))
    decoded = 0
    wall_at = None
    completed = True

    for idx, (text, _name, _val) in enumerate(sections):
        try:
            mgr.add_context(text, key=_section_key(_name))
            decoded += len(engine.tokenize(text))
        except RuntimeError:
            # no_eviction overruns n_ctx here: the session hits the context wall and
            # cannot continue. This is the failure EVOKE is built to prevent.
            wall_at = idx
            completed = False
            break

    correct = 0
    answered = 0
    recoveries_for_q = 0
    samples: list[tuple[str, int, str]] = []
    if completed:
        for qi in questions:
            text, name, value = sections[qi]
            if not _resident(mgr, name):
                ek = _evicted(mgr, name)
                if ek and mgr._config.recovery_mode == "kv_restore":
                    # Restore all of the section's blocks before a single budget pass,
                    # so a multi-block section is never partially evicted mid-recovery
                    # under tight budget (which would leave it incomplete at the probe).
                    for k in ek:
                        mgr.recover(k, defer_budget=True)
                    recoveries_for_q += 1
                else:
                    mgr.add_context(text, key=_section_key(name))
                    decoded += len(engine.tokenize(text))
            try:
                mgr.add_context(
                    f"\n{IM_START}user\nWhat is the service-id of module {name}? "
                    f"Answer with only the number.{IM_END}\n{IM_START}assistant\n",
                    key=f"q{qi}",
                )
                ans = mgr.generate(
                    64,
                    stop_token_ids=q_stop,
                    think_close="</think>",
                    thinking_budget=512,
                    answer_budget=32,
                )
            except RuntimeError:
                wall_at = qi
                completed = False
                break
            answered += 1
            if str(value) in ans.replace(",", "").replace(" ", ""):
                correct += 1
            if len(samples) < 6:
                samples.append(
                    (name, value, ans.encode("ascii", "replace").decode("ascii")[:80])
                )

    stats = mgr.get_stats()
    return {
        "arm": arm,
        "completed": completed,
        "wall_at_section": wall_at,
        "peak_active": mgr.peak_active_tokens,
        "final_active": stats.active_tokens,
        "decoded": decoded,
        "recoveries": stats.total_recoveries,
        "answered": answered,
        "correct": correct,
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=32768)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--sections", type=int, default=150)
    ap.add_argument("--section-tokens", type=int, default=400)
    ap.add_argument("--questions", type=int, default=16)
    args = ap.parse_args()

    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("FAIL: set EVOKE_MODEL_PATH")
        return 1

    engine = LlamaCppEngine(model, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
    if not engine.supports_kv_block:
        print(
            "FAIL: kv_block primitives not bound -- set LLAMA_CPP_LIB to the fork build"
        )
        return 1

    sections = [_section(i, args.section_tokens, engine) for i in range(args.sections)]
    total_tokens = sum(len(engine.tokenize(t)) for t, _, _ in sections)
    # Questions target sections spread across the corpus, all read early enough that a
    # tight budget evicts them well before the probe, forcing a genuine re-reference.
    step = max(1, args.sections // (args.questions + 2))
    questions = [
        step * (k + 1) for k in range(args.questions) if step * (k + 1) < args.sections
    ]

    print(
        f"\nSCALE  corpus={total_tokens} tokens ({args.sections} sections) "
        f"budget={args.budget} n_ctx={args.n_ctx}"
    )
    rows = [
        run_arm(engine, arm, args.budget, args.n_ctx, sections, questions)
        for arm in ("evoke", "no_eviction", "no_recovery")
    ]

    print(
        f"{'arm':12s} {'done':>5s} {'wall@sec':>9s} {'peak_kv':>8s} {'decoded':>9s} "
        f"{'recov':>6s} {'correct':>9s}"
    )
    for r in rows:
        wall = "-" if r["wall_at_section"] is None else str(r["wall_at_section"])
        acc = f"{r['correct']}/{r['answered']}" if r["answered"] else "-"
        print(
            f"{r['arm']:12s} {str(r['completed']):>5s} {wall:>9s} {r['peak_active']:>8d} "
            f"{r['decoded']:>9d} {r['recoveries']:>6d} {acc:>9s}"
        )

    ev = next(r for r in rows if r["arm"] == "evoke")
    nr = next(r for r in rows if r["arm"] == "no_recovery")
    ne = next(r for r in rows if r["arm"] == "no_eviction")
    print()
    print(
        f"  peak KV: evoke {ev['peak_active']} vs no_eviction {ne['peak_active']} "
        f"({ne['peak_active'] / max(1, ev['peak_active']):.1f}x)"
    )
    print(
        f"  decode:  evoke {ev['decoded']} vs no_recovery {nr['decoded']} "
        f"({nr['decoded'] / max(1, ev['decoded']):.2f}x)"
    )
    print(
        f"  no_eviction completed={ne['completed']} "
        + ("(hit the n_ctx wall)" if not ne["completed"] else "")
    )
    for r in (ev, nr):
        print(f"\n  sample answers [{r['arm']}]:")
        for name, value, ans in r["samples"]:
            print(f"    {name} expect={value}: {ans!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
