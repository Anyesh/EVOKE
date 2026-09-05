"""Trajectory-based agentic eval for EVOKE.

Replays real multi-turn agent trajectories (SWE-agent / OpenHands traces)
through EVOKE and baseline eviction policies, scoring next-action agreement
at revisit steps instead of a single planted-fact probe. See the wiki
decision `benchmark-regime-split-and-repo-correction` for the design this
implements.

Two regimes are run as separate arms and never averaged together:

  recompute: a hibernation point's evicted content is re-supplied from the
  trajectory log itself, the way a real agent harness resends context it
  still has on its own side. Every policy converges to the same context, so
  this arm measures resume COST (tokens re-added, wall-clock), not ranking
  quality. Kill/validate thresholds do not apply here.

  discard: evicted tokens are gone for good (recovery_mode="discard", no
  identity recovery). This is where ranking quality actually shows up, and
  it is the only regime the kill/validate conditions judge.

Requires EVOKE_MODEL_PATH and LLAMA_CPP_LIB (the attention-scored baselines
need the fork's kv_block / attention-capture primitives; see agent_bench.py
for the same requirement). Requires EVOKE_TRAJECTORY_DIR pointing at a
directory of trajectory files (see `load_trajectories` for the expected
schema). Trajectory data itself (nebius/SWE-rebench-openhands-trajectories,
nebius/SWE-agent-trajectories) is not fetched by this script; download and
normalize into that directory first.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.jlens_scorer import JLensScorer
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager
from trajectory_normalizers import normalize as normalize_trajectory
from trajectory_types import Step, ToolCall, Trajectory


def load_trajectories(directory: str) -> list[Trajectory]:
    path = Path(directory)
    if not path.is_dir():
        raise RuntimeError(f"EVOKE_TRAJECTORY_DIR not a directory: {directory}")
    out: list[Trajectory] = []
    for file in sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")):
        lines = (
            file.read_text().splitlines()
            if file.suffix == ".jsonl"
            else [file.read_text()]
        )
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            raw = json.loads(line)
            traj_id = f"{file.stem}#{i}"
            out.append(normalize_trajectory(raw, traj_id))
    return out


def find_revisit_steps(
    traj: Trajectory, *, recent_window: int = 6, min_gap: int = 10
) -> list[int]:
    # Heuristic, not ground truth: a step counts as a revisit if its tool
    # call's target was referenced min_gap steps back and not touched again
    # within recent_window. Validate the resulting counts look sane (the
    # filter below requires >=3 per trajectory) before trusting scores.
    last_seen: dict[str, int] = {}
    revisits: list[int] = []
    for i, step in enumerate(traj.steps):
        if step.tool_call is None:
            continue
        target = step.tool_call.target()
        if not target:
            continue
        prior = last_seen.get(target)
        if prior is not None and (i - prior) >= min_gap:
            recently_touched = any(
                s.tool_call is not None and s.tool_call.target() == target
                for s in traj.steps[max(0, i - recent_window) : i]
            )
            if not recently_touched:
                revisits.append(i)
        last_seen[target] = i
    return revisits


def trajectory_stats(traj: Trajectory, engine: LlamaCppEngine) -> dict:
    token_count = sum(len(engine.tokenize(s.text)) for s in traj.steps) + len(
        engine.tokenize(traj.system_prompt)
    )
    tool_output_tokens = sum(
        len(engine.tokenize(s.text)) for s in traj.steps if s.role == "tool"
    )
    tool_calls = sum(1 for s in traj.steps if s.tool_call is not None)
    revisits = find_revisit_steps(traj)
    return dict(
        token_count=token_count,
        tool_output_frac=tool_output_tokens / token_count if token_count else 0.0,
        tool_calls=tool_calls,
        revisit_count=len(revisits),
    )


def filter_trajectories(
    trajectories: list[Trajectory],
    engine: LlamaCppEngine,
    *,
    min_tokens: int = 8_000,
    max_tokens: int = 24_000,
    min_revisits: int = 3,
    min_tool_output_frac: float = 0.6,
    min_tool_calls: int = 15,
) -> list[Trajectory]:
    kept = []
    for traj in trajectories:
        stats = trajectory_stats(traj, engine)
        if not (min_tokens <= stats["token_count"] <= max_tokens):
            continue
        if stats["revisit_count"] < min_revisits:
            continue
        if stats["tool_output_frac"] < min_tool_output_frac:
            continue
        if stats["tool_calls"] < min_tool_calls:
            continue
        kept.append(traj)
    return kept


_FENCE_RE = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)


def extract_action(predicted_text: str) -> str:
    # Score the command the model actually issued, not its whole response:
    # SWE-agent's environment echoes boilerplate like "(Open file: X)" after
    # most commands, which a whole-text substring search picks up regardless
    # of what was really run. Fall back to the raw text when there is no
    # fenced block to isolate an action from.
    match = _FENCE_RE.search(predicted_text)
    if not match:
        return predicted_text
    tag, block = match.group(1), match.group(2).strip()
    first_line = next((l.strip() for l in block.splitlines() if l.strip()), "")
    return f"{tag} {first_line}".strip()


def score_prediction(
    predicted_text: str, ground_truth: ToolCall
) -> tuple[bool, bool, bool]:
    gt_name, _gt_target, gt_args = ground_truth.signature_tiers()
    action = extract_action(predicted_text).lower()
    name_hit = gt_name.lower() in action
    target_hit = name_hit and ground_truth.target().lower() in action
    args_hit = target_hit and gt_args.lower() in action
    return name_hit, target_hit, args_hit


POLICIES: dict[str, dict] = {
    "no_eviction": dict(
        w_recency=1.0,
        w_coherence=0.0,
        sink_count=0,
        eviction_policy="watermark",
        high_watermark=1.0,
        low_watermark=1.0,
    ),
    "recency": dict(w_recency=1.0, w_coherence=0.0, sink_count=0),
    "h2o": dict(
        w_attention=1.0,
        w_recency=0.0,
        w_coherence=0.0,
        attention_score_mode="cumulative",
        recent_tail_protect_frac=0.1,
        eviction_policy="hard",
    ),
    "snapkv": dict(
        w_attention=1.0,
        w_recency=0.0,
        w_coherence=0.0,
        attention_score_mode="snapkv",
        snapkv_observation_window=32,
        recent_tail_protect_frac=0.05,
        eviction_policy="hard",
    ),
    "evoke_attention": dict(w_attention=0.5, w_recency=0.2, w_coherence=0.3),
    # J-lens workspace signal, the arm the paper's headline claim (fact AUC
    # 0.891 vs SnapKV 0.622 on Qwen2.5-7B) actually rests on. Same
    # hard-eviction + tail-guard shape as h2o/snapkv for a clean head-to-head.
    # Requires EVOKE_JLENS_PROBE; see agent_bench.py's "jlens" entry.
    "jlens": dict(
        w_recency=0.0,
        w_coherence=0.0,
        w_jlens=1.0,
        recent_tail_protect_frac=0.1,
        eviction_policy="hard",
    ),
}

NEEDS_KV_BLOCK = {"h2o", "snapkv", "evoke_attention", "jlens"}
NEEDS_JLENS_PROBE = {"jlens"}


@dataclass
class HibernationResult:
    step_index: int
    name_hit: bool
    target_hit: bool
    args_hit: bool
    resume_tokens: int
    resume_ms: float
    predicted_text: str = ""
    gt_name: str = ""
    gt_target: str = ""


@dataclass
class RunResult:
    policy: str
    regime: str
    traj_id: str
    hibernations: list[HibernationResult] = field(default_factory=list)


def run_trajectory(
    engine: LlamaCppEngine,
    traj: Trajectory,
    policy_name: str,
    overrides: dict,
    regime: str,
    *,
    budget_frac: float = 0.25,
) -> RunResult:
    engine.reset()
    cfg_kwargs = dict(
        max_active_tokens=131072,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
        recovery_mode="discard",
    )
    cfg_kwargs.update(overrides)
    config = EvokeConfig(**cfg_kwargs)
    attn_scorer = None
    if config.w_attention > 0 and engine.supports_kv_block:
        attn_scorer = AttentionScorer(
            engine,
            layer=config.attention_capture_layer,
            n_window=config.attention_window,
            decay=config.attention_decay,
            score_mode=config.attention_score_mode,
            snapkv_observation_window=config.snapkv_observation_window,
        )
    jlens_scorer = None
    if config.w_jlens > 0:
        probe = os.environ.get("EVOKE_JLENS_PROBE", "")
        if not probe:
            raise RuntimeError("set EVOKE_JLENS_PROBE to the probe artifact npz")
        layers_env = os.environ.get("EVOKE_JLENS_LAYERS", "")
        jlens_scorer = JLensScorer(
            engine,
            probe_path=probe,
            layers=[int(x) for x in layers_env.split(",")] if layers_env else None,
        )
    mgr = EvokeManager(
        engine, config, attention_scorer=attn_scorer, jlens_scorer=jlens_scorer
    )

    revisit_set = set(find_revisit_steps(traj))
    result = RunResult(policy=policy_name, regime=regime, traj_id=traj.traj_id)

    mgr.add_context(traj.system_prompt, "system")
    cumulative_tokens = len(engine.tokenize(traj.system_prompt))
    fed_so_far: list[Step] = []

    for i, step in enumerate(traj.steps):
        step_tokens = len(engine.tokenize(step.text))
        cumulative_tokens += step_tokens

        if i in revisit_set and step.tool_call is not None:
            budget = max(int(cumulative_tokens * budget_frac), config.block_size)
            resume_start_tokens = mgr.get_stats().active_tokens
            t0 = time.perf_counter()
            mgr.set_budget(budget)

            if regime == "recompute":
                # Overstates resume cost versus a real orchestrator (which
                # would resend only the missing diff): EvokeManager has no
                # cheap resident-key check to diff against, so this re-adds
                # everything fed so far. The discard regime, the one
                # kill/validate actually judges, is unaffected by this gap.
                for prior in fed_so_far:
                    mgr.add_context(prior.text, "gapfill")

            mgr.process_user_message(
                "What is the next action? Respond with the tool name and arguments."
            )
            prediction = mgr.generate(64)
            resume_ms = (time.perf_counter() - t0) * 1000.0
            resume_tokens = max(mgr.get_stats().active_tokens - resume_start_tokens, 0)

            name_hit, target_hit, args_hit = score_prediction(
                prediction, step.tool_call
            )
            result.hibernations.append(
                HibernationResult(
                    step_index=i,
                    name_hit=name_hit,
                    target_hit=target_hit,
                    args_hit=args_hit,
                    resume_tokens=resume_tokens,
                    resume_ms=resume_ms,
                    predicted_text=prediction,
                    gt_name=step.tool_call.name,
                    gt_target=step.tool_call.target(),
                )
            )

        mgr.add_context(step.text, f"step#{i}")
        fed_so_far.append(step)

    return result


def agreement_rate(results: list[RunResult], tier: str = "target_hit") -> float:
    hits = 0
    total = 0
    for r in results:
        for h in r.hibernations:
            total += 1
            hits += int(getattr(h, tier))
    return hits / total if total else 0.0


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    traj_dir = os.environ.get("EVOKE_TRAJECTORY_DIR")
    if not model or not traj_dir:
        print("set EVOKE_MODEL_PATH and EVOKE_TRAJECTORY_DIR")
        return 1

    engine = LlamaCppEngine(model, n_ctx=32768, n_gpu_layers=-1, verbose=False)
    print(f"trajectory bench | model={Path(model).stem}")
    print(f"kv_block primitives available: {engine.supports_kv_block}")

    raw_trajectories = load_trajectories(traj_dir)
    trajectories = filter_trajectories(raw_trajectories, engine)
    print(
        f"loaded {len(raw_trajectories)} trajectories, {len(trajectories)} pass the filter"
    )
    max_trajectories = os.environ.get("EVOKE_MAX_TRAJECTORIES", "")
    if max_trajectories:
        trajectories = trajectories[: int(max_trajectories)]
        print(f"capped to {len(trajectories)} trajectories for this run")
    if not trajectories:
        print(
            "no trajectories passed the filter; loosen filter_trajectories "
            "thresholds or check the source dataset covers this length/complexity range"
        )
        return 1

    policy_filter = os.environ.get("EVOKE_POLICIES", "")
    selected_policies = (
        [p.strip() for p in policy_filter.split(",") if p.strip()]
        if policy_filter
        else list(POLICIES)
    )
    regime_filter = os.environ.get("EVOKE_REGIMES", "")
    selected_regimes = (
        [r.strip() for r in regime_filter.split(",") if r.strip()]
        if regime_filter
        else ["discard", "recompute"]
    )

    dump_path = os.environ.get("EVOKE_DUMP_PREDICTIONS", "")
    dump_records: list[dict] = []

    try:
        for regime in selected_regimes:
            print(f"\n=== regime: {regime} ===")
            for policy_name in selected_policies:
                overrides = POLICIES[policy_name]
                if policy_name in NEEDS_KV_BLOCK and not engine.supports_kv_block:
                    print(f"{policy_name:<16} SKIP (no LLAMA_CPP_LIB)")
                    continue
                if policy_name in NEEDS_JLENS_PROBE and not os.environ.get(
                    "EVOKE_JLENS_PROBE"
                ):
                    print(f"{policy_name:<16} SKIP (no EVOKE_JLENS_PROBE)")
                    continue
                results = [
                    run_trajectory(engine, traj, policy_name, overrides, regime)
                    for traj in trajectories
                ]
                if dump_path:
                    for r in results:
                        for h in r.hibernations:
                            dump_records.append(
                                dict(
                                    regime=regime,
                                    policy=policy_name,
                                    traj_id=r.traj_id,
                                    step_index=h.step_index,
                                    predicted_text=h.predicted_text,
                                    gt_name=h.gt_name,
                                    gt_target=h.gt_target,
                                    name_hit=h.name_hit,
                                    target_hit=h.target_hit,
                                    args_hit=h.args_hit,
                                )
                            )
                total_resume_tokens = sum(
                    h.resume_tokens for r in results for h in r.hibernations
                )
                total_resume_ms = sum(
                    h.resume_ms for r in results for h in r.hibernations
                )
                print(
                    f"{policy_name:<16} name={agreement_rate(results, 'name_hit'):.3f} "
                    f"target={agreement_rate(results, 'target_hit'):.3f} "
                    f"args={agreement_rate(results, 'args_hit'):.3f} "
                    f"resume_tok={total_resume_tokens} resume_ms={total_resume_ms:.1f}"
                )

        if not os.environ.get("EVOKE_SKIP_CEILING"):
            print("\n=== self-agreement ceiling (no_eviction x2, discard regime) ===")
            ceiling_runs = [
                [
                    run_trajectory(
                        engine, traj, "no_eviction", POLICIES["no_eviction"], "discard"
                    )
                    for traj in trajectories
                ]
                for _ in range(2)
            ]
            ceiling_a = agreement_rate(ceiling_runs[0], "target_hit")
            ceiling_b = agreement_rate(ceiling_runs[1], "target_hit")
            print(
                f"seed 1: {ceiling_a:.3f}  seed 2: {ceiling_b:.3f}  "
                f"ceiling: {(ceiling_a + ceiling_b) / 2:.3f}"
            )
    finally:
        engine.close()

    if dump_path:
        with open(dump_path, "w") as f:
            for rec in dump_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nwrote {len(dump_records)} predictions to {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
