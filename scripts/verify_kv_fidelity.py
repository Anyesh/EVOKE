"""Strong fidelity test for the kv_restore recovery path on a pure-attention model.

The existing verify_kv_restore.py only proves a needle is *attendable* after restore.
A high-salience passkey survives even a degraded splice, so needle recall is necessary
but not sufficient for the fidelity claim. This test asks the stronger question: does a
restored block reproduce the model's behavior, measured by greedy continuation against a
run where the block was never disturbed.

Four arms over identical content, greedy (argmax) decode so the comparison is exact:

  REF      block never evicted
  SPARSE   block evicted then restored IN PLACE at its original position (zero RoPE shift)
  COMPACT  block evicted then restored AT THE TAIL (survivors recompacted, RoPE re-anchored)
  DISCARD  block evicted, never restored (sensitivity control)

Verdict:
  - SPARSE must equal REF token-for-token. The in-place splice changes no position, so any
    divergence means the saved K/V bytes themselves do not round-trip. This is the linchpin.
  - DISCARD must diverge from REF, otherwise the probe is insensitive to the block and the
    test proves nothing.
  - COMPACT vs REF is reported, not gated: it measures whether relocating a faithful block to
    a new position (new RoPE phase, V unchanged) preserves behavior, which is the
    position-bias question the project has flagged.

Requires EVOKE_MODEL_PATH (a pure-attention GGUF, e.g. Qwen2.5-7B) and LLAMA_CPP_LIB
pointing at the fork build with the kv_block primitives.
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

BACKGROUND = "General project background and routine status notes. " * 40
TARGET = (
    "Internal migration memo. The migration is led by Priya Nadkarni. "
    "The cutover date is the fourteenth of November. The rollback owner is "
    "Tomas Eklund. The budget ceiling is 4.2 million dollars. Treat these as authoritative."
)
FILLER = "Unrelated implementation detail about the build pipeline. " * 60
PROBE = (
    "\n\nQuestion: From the internal migration memo, who leads the migration, "
    "what is the cutover date, who owns the rollback, and what is the budget ceiling?\nAnswer:"
)

MODES = ("ref", "sparse", "compact", "discard")


def _greedy_ids(engine: LlamaCppEngine, n: int) -> list[int]:
    eos = engine.eos_token
    out: list[int] = []
    for _ in range(n):
        if engine.next_write_pos + 1 >= engine.n_ctx:
            break
        tok = engine.generate_next()
        out.append(tok)
        if tok == eos:
            break
    return out


def _config_for(mode: str) -> EvokeConfig:
    # Budget set far above the content so the only eviction is the explicit
    # force_evict below; otherwise _enforce_budget would drop blocks on its own
    # and the arms would no longer be comparable.
    return EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=64,
        recovery_mode="discard" if mode == "discard" else "kv_restore",
        position_mode="sparse" if mode == "sparse" else "compact",
    )


def run_arm(engine: LlamaCppEngine, mode: str, n_gen: int) -> list[int]:
    engine.reset()
    mgr = EvokeManager(engine, _config_for(mode))
    mgr.load_document(BACKGROUND)
    mgr.add_context(TARGET, key="target")
    mgr.add_context(FILLER, key="filler")

    if mode != "ref":
        target_blocks = [
            b for b in mgr._positions.active_blocks if b.key.startswith("target#")
        ]
        if not target_blocks:
            raise SystemExit("FAIL: target block not found after add_context")
        mgr.force_evict([b.block_id for b in target_blocks])
        if mode in ("sparse", "compact"):
            recovered = 0
            for crumb in mgr.get_breadcrumbs():
                if crumb.key.startswith("target#") and mgr.recover(crumb.key):
                    recovered += 1
            if recovered == 0:
                raise SystemExit(f"FAIL[{mode}]: recover() returned False for target")

    mgr.process_user_message(PROBE)
    return _greedy_ids(engine, n_gen)


def _match_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--gen-tokens", type=int, default=48)
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
    print("kv_block primitives bound\n")

    results: dict[str, list[int]] = {}
    for mode in MODES:
        results[mode] = run_arm(engine, mode, args.gen_tokens)
        text = (
            engine.detokenize(results[mode]).encode("ascii", "replace").decode("ascii")
        )
        print(f"{mode:8s}: {text[:140]!r}")

    ref = results["ref"]
    print(f"\ngen_tokens = {args.gen_tokens}")
    for mode in ("sparse", "compact", "discard"):
        m = _match_prefix(results[mode], ref)
        exact = results[mode] == ref
        print(f"{mode:8s} vs ref: prefix_match={m}/{len(ref)} exact={exact}")

    sparse_faithful = results["sparse"] == ref
    sensitive = results["discard"] != ref
    compact_faithful = results["compact"] == ref

    print()
    if sparse_faithful and sensitive:
        print(
            "FIDELITY PASS: in-place restore reproduces REF token-for-token "
            "(the saved K/V round-trips), and the probe is sensitive (DISCARD diverges)."
        )
        print(
            f"RELOCATION: compact-restore {'matches' if compact_faithful else 'DIVERGES from'} "
            "REF -- "
            + (
                "moving a faithful block to a new position preserved behavior."
                if compact_faithful
                else "moving a faithful block to a new position changed behavior "
                "(position-dependent, as expected)."
            )
        )
        return 0
    if sparse_faithful and not sensitive:
        print(
            "INCONCLUSIVE: SPARSE==REF but DISCARD==REF too -- the probe does not depend on "
            "the target block, so it cannot confirm fidelity. Strengthen TARGET/PROBE."
        )
        return 1
    print(
        "FIDELITY FAIL: in-place restore (SPARSE) diverges from REF -- the saved K/V does not "
        "round-trip through evict+restore, so recompute-free recovery is not faithful as built."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
