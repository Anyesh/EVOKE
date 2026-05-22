"""Host-RAM pressure run for KVRestoreBackend.

Runs a long single-session bench with `kv_restore_ram_budget_bytes` set tight
so the LRU saved-block pool fires repeatedly, then plots the degradation curve:
saved-in-RAM, spilled-to-disk, breadcrumb-only, fully discarded. Demonstrates
graceful degradation rather than asserting it.

Default workload: 20-fact multi-fact session at 8000-token haystack with
budget 1024 (forces ~60 evictions over the session), RAM budget set to hold
only 8 blocks worth of K/V (forces ~52 LRU drops); spill path enabled so the
demoted bytes fall back to disk rather than vanishing.

Environment:
- EVOKE_HRM_PARAGRAPHS         haystack length (default 80)
- EVOKE_HRM_FACTS              number of planted facts (default 20)
- EVOKE_HRM_BUDGET             KV budget (default 1024)
- EVOKE_HRM_RAM_BLOCKS         RAM budget in saved-blocks (default 8)
- EVOKE_HRM_SPILL_PATH         disk spill dir (default C:/tmp/evoke_spill)
- EVOKE_HRM_JSON               output JSON
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.embed import RetrievalEmbedder
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

from niah_bench import _smart_recover, _think_close_for, build_haystack


# Generic planted-fact template; values are unique per index so a 20-fact run
# has 20 distinct strings to recover. The plant text is intentionally short
# (~25 tokens) so a single block usually contains exactly one fact.
_PLANT_TEMPLATE = (
    "Operational record #{idx}: the secure code for asset {asset} is "
    "{code}. Treat as confidential and rotate quarterly."
)


def _gen_plant(idx: int, seed: int) -> tuple[str, str, str]:
    rng = random.Random(seed * 10000 + idx)
    asset = f"alpha-{rng.randint(100, 999)}"
    code = f"{rng.choice(['BLUE', 'RED', 'GREEN', 'GOLD', 'BLACK'])}-{rng.randint(10, 99)}-{rng.choice(['ALPHA', 'BETA', 'GAMMA', 'DELTA'])}"
    return (
        _PLANT_TEMPLATE.format(idx=idx, asset=asset, code=code),
        f"What is the secure code for asset {asset}?",
        code,
    )


@dataclass
class PoolSnapshot:
    turn: int
    cached_blocks: int
    evictions: int
    recoveries: int
    saved_in_ram: int
    spilled_to_disk: int
    breadcrumb_only: int
    lru_drops: int
    spill_drops: int
    active_tokens: int


@dataclass
class HostRamResult:
    seed: int
    budget: int
    ram_budget_bytes: int | None
    spill_path: str | None
    n_facts: int
    snapshots: list[PoolSnapshot] = field(default_factory=list)
    fact_recall: dict[int, bool] = field(default_factory=dict)


def _snapshot(mgr: EvokeManager, turn: int) -> PoolSnapshot:
    backend = mgr._recovery
    saved = len(getattr(backend, "_saved", {}))
    spilled = len(getattr(backend, "_spilled", {}))
    lru = int(getattr(backend, "_lru_evictions", 0))
    spill = int(getattr(backend, "_spill_evictions", 0))
    crumbs = backend.list_evicted()
    breadcrumb_only = max(0, len(crumbs) - saved - spilled)
    stats = mgr.get_stats()
    return PoolSnapshot(
        turn=turn,
        cached_blocks=stats.active_blocks,
        evictions=stats.total_evictions,
        recoveries=stats.total_recoveries,
        saved_in_ram=saved,
        spilled_to_disk=spilled,
        breadcrumb_only=breadcrumb_only,
        lru_drops=lru,
        spill_drops=spill,
        active_tokens=stats.active_tokens,
    )


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("set EVOKE_MODEL_PATH")
        return 1
    n_paragraphs = int(os.environ.get("EVOKE_HRM_PARAGRAPHS", "80"))
    n_facts = int(os.environ.get("EVOKE_HRM_FACTS", "20"))
    budget = int(os.environ.get("EVOKE_HRM_BUDGET", "1024"))
    ram_blocks = int(os.environ.get("EVOKE_HRM_RAM_BLOCKS", "8"))
    spill_path = os.environ.get("EVOKE_HRM_SPILL_PATH", r"C:\tmp\spill")
    seed = int(os.environ.get("EVOKE_HRM_SEED", "0"))
    out_json = os.environ.get("EVOKE_HRM_JSON")

    # 64-token block_size; bytes-per-block for Qwen 2.5 7B is 64 cells *
    # 28 layers * 2 (K and V) * 512 n_embd_kv_gqa * 2 bytes = 3.67 MiB.
    # ram_blocks=8 -> ~29 MiB budget. This is intentionally tight to force
    # the LRU pool to fire on most evictions.
    bytes_per_block = 64 * 28 * 2 * 512 * 2
    ram_budget_bytes = ram_blocks * bytes_per_block

    haystack = build_haystack(n_paragraphs, seed)
    facts = [_gen_plant(i, seed) for i in range(n_facts)]

    # Insert planted facts evenly spread through the haystack.
    insertion_step = max(1, n_paragraphs // (n_facts + 1))
    pieces: list[str] = []
    fact_iter = iter(enumerate(facts))
    next_fact_idx, next_fact = next(fact_iter, (None, None))
    for i, paragraph in enumerate(haystack):
        pieces.append(paragraph)
        if (
            next_fact_idx is not None
            and (i + 1) % insertion_step == 0
            and next_fact_idx < n_facts
        ):
            pieces.append(next_fact[0])
            try:
                next_fact_idx, next_fact = next(fact_iter)
            except StopIteration:
                next_fact_idx = None

    document = "\n\n".join(pieces)

    engine = LlamaCppEngine(model, n_ctx=16384, n_gpu_layers=-1, verbose=False)
    config_kwargs: dict = dict(
        max_active_tokens=budget,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
        recovery_mode="kv_restore",
        use_retrieval_embeddings=True,
        w_attention=0.5,
        w_recency=0.2,
        w_coherence=0.3,
        kv_restore_ram_budget_bytes=ram_budget_bytes,
        kv_restore_spill_path=spill_path,
    )
    config = EvokeConfig(**config_kwargs)
    retrieval = RetrievalEmbedder()
    attn = (
        AttentionScorer(
            engine,
            layer=config.attention_capture_layer,
            n_window=config.attention_window,
            decay=config.attention_decay,
            score_mode=config.attention_score_mode,
        )
        if engine.supports_kv_block
        else None
    )
    mgr = EvokeManager(
        engine, config, attention_scorer=attn, retrieval_embedder=retrieval
    )

    print(
        f"hostram run | model={Path(model).stem} budget={budget} "
        f"ram_blocks={ram_blocks} (bytes={ram_budget_bytes:,}) "
        f"spill_path={spill_path!r}"
    )
    print(
        f"facts={n_facts} paragraphs={n_paragraphs} block_size=64 "
        f"bytes_per_block={bytes_per_block:,}"
    )

    result = HostRamResult(
        seed=seed,
        budget=budget,
        ram_budget_bytes=ram_budget_bytes,
        spill_path=spill_path,
        n_facts=n_facts,
    )

    t0 = time.perf_counter()
    mgr.add_context(document, key=f"hrm_doc_s{seed}")
    snap = _snapshot(mgr, turn=0)
    result.snapshots.append(snap)
    print(
        f"\nAfter document ingest: cached={snap.cached_blocks} evict={snap.evictions} "
        f"saved={snap.saved_in_ram} spilled={snap.spilled_to_disk} "
        f"crumb-only={snap.breadcrumb_only} lru_drops={snap.lru_drops} "
        f"spill_drops={snap.spill_drops}"
    )

    think_close = _think_close_for(os.environ.get("EVOKE_MODEL_PATH", ""))
    print(
        f"\n{'turn':<5}{'probe':<7}{'cached':<8}{'evict':<7}{'recov':<7}"
        f"{'saved':<7}{'spilled':<8}{'crumb':<7}{'lru':<6}{'spill':<6}"
    )
    print("-" * 75)
    for turn, (plant_text, question, expected) in enumerate(facts, 1):
        probe = f"\n\nQuestion: {question}\nAnswer:"
        mgr._last_user_text = probe
        _smart_recover(mgr, k=config.smart_recover_k)
        mgr.process_user_message(probe)
        if think_close:
            answer = mgr.generate(
                512, think_close=think_close, thinking_budget=2048, answer_budget=256
            )
        else:
            answer = mgr.generate(128)
        passed = expected.lower() in answer.lower()
        result.fact_recall[turn] = passed
        snap = _snapshot(mgr, turn=turn)
        result.snapshots.append(snap)
        mark = "PASS" if passed else "fail"
        print(
            f"{turn:<5}{mark:<7}{snap.cached_blocks:<8}{snap.evictions:<7}"
            f"{snap.recoveries:<7}{snap.saved_in_ram:<7}"
            f"{snap.spilled_to_disk:<8}{snap.breadcrumb_only:<7}"
            f"{snap.lru_drops:<6}{snap.spill_drops:<6}"
        )

    elapsed = time.perf_counter() - t0
    passes = sum(1 for v in result.fact_recall.values() if v)
    print(
        f"\nDONE in {elapsed:.1f}s. recall={passes}/{n_facts}; "
        f"final pool: saved={result.snapshots[-1].saved_in_ram} "
        f"spilled={result.snapshots[-1].spilled_to_disk} "
        f"crumb-only={result.snapshots[-1].breadcrumb_only}; "
        f"total lru_drops={result.snapshots[-1].lru_drops} "
        f"total spill_drops={result.snapshots[-1].spill_drops}"
    )

    engine.close()
    if out_json:
        Path(out_json).write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8"
        )
        print(f"\nresults JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
