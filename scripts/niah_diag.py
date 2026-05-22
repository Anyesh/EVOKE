"""NIAH diagnostic: one cell, full instrumentation.

Runs ONE (needle, depth, strategy, budget) cell with verbose output showing
every step of EVOKE's eviction/recovery path. Designed to answer: where does
the needle go between haystack ingestion and answer generation, and why does
smart-recovery pick the blocks it picks?

For each step the diag prints:
1. After haystack ingestion: total blocks, which block contains the needle,
   how many tokens of the needle are in that block, the block's logical span.
2. After eviction (during haystack build): the needle's status (resident,
   evicted, partially-overlapping with active region).
3. After probe decode: the query embedding (first 8 dims), the count of
   resident vs evicted blocks.
4. Smart-recovery scoring: every breadcrumb ranked by cosine similarity to
   the query, with the needle row marked, and the top-K selected for recovery.
5. After recovery: the new active block set, position of the needle (if
   recovered), what blocks the model will attend to during generation.
6. The generated answer and pass/fail.

Required env: EVOKE_MODEL_PATH, LLAMA_CPP_LIB.
Optional env: EVOKE_NIAH_NEEDLE_ID (default password), EVOKE_NIAH_DEPTH (90),
EVOKE_NIAH_STRATEGY (evoke_kv_restore), EVOKE_NIAH_BUDGET (1024),
EVOKE_NIAH_PARAGRAPHS (30), EVOKE_NIAH_SEED (0).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.embed import RetrievalEmbedder
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager
from evoke.types import ActiveBlock

from niah_bench import (
    NEEDLES,
    STRATEGIES,
    build_haystack,
    make_document,
)


def _decode_block(engine: LlamaCppEngine, block: ActiveBlock, n: int = 60) -> str:
    text = engine.detokenize(block.token_ids[:n])
    return text.replace("\n", " ").strip()


def _find_needle_blocks(
    engine: LlamaCppEngine, blocks: list[ActiveBlock], expected: str
) -> list[ActiveBlock]:
    matches: list[ActiveBlock] = []
    needle_lower = expected.lower()
    for block in blocks:
        text = engine.detokenize(block.token_ids).lower()
        if needle_lower in text:
            matches.append(block)
    return matches


def _peek_needle_in_recovery(
    mgr: EvokeManager,
    needle_keys: set[str],
    needle_expected: str,
    engine: LlamaCppEngine,
) -> list[tuple[str, str]]:
    # Inspect the recovery backend's saved/breadcrumb storage for the needle.
    # KVRestoreBackend stores SavedBlock (has token_ids); BreadcrumbBackend
    # stores only metadata + embedding. Return list of (key, where) tuples.
    found: list[tuple[str, str]] = []
    backend = mgr._recovery
    # Most reliable: any breadcrumb whose key matches a known needle key
    for crumb in mgr.get_breadcrumbs():
        if crumb.key in needle_keys:
            found.append((crumb.key, "breadcrumb_by_key"))
    # KVRestoreBackend.peek_embedding works for any stored key; but to confirm
    # the saved content includes the needle text we need to look at _saved or
    # equivalent. Probe the protected attribute defensively.
    saved_dict = getattr(backend, "_saved", None)
    if isinstance(saved_dict, dict):
        for key, sb in saved_dict.items():
            token_ids = getattr(sb, "token_ids", None)
            if token_ids is None:
                continue
            text = engine.detokenize(token_ids).lower()
            if needle_expected.lower() in text and (key, "saved_by_text") not in found:
                found.append((key, "saved_by_text"))
    return found


def _query_embedding(engine: LlamaCppEngine, n_last: int = 32) -> np.ndarray | None:
    pos = engine.next_write_pos
    if pos == 0:
        return None
    start = max(0, pos - n_last)
    embs = engine.get_embeddings(list(range(start, pos)))
    if embs is None or len(embs) == 0:
        return None
    mask = (embs != 0).any(axis=1)
    if not mask.any():
        return None
    avg = embs[mask].mean(axis=0)
    norm = float(np.linalg.norm(avg))
    if norm == 0.0:
        return None
    return avg / norm


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("set EVOKE_MODEL_PATH")
        return 1
    needle_id = os.environ.get("EVOKE_NIAH_NEEDLE_ID", "password")
    depth_pct = int(os.environ.get("EVOKE_NIAH_DEPTH", "90"))
    strategy = os.environ.get("EVOKE_NIAH_STRATEGY", "evoke_kv_restore")
    budget = int(os.environ.get("EVOKE_NIAH_BUDGET", "1024"))
    n_paragraphs = int(os.environ.get("EVOKE_NIAH_PARAGRAPHS", "30"))
    seed = int(os.environ.get("EVOKE_NIAH_SEED", "0"))

    needle = next((n for n in NEEDLES if n["id"] == needle_id), None)
    if needle is None:
        print(f"unknown needle id: {needle_id}")
        return 1
    overrides = STRATEGIES.get(strategy)
    if overrides is None:
        print(f"unknown strategy: {strategy}")
        return 1

    print("=" * 78)
    print(
        f"NIAH DIAG | needle={needle_id} depth={depth_pct}% "
        f"strategy={strategy} budget={budget}"
    )
    print(f"  expected substring: {needle['expected']!r}")
    print(f"  probe: {needle['question']!r}")
    print("=" * 78)

    engine = LlamaCppEngine(model, n_ctx=16384, n_gpu_layers=-1, verbose=False)
    engine.reset()
    config = EvokeConfig(
        max_active_tokens=budget,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
        **overrides,
    )
    attn_scorer = None
    if config.w_attention > 0 and engine.supports_kv_block:
        attn_scorer = AttentionScorer(
            engine,
            layer=config.attention_capture_layer,
            n_window=config.attention_window,
            decay=config.attention_decay,
            score_mode=config.attention_score_mode,
        )
    retrieval = None
    if config.use_retrieval_embeddings:
        retrieval = RetrievalEmbedder()
        print(f"  using retrieval embedder: {retrieval._model_name}")
    mgr = EvokeManager(
        engine,
        config,
        attention_scorer=attn_scorer,
        retrieval_embedder=retrieval,
    )

    haystack = build_haystack(n_paragraphs, seed)
    document = make_document(haystack, needle, depth_pct)
    print(f"\n[STEP 1] Adding document ({len(document)} chars) ...")
    mgr.add_context(document, key=f"doc_{needle_id}_{depth_pct}")

    active = mgr._positions.active_blocks
    stats = mgr.get_stats()
    print(
        f"  after ingestion: active_blocks={len(active)} "
        f"active_tokens={stats.active_tokens} evictions={stats.total_evictions}"
    )
    breadcrumbs = list(mgr.get_breadcrumbs())
    print(f"  breadcrumbs: {len(breadcrumbs)} (evicted blocks retained for recovery)")

    print(f"\n[STEP 2] Locating the needle ({needle['expected']!r}) in cache state ...")
    needle_active = _find_needle_blocks(engine, active, needle["expected"])
    needle_keys: set[str] = set()
    for b in needle_active:
        if b.key:
            needle_keys.add(b.key)
    needle_in_recovery = _peek_needle_in_recovery(
        mgr, needle_keys, needle["expected"], engine
    )
    print(
        f"  needle in {len(needle_active)} ACTIVE block(s), "
        f"{len(needle_in_recovery)} recovery-backend entr(ies)"
    )
    for b in needle_active:
        snippet = _decode_block(engine, b, n=80)
        print(
            f"    ACTIVE block_id={b.block_id} key={b.key!r} "
            f"pos=[{b.logical_start},{b.logical_end}) text={snippet!r}"
        )
    for key, where in needle_in_recovery:
        print(f"    RECOVERY key={key!r} where={where}")
        needle_keys.add(key)
    if not needle_keys:
        print(
            "  WARNING: needle not located in any block. Block boundary may "
            "have split the needle string across two blocks."
        )

    print(f"\n[STEP 3] Processing probe ...")
    probe = f"\n\nQuestion: {needle['question']}\nAnswer:"
    mgr.process_user_message(probe)
    stats_post_probe = mgr.get_stats()
    active_post_probe = list(mgr._positions.active_blocks)
    print(
        f"  after probe: active_blocks={len(active_post_probe)} "
        f"active_tokens={stats_post_probe.active_tokens} "
        f"evictions={stats_post_probe.total_evictions}"
    )
    needle_in_recovery_post = _peek_needle_in_recovery(
        mgr, needle_keys, needle["expected"], engine
    )
    needle_active_post = _find_needle_blocks(
        engine, active_post_probe, needle["expected"]
    )
    for key, _ in needle_in_recovery_post:
        needle_keys.add(key)
    print(
        f"  needle now in {len(needle_active_post)} ACTIVE, "
        f"{len(needle_in_recovery_post)} recovery entr(ies)"
    )

    print(f"\n[STEP 4] Smart-recovery similarity scoring ...")
    n_last = int(os.environ.get("EVOKE_NIAH_QUERY_NLAST", "8"))
    print(f"  (query embedding from last {n_last} tokens)")
    if retrieval is not None:
        # Use the retrieval embedder on the raw probe text (matches the
        # Session._compute_query_embedding short-circuit). Bypasses the
        # LM-hidden-state mean which collapses to common-mode similarity.
        query_emb = retrieval.embed(needle["question"])
        print("  (query embedding via RetrievalEmbedder on probe text)")
    else:
        query_emb = _query_embedding(engine, n_last=n_last)
    if query_emb is None:
        print("  query embedding unavailable (engine returned zeros)")
        return 1
    print(f"  query_emb (first 8 dims): {query_emb[:8]}")

    crumbs = list(mgr.get_breadcrumbs())
    if not crumbs:
        print("  no breadcrumbs to score")
    else:
        scored = []
        saved_dict = getattr(mgr._recovery, "_saved", None)
        for crumb in crumbs:
            block_emb = mgr._recovery.peek_embedding(crumb.key)
            if block_emb is None:
                sim = 0.0
            else:
                sim = float(np.dot(query_emb, block_emb))
            text = ""
            if isinstance(saved_dict, dict):
                sb = saved_dict.get(crumb.key)
                if sb is not None and getattr(sb, "token_ids", None):
                    text = (
                        engine.detokenize(sb.token_ids[:50]).replace("\n", " ").strip()
                    )
            is_needle = crumb.key in needle_keys
            scored.append((sim, crumb.key, text, is_needle))
        scored.sort(key=lambda x: x[0], reverse=True)
        k = 4
        print(
            f"  {len(scored)} breadcrumbs ranked by similarity to query "
            f"(top-{k} would be recovered):"
        )
        for rank, (sim, key, text, is_needle) in enumerate(scored, 1):
            mark_rec = "[RECOVER]" if rank <= k else "         "
            mark_needle = "(***NEEDLE***)" if is_needle else ""
            print(
                f"    {mark_rec} rank={rank:3d} sim={sim:+.4f} "
                f"key={key!r} {mark_needle}"
            )
            if rank <= max(k + 2, 10) or is_needle:
                print(f"              text={text!r}")
        if not any(is_needle for _, _, _, is_needle in scored):
            print(
                "  NOTE: no breadcrumb contains the needle; it must be in "
                "active or was never created as its own block"
            )

    print(f"\n[STEP 5] Performing actual recovery (resident-gated top-4) ...")
    resident_max = -1.0
    resident_max_block = None
    current_turn_start = mgr._current_turn_start_id
    for block in mgr._positions.active_blocks:
        if block.is_sink or block.representative_embedding is None:
            continue
        if block.block_id >= current_turn_start:
            continue
        sim = float(np.dot(query_emb, block.representative_embedding))
        if sim > resident_max:
            resident_max = sim
            resident_max_block = block
    print(f"  best resident similarity = {resident_max:+.4f}")
    if resident_max_block is not None:
        snippet = _decode_block(engine, resident_max_block, n=80)
        is_needle_block = resident_max_block.key in needle_keys
        marker = "  ***NEEDLE***" if is_needle_block else ""
        print(
            f"    best resident: block_id={resident_max_block.block_id} "
            f"key={resident_max_block.key!r}{marker}"
        )
        print(f"      text={snippet!r}")
    recovered_keys = []
    if overrides.get("recovery_mode") != "discard":
        scored_for_recover = []
        for crumb in mgr.get_breadcrumbs():
            block_emb = mgr._recovery.peek_embedding(crumb.key)
            sim = float(np.dot(query_emb, block_emb)) if block_emb is not None else 0.0
            scored_for_recover.append((sim, crumb.key))
        scored_for_recover.sort(key=lambda x: x[0], reverse=True)
        candidates = [(s, k) for s, k in scored_for_recover if s > resident_max]
        threshold = config.smart_recover_min_similarity
        if threshold > 0.0:
            candidates = [(s, k) for s, k in candidates if s >= threshold]
        print(
            f"  candidates after resident-gate: {len(candidates)} "
            f"(of {len(scored_for_recover)} breadcrumbs)"
        )
        ordered = list(candidates[:4])
        ordered.reverse()
        for sim, key in ordered:
            ok = mgr.recover(key)
            if ok:
                recovered_keys.append((key, sim))
    print(f"  recovered: {len(recovered_keys)}")
    for key, sim in recovered_keys:
        print(f"    {key!r} (sim={sim:+.4f})")

    print(f"\n[STEP 6] Final cache state before generation ...")
    final_active = list(mgr._positions.active_blocks)
    final_stats = mgr.get_stats()
    print(
        f"  active_blocks={len(final_active)} "
        f"active_tokens={final_stats.active_tokens} "
        f"total_evictions={final_stats.total_evictions} "
        f"total_recoveries={final_stats.total_recoveries}"
    )
    needle_active_final = _find_needle_blocks(engine, final_active, needle["expected"])
    print(f"  needle in {len(needle_active_final)} active block(s) at generation time")
    for b in needle_active_final:
        snippet = _decode_block(engine, b, n=80)
        print(
            f"    ACTIVE block_id={b.block_id} "
            f"pos=[{b.logical_start},{b.logical_end}) text={snippet!r}"
        )
    print("  final active block summary (positional order, first 50 chars each):")
    for b in final_active:
        text = _decode_block(engine, b, n=50)
        marker = (
            "  ***NEEDLE***"
            if needle["expected"].lower() in engine.detokenize(b.token_ids).lower()
            else ""
        )
        print(
            f"    block_id={b.block_id:4d} pos=[{b.logical_start:5d},"
            f"{b.logical_end:5d}) text={text!r}{marker}"
        )

    print(f"\n[STEP 7] Generating answer ...")
    answer = mgr.generate(64)
    answer_clean = answer.strip().replace("\n", " ")
    passed = needle["expected"].lower() in answer.lower()
    print(f"  raw answer: {answer_clean!r}")
    print(f"  expected substring: {needle['expected']!r}")
    print(f"  result: {'PASS' if passed else 'FAIL'}")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
