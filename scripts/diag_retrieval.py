from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager
from evoke.scorer import cosine_similarity

MODEL = str(
    Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

FILLER = (
    "This is a passage about various topics. The weather today is partly cloudy "
    "with a chance of rain in the afternoon. Markets showed mixed performance "
    "with technology stocks leading gains while energy shares declined. "
    "Scientists have discovered a new species of deep-sea fish in the Pacific "
    "Ocean that displays bioluminescent properties. "
)


def run_diagnostic(block_size: int = 32):
    print(f"=== Retrieval diagnostic (block_size={block_size}) ===\n")

    engine = LlamaCppEngine(MODEL, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    filler = FILLER * 60
    needle = "The secret password for the vault is CRYSTALLINE-HORIZON-42."
    words = filler.split()
    insert_idx = int(len(words) * 0.5)
    words.insert(insert_idx, needle)
    doc = " ".join(words)

    config = EvokeConfig(
        max_active_tokens=1024,
        block_size=block_size,
        w_recency=0.4,
        w_sink=1.0,
        w_coherence=0.6,
        retrieval_threshold=0.85,
        max_retrieve_blocks=4,
        demotion_policy="watermark",
        high_watermark=0.95,
        low_watermark=0.75,
    )

    mgr = EvokeManager(engine, config)
    mgr.load_document(doc)

    stats = mgr.get_stats()
    print(
        f"After load: active={stats.active_tokens}, archive={stats.archive_tokens}, "
        f"active_blocks={stats.active_blocks}, archive_blocks={stats.archive_blocks}"
    )
    print(f"Demotions so far: {stats.total_demotions}\n")

    archive_blocks = mgr._archive.all_blocks()
    needle_blocks = []
    for ab in archive_blocks:
        if "CRYSTALLINE" in ab.text or "password" in ab.text or "vault" in ab.text:
            needle_blocks.append(ab)
            print(
                f"NEEDLE archive block id={ab.block_id} "
                f"orig=[{ab.pos_start},{ab.pos_end}) "
                f"size={ab.size}"
            )
            print(f"  text: {ab.text[:120]!r}")
            print()

    if not needle_blocks:
        active_blocks = mgr._positions.active_blocks
        for ab in active_blocks:
            text = engine.detokenize(ab.token_ids)
            if "CRYSTALLINE" in text or "password" in text or "vault" in text:
                print(
                    f"NEEDLE still ACTIVE block id={ab.block_id} "
                    f"orig=[{ab.original_start},{ab.original_end})"
                )
                print(f"  text: {text[:120]!r}")
                print()

    query = "What is the secret password for the vault?"
    mgr.process_user_message(query)

    stats_after = mgr.get_stats()
    print(
        f"After query: active={stats_after.active_tokens}, archive={stats_after.archive_tokens}"
    )
    print(f"Promotions: {stats_after.total_promotions}")
    print(f"Demotions: {stats_after.total_demotions}\n")

    query_emb = mgr._scorer._recent_embedding
    if query_emb is None:
        print("ERROR: no query embedding computed")
        engine.close()
        return

    remaining_archive = mgr._archive.all_blocks()
    if remaining_archive:
        sims = []
        for ab in remaining_archive:
            sim = cosine_similarity(query_emb, ab.representative_embedding)
            sims.append((sim, ab))
        sims.sort(key=lambda x: x[0], reverse=True)

        print(f"Archive similarity distribution (n={len(sims)}):")
        all_s = [s for s, _ in sims]
        print(
            f"  min={min(all_s):.4f}  max={max(all_s):.4f}  "
            f"mean={np.mean(all_s):.4f}  std={np.std(all_s):.4f}\n"
        )

        print("Top 10 archive blocks by similarity to query:")
        for sim, ab in sims[:10]:
            has_needle = "CRYSTALLINE" in ab.text or "password" in ab.text
            marker = " *** NEEDLE ***" if has_needle else ""
            print(
                f"  sim={sim:.4f}  id={ab.block_id}  "
                f"orig=[{ab.pos_start},{ab.pos_end})  "
                f"text={ab.text[:80]!r}{marker}"
            )
        print()

        for ab in needle_blocks:
            if ab.block_id in {b.block_id for _, b in sims}:
                needle_sim = next(s for s, b in sims if b.block_id == ab.block_id)
                rank = (
                    next(
                        i for i, (_, b) in enumerate(sims) if b.block_id == ab.block_id
                    )
                    + 1
                )
                print(
                    f"Needle block id={ab.block_id}: sim={needle_sim:.4f}, rank={rank}/{len(sims)}"
                )
    else:
        print("Archive is empty (all blocks promoted or still active)")

    promoted_events = [e for e in mgr.get_event_log() if e.event_type == "promotion"]
    if promoted_events:
        print(f"\nPromoted block IDs: {[e.block_ids for e in promoted_events]}")
        for e in promoted_events:
            for bid in e.block_ids:
                active = [b for b in mgr._positions.active_blocks if b.block_id == bid]
                if active:
                    text = engine.detokenize(active[0].token_ids)
                    has_needle = "CRYSTALLINE" in text or "password" in text
                    marker = " *** NEEDLE ***" if has_needle else ""
                    print(f"  Promoted id={bid}: {text[:100]!r}{marker}")

    answer = mgr.generate(64)
    print(f"\nGenerated answer: {answer!r}")

    engine.close()


if __name__ == "__main__":
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    run_diagnostic(bs)
