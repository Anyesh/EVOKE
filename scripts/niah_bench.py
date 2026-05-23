"""Needle-in-a-Haystack benchmark for EVOKE.

Standard long-context evaluation: a distinctive fact (the needle) is planted
at a known depth inside a long synthetic haystack, the model is asked a probe
question matching the needle, and the answer is scored for the expected
substring. Run across every EVOKE policy and budget; pass-rate per cell
isolates policy-quality on long-context recall under budget pressure.

The haystack is generated deterministically from a seed so the bench is
reproducible from the repo alone (no external corpus download). Topics and
sentence templates are diverse enough that the model engages with the text
rather than treating it as boilerplate.

Requires EVOKE_MODEL_PATH. LLAMA_CPP_LIB is required for the kv_restore,
evoke_attention, and h2o strategies (the attention-capture primitive).

Environment overrides:
- EVOKE_BUDGETS=512,1024,2048
- EVOKE_NIAH_NEEDLES=password,capital (subset of needle ids)
- EVOKE_NIAH_DEPTHS=5,25,50,75,95
- EVOKE_NIAH_PARAGRAPHS=40 (haystack size)
- EVOKE_NIAH_SEED=0
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.embed import RetrievalEmbedder
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager


TOPICS = [
    "the migratory patterns of the spotted ironwing albatross",
    "fermented tea cultivation in the highland terraces of southwestern China",
    "the geometry of fan vaulting in late Gothic ecclesiastical architecture",
    "polymer chemistry of high-temperature silicone elastomers",
    "the slow erosion of basalt columns along volcanic coastlines",
    "the manufacture of cuneiform tablets in Mesopotamian scribal academies",
    "echolocation strategies of insectivorous bats in dense forest canopy",
    "the typography of medieval illuminated manuscripts and their guild traditions",
    "deep-sea hydrothermal vent communities and chemosynthetic metabolism",
    "the harmonic theory of Baroque keyboard counterpoint",
    "alpine glacier retreat and the formation of proglacial lakes",
    "ceramic glaze formulation in Song dynasty kilns",
    "the propagation of stress waves through granular soils",
    "lichenometry as a dating technique on exposed rock surfaces",
    "the social organization of leafcutter ant colonies and their fungal gardens",
    "celestial navigation techniques in pre-instrument Polynesian voyaging",
    "the structural mechanics of compressed-earth vaulted ceilings",
    "fluorescence spectroscopy of rare-earth-doped phosphors",
    "the cultivation of saffron crocus and the harvest of its dried stigmas",
    "tidal predictions and harmonic analysis of coastal estuaries",
    "the design of pre-electric mechanical music boxes and cylinder organs",
    "permafrost dynamics and the release of trapped methane in warming peatlands",
    "the calligraphic conventions of classical Japanese cursive scripts",
    "Bayesian inference in the analysis of historical climate proxies",
    "the orbital mechanics of binary star systems with eccentric companions",
    "the chemistry of natural indigo extraction and vat fermentation",
    "ethnomusicology of the West African kora and its tuning traditions",
    "the metallurgy of pattern-welded blades in early medieval Europe",
    "the population dynamics of solitary bees in fragmented agricultural landscapes",
    "epigraphy of votive inscriptions on Roman provincial altars",
]

SENT_TEMPLATES = [
    "{topic} has attracted considerable scholarly interest over the past few decades.",
    "Researchers have documented at least {count} distinct sub-patterns within {topic}.",
    "Recent advances in instrumentation have changed how investigators approach {topic}.",
    "The history of {topic} is intertwined with developments in {discipline}.",
    "Several monographs from the {era} survey early empirical work on {topic}.",
    "Specialised {tool} is now standard equipment in any serious study of {topic}.",
    "Field expeditions investigating {topic} often combine remote sensing with on-site sampling.",
    "Comparative analyses across regions suggest that {topic} varies more than was initially assumed.",
    "A series of {count} workshops in the past five years has consolidated terminology around {topic}.",
    "Practitioners frequently cite the {era} as the period when {topic} reached a mature theoretical footing.",
    "The {discipline} community continues to debate how best to model {topic} under stress.",
    "Long-term datasets on {topic} are unusually scarce given the topic's broader importance.",
    "Open questions about {topic} include the influence of episodic disturbance and recovery timescales.",
    "Cross-disciplinary collaborations between {discipline} and computational modelling now drive most progress on {topic}.",
    "Conservation implications of {topic} have shifted policy priorities in several jurisdictions.",
]

DISCIPLINES = [
    "materials science",
    "ecology",
    "biophysics",
    "art history",
    "geochemistry",
    "computational linguistics",
    "mathematical biology",
    "structural engineering",
    "historical metallurgy",
    "marine ecology",
]

ERAS = [
    "the 1960s",
    "the post-war decades",
    "the late nineteenth century",
    "the early modern period",
    "the 1980s",
    "the inter-war years",
    "the early Renaissance",
    "the Edwardian period",
]

TOOLS = [
    "mass spectrometry",
    "synchrotron diffraction",
    "acoustic telemetry",
    "isotopic labelling",
    "high-resolution microscopy",
    "ground-penetrating radar",
    "X-ray fluorescence",
    "stable-isotope analysis",
]

# Each needle has a distinct topic, an unambiguous expected substring, and a
# probe phrasing that should cue retrieval even after eviction. Expected
# substrings are designed so a model that genuinely retrieved the needle
# produces them verbatim; a model that hallucinates will produce something
# else.
NEEDLES = [
    {
        "id": "password",
        "text": (
            "Important administrative note: the secret password for the "
            "laboratory vault is icarus-pinwheel-43. Personnel are reminded "
            "not to write it on shared notepads."
        ),
        "question": "What is the secret password for the laboratory vault?",
        "expected": "icarus-pinwheel-43",
    },
    {
        "id": "capital",
        "text": (
            "Geographic footnote for the next briefing: the capital city of "
            "the small inland country of Polopia is the historic settlement "
            "of Zithrand, located near the western foothills."
        ),
        "question": "What is the capital city of the country called Polopia?",
        "expected": "Zithrand",
    },
    {
        "id": "amount",
        "text": (
            "Lab record: Dr. Halvorsen's calibration run on the spectrometer "
            "used exactly 47 milligrams of cesium chloride dissolved in "
            "buffered ammonium acetate. Results pending second-pass analysis."
        ),
        "question": "How many milligrams of cesium chloride did Dr. Halvorsen use in the calibration run?",
        "expected": "47",
    },
    {
        "id": "code",
        "text": (
            "Operational note for the duty officer: the activation code for "
            "the orbital relay station is BLUE-MOUNTAIN-7-DELTA. Confirm "
            "receipt before the next handover window."
        ),
        "question": "What is the activation code for the orbital relay station?",
        "expected": "BLUE-MOUNTAIN-7-DELTA",
    },
    {
        "id": "date",
        "text": (
            "Archival entry: the Treaty of Vrenholm was signed on the "
            "twenty-third of October, 1786, at a quarter past four in the "
            "afternoon, in the upper hall of the merchants' guild."
        ),
        "question": "On what date and time was the Treaty of Vrenholm signed?",
        "expected": "1786",
    },
]

DEFAULT_DEPTHS = [5, 25, 50, 75, 95]

STRATEGIES: dict[str, dict] = {
    # Full-context baseline: eviction never fires because the budget cap is
    # bigger than any document the bench can produce. Used as the apples-to-
    # apples partner for kv_quant runs ("keep everything at quarter precision"
    # vs evoke's "keep a quarter of tokens at full precision").
    "no_eviction": dict(
        w_recency=1.0,
        w_coherence=0.0,
        sink_count=0,
        recovery_mode="discard",
        max_active_tokens=131072,
        eviction_policy="watermark",
        high_watermark=1.0,
        low_watermark=1.0,
    ),
    "recency": dict(
        w_recency=1.0, w_coherence=0.0, sink_count=0, recovery_mode="discard"
    ),
    "streaming_llm": dict(
        w_recency=1.0, w_coherence=0.0, sink_count=4, recovery_mode="discard"
    ),
    "evoke_discard": dict(recovery_mode="discard"),
    "evoke_breadcrumb": dict(recovery_mode="breadcrumb"),
    "evoke_kv_restore": dict(recovery_mode="kv_restore", use_retrieval_embeddings=True),
    # Recovery-aware eviction variant: same selection rule as evoke_kv_restore
    # but the scorer weighs per-block recovery_strength (set on recover, decayed
    # per turn via tick_turn) at w_recovery=1.0. Closes the recover-then-evict
    # thrash that the session-length sweep diagnosed at T=28+. Multifact compares
    # this entry against plain evoke_kv_restore in the same JSON so any pass-rate
    # delta is attributable to the protection mechanism alone (selection,
    # recovery primitive, retrieval embedder all unchanged).
    "evoke_recovery_aware": dict(
        recovery_mode="kv_restore",
        use_retrieval_embeddings=True,
        w_recovery=1.0,
        recovery_decay=0.7,
        recovery_strength_init=1.0,
    ),
    "evoke_attention": dict(
        recovery_mode="kv_restore",
        w_attention=0.5,
        w_recency=0.2,
        w_coherence=0.3,
        use_retrieval_embeddings=True,
    ),
    "h2o": dict(
        recovery_mode="discard",
        w_attention=1.0,
        w_recency=0.0,
        w_coherence=0.0,
        attention_score_mode="cumulative",
        recent_tail_protect_frac=0.1,
        eviction_policy="hard",
    ),
    # SnapKV baseline (Liu et al., NeurIPS 2024) reimplemented on top of
    # EVOKE's AttentionScorer + manager snapshot hook. Per-block importance is
    # the sum of softmax attention from the last `snapkv_observation_window`
    # tokens of the most recent user message to that block; the score is
    # frozen at the end of process_user_message and the eviction pass keeps
    # the top-K blocks by that snapshot for the rest of the turn. No recovery
    # (SnapKV has no concept of bringing evicted tokens back), hard eviction
    # at the budget so the cache settles to SnapKV's compressed footprint
    # before generation begins, and a 5% recent-tail guard so the just-added
    # observation window is never evicted by its own score.
    "snapkv": dict(
        recovery_mode="discard",
        w_attention=1.0,
        w_recency=0.0,
        w_coherence=0.0,
        attention_score_mode="snapkv",
        snapkv_observation_window=32,
        recent_tail_protect_frac=0.05,
        eviction_policy="hard",
    ),
    # InfLLM baseline (Xiao et al., NeurIPS 2024) adapted onto EVOKE's
    # kv_restore + smart-recovery infrastructure. The original paper masks
    # attention to a top-K block selection per decode step, keeping every
    # block resident in an external memory segment. We adapt that to a
    # cache-resident model: aggressive eviction (only sinks and a 25%
    # local-window tail stay resident) plus K=8 query-based smart recovery
    # at each user-message boundary, with recovered blocks spliced back via
    # kv_block_load rather than attended-to-from-elsewhere. The substantive
    # InfLLM choices — block-level external memory, similarity-based block
    # retrieval, larger K than typical heavy-hitter policies — are
    # preserved; the per-decode-step retrieval rhythm of the original is
    # approximated by the per-turn boundary (a reasonable approximation for
    # NIAH / multifact / agent where one user message yields one short
    # answer). bge-small retrieval embeddings stand in for InfLLM's
    # attention-weighted representative-token pooling. The policy weights
    # are zeroed out so retrieval is purely similarity-driven (no recency
    # or coherence priors) — sink protection and the local-window guard
    # carry the resident-set logic.
    "infllm": dict(
        recovery_mode="kv_restore",
        w_recency=0.0,
        w_coherence=0.0,
        w_attention=0.0,
        sink_count=4,
        smart_recover_k=8,
        use_retrieval_embeddings=True,
        block_embedding_strategy="mean",
        eviction_policy="watermark",
        high_watermark=0.5,
        low_watermark=0.3,
        recent_tail_protect_frac=0.25,
    ),
}


def build_haystack(n_paragraphs: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    paragraphs: list[str] = []
    for _ in range(n_paragraphs):
        topic = rng.choice(TOPICS)
        n_sentences = rng.randint(4, 7)
        sentences: list[str] = []
        for _ in range(n_sentences):
            template = rng.choice(SENT_TEMPLATES)
            sentences.append(
                template.format(
                    topic=topic,
                    count=rng.randint(3, 24),
                    discipline=rng.choice(DISCIPLINES),
                    tool=rng.choice(TOOLS),
                    era=rng.choice(ERAS),
                )
            )
        paragraphs.append(" ".join(sentences))
    return paragraphs


def make_document(haystack: list[str], needle: dict, depth_pct: int) -> str:
    pos = int(len(haystack) * depth_pct / 100)
    inserted = haystack[:pos] + [needle["text"]] + haystack[pos:]
    return "\n\n".join(inserted)


@dataclass
class NiahResult:
    needle_id: str
    depth: int
    strategy: str
    budget: int
    probe_ok: bool
    evictions: int
    recoveries: int
    active_tokens: int
    elapsed: float
    answer: str


# Module-level retrieval embedder so the bge-small model loads once per
# bench run rather than once per cell (~30 cells × ~5 s warm-up otherwise).
_RETRIEVAL_EMBEDDER = RetrievalEmbedder()


def _build_scorer(
    engine: LlamaCppEngine, config: EvokeConfig
) -> AttentionScorer | None:
    if config.w_attention <= 0 or not engine.supports_kv_block:
        return None
    return AttentionScorer(
        engine,
        layer=config.attention_capture_layer,
        n_window=config.attention_window,
        decay=config.attention_decay,
        score_mode=config.attention_score_mode,
        snapkv_observation_window=config.snapkv_observation_window,
    )


def _think_close_for(model_path: str) -> str | None:
    name = Path(model_path).name.lower()
    if "qwen3" in name or "qwen-3" in name:
        return "</think>"
    return None


def _smart_recover(mgr: EvokeManager, k: int = 4) -> int:
    # Mirrors Session._smart_recover: after the probe is decoded, score the
    # evicted blocks by embedding similarity against the freshest n_last=32
    # tokens (the probe) and recover the top-K most coherent. Without this
    # step the bench would not exercise the recovery story that the EVOKE
    # production server runs on every user turn, and every policy with
    # recovery_mode != "discard" would silently behave like discard.
    crumbs = list(mgr.get_breadcrumbs())
    if not crumbs:
        return 0
    if mgr._retrieval_embedder is not None and mgr._last_user_text:
        # Retrieval-embedder path: query embedding comes from the raw probe
        # text and lives in the same 384-dim bge-small space as the block
        # embeddings. Mixing this path with the LM-hidden-state path below
        # would cross-dim-cosine-crash.
        query_emb = mgr._retrieval_embedder.embed(mgr._last_user_text)
    else:
        pos = mgr._engine.next_write_pos
        if pos == 0:
            return 0
        n_last = 32
        start = max(0, pos - n_last)
        try:
            embs = mgr._engine.get_embeddings(list(range(start, pos)))
        except (NotImplementedError, RuntimeError):
            return 0
        if embs is None or len(embs) == 0:
            return 0
        nonzero_mask = (embs != 0).any(axis=1)
        if not nonzero_mask.any():
            return 0
        avg = embs[nonzero_mask].mean(axis=0)
        norm = float(np.linalg.norm(avg))
        if norm == 0.0:
            return 0
        query_emb = avg / norm
    scored: list[tuple[float, str]] = []
    for crumb in crumbs:
        block_emb = mgr._recovery.peek_embedding(crumb.key)
        if block_emb is None:
            scored.append((0.0, crumb.key))
        else:
            scored.append((float(np.dot(query_emb, block_emb)), crumb.key))
    scored.sort(key=lambda x: x[0], reverse=True)
    resident_max = -1.0
    current_turn_start = mgr._current_turn_start_id
    for block in mgr._positions.active_blocks:
        if block.is_sink or block.representative_embedding is None:
            continue
        if block.block_id >= current_turn_start:
            continue
        sim = float(np.dot(query_emb, block.representative_embedding))
        if sim > resident_max:
            resident_max = sim
    scored = [(s, key) for s, key in scored if s > resident_max]
    threshold = mgr._config.smart_recover_min_similarity
    if threshold > 0.0:
        scored = [(s, key) for s, key in scored if s >= threshold]
    ordered = list(scored[:k])
    ordered.reverse()
    recovered = 0
    for _, key in ordered:
        if mgr.recover(key):
            recovered += 1
    return recovered


def run_cell(
    engine: LlamaCppEngine,
    needle: dict,
    depth_pct: int,
    strategy: str,
    overrides: dict,
    budget: int,
    n_paragraphs: int,
    seed: int,
) -> NiahResult:
    engine.reset()
    # Build config kwargs in two stages so per-ablation overrides can carry
    # block_size (or any other default) without colliding with a hardcoded
    # keyword (Python: TypeError: got multiple values for keyword arg).
    config_kwargs: dict = dict(
        max_active_tokens=budget,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
    )
    config_kwargs.update(overrides)
    attn_layer_env = os.environ.get("EVOKE_ATTN_LAYER")
    if attn_layer_env is not None and "w_attention" in config_kwargs:
        config_kwargs["attention_capture_layer"] = int(attn_layer_env)
    config = EvokeConfig(**config_kwargs)
    attn_scorer = _build_scorer(engine, config)
    retrieval = _RETRIEVAL_EMBEDDER if config.use_retrieval_embeddings else None
    mgr = EvokeManager(
        engine,
        config,
        attention_scorer=attn_scorer,
        retrieval_embedder=retrieval,
    )

    haystack = build_haystack(n_paragraphs, seed)
    document = make_document(haystack, needle, depth_pct)
    mgr.add_context(document, key=f"doc_{needle['id']}_{depth_pct}")

    probe = f"\n\nQuestion: {needle['question']}\nAnswer:"
    t0 = time.perf_counter()
    if overrides.get("recovery_mode") != "discard":
        # Recover BEFORE decoding the probe so recovered blocks land earlier
        # in the cache than the probe. With the old "recover after probe"
        # order, the probe got buried mid-cache while recovered blocks held
        # the model's freshest attention slot — and since each 64-token
        # recovered block ends in post-needle haystack content, the model
        # continued from haystack noise instead of looking back at the
        # needle. Recovering first puts the probe as the freshest context
        # and recovered blocks become earlier context the model attends
        # back to. _last_user_text is set manually here because
        # process_user_message hasn't run yet. K is read from the config so
        # baselines that tune K differently (InfLLM at K=8 vs EVOKE at K=4)
        # exercise their own retrieval breadth.
        mgr._last_user_text = probe
        _smart_recover(mgr, k=config.smart_recover_k)
    mgr.process_user_message(probe)
    # Thinking models (Qwen 3.x and similar) emit <think>...</think> before
    # the actual answer; without think_close the 128-token budget gets
    # consumed by the thinking trace and no answer reaches the scorer.
    # Detected from model name so the bench stays self-configuring across
    # families.
    think_close = _think_close_for(os.environ.get("EVOKE_MODEL_PATH", ""))
    if think_close:
        # Thinking models emit verbose answer preambles after the </think>
        # close ("The capital city of the country called Polopia is the
        # historic settlement of..."). 128 answer tokens get the preamble
        # but cut off the actual needle string; 256 gives the answer room
        # to complete. The thinking_budget covers the trace itself.
        answer = mgr.generate(
            512, think_close=think_close, thinking_budget=2048, answer_budget=256
        )
    else:
        # 256 (was 128) so the substring matcher catches needle strings the
        # model is still in the middle of emitting. The reviewer flagged a
        # cell at budget 512 where the model recovered the needle, began
        # answering ``...the historic settlement of Zit'' and ran out of
        # the 128-token budget mid-word; the matcher then missed
        # ``Zithrand'' and the cell was scored fail. 256 closes that gap
        # without changing any policy's recovery work — every policy's
        # generation budget moves together so no row is selectively
        # advantaged.
        answer = mgr.generate(256)
    elapsed = time.perf_counter() - t0
    stats = mgr.get_stats()
    expected = needle["expected"].lower()
    answer_lower = answer.lower()
    return NiahResult(
        needle_id=needle["id"],
        depth=depth_pct,
        strategy=strategy,
        budget=budget,
        probe_ok=expected in answer_lower,
        evictions=stats.total_evictions,
        recoveries=stats.total_recoveries,
        active_tokens=stats.active_tokens,
        elapsed=elapsed,
        answer=answer.strip().replace("\n", " ")[:80],
    )


def _resolve_needles() -> list[dict]:
    env = os.environ.get("EVOKE_NIAH_NEEDLES")
    if not env:
        return NEEDLES
    wanted = {x.strip() for x in env.split(",")}
    return [n for n in NEEDLES if n["id"] in wanted]


def _resolve_depths() -> list[int]:
    env = os.environ.get("EVOKE_NIAH_DEPTHS")
    if not env:
        return DEFAULT_DEPTHS
    return [int(d.strip()) for d in env.split(",")]


def _aggregate(results: list[NiahResult]) -> dict[tuple[int, str], tuple[int, int]]:
    agg: dict[tuple[int, str], tuple[int, int]] = {}
    for r in results:
        key = (r.budget, r.strategy)
        passed, total = agg.get(key, (0, 0))
        agg[key] = (passed + int(r.probe_ok), total + 1)
    return agg


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("set EVOKE_MODEL_PATH")
        return 1
    budgets = [int(b) for b in os.environ.get("EVOKE_BUDGETS", "1024,2048").split(",")]
    n_paragraphs = int(os.environ.get("EVOKE_NIAH_PARAGRAPHS", "40"))
    seed = int(os.environ.get("EVOKE_NIAH_SEED", "0"))
    needles = _resolve_needles()
    depths = _resolve_depths()
    out_json = os.environ.get("EVOKE_NIAH_JSON")

    kv_quant = os.environ.get("EVOKE_KV_QUANT", "").lower().strip()
    engine_kwargs: dict = {}
    if kv_quant and kv_quant not in ("f16", "none"):
        # kv_block_save/load splice assumes F16 layout; quantized layouts
        # break that invariant, so disable kv_restore-class strategies and
        # only run the non-recovery baselines + the no_eviction comparator
        # in this mode.
        engine_kwargs["type_k"] = kv_quant
        engine_kwargs["type_v"] = kv_quant
    engine = LlamaCppEngine(
        model, n_ctx=16384, n_gpu_layers=-1, verbose=False, **engine_kwargs
    )
    print(f"niah eval | model={Path(model).stem}")
    if kv_quant and kv_quant not in ("f16", "none"):
        print(f"kv cache quantization: type_k=type_v={kv_quant}")
    print(f"kv_block primitives available: {engine.supports_kv_block}")
    print(
        f"haystack: {n_paragraphs} paragraphs, seed={seed}, "
        f"{len(needles)} needles, {len(depths)} depths, {len(budgets)} budgets"
    )
    header = (
        f"{'budget':<8}{'needle':<10}{'depth':<7}{'strategy':<18}{'probe':<7}"
        f"{'evict':<7}{'recov':<7}{'active':<8}{'sec':<7}answer"
    )
    print(header)
    print("-" * len(header))

    results: list[NiahResult] = []
    try:
        for budget in budgets:
            for needle in needles:
                for depth_pct in depths:
                    for name, overrides in STRATEGIES.items():
                        # h2o and evoke_attention need the EVOKE fork for
                        # attention capture; evoke_kv_restore needs the fork
                        # for the K/V splice primitive. Skipping the cell
                        # avoids a silent fallback to recency that would
                        # mislabel the row.
                        needs_kv_block = name in (
                            "evoke_kv_restore",
                            "evoke_recovery_aware",
                            "evoke_attention",
                            "h2o",
                            "snapkv",
                            "infllm",
                        )
                        if needs_kv_block and not engine.supports_kv_block:
                            print(
                                f"{budget:<8}{needle['id']:<10}{depth_pct:<7}"
                                f"{name:<18}SKIP (no LLAMA_CPP_LIB)"
                            )
                            continue
                        if (
                            needs_kv_block
                            and kv_quant
                            and kv_quant not in ("f16", "none")
                        ):
                            print(
                                f"{budget:<8}{needle['id']:<10}{depth_pct:<7}"
                                f"{name:<18}SKIP (kv_block splice unsafe under quantized KV cache)"
                            )
                            continue
                        try:
                            r = run_cell(
                                engine,
                                needle,
                                depth_pct,
                                name,
                                overrides,
                                budget,
                                n_paragraphs,
                                seed,
                            )
                            mark = "PASS" if r.probe_ok else "fail"
                            print(
                                f"{budget:<8}{r.needle_id:<10}{r.depth:<7}"
                                f"{r.strategy:<18}{mark:<7}{r.evictions:<7}"
                                f"{r.recoveries:<7}{r.active_tokens:<8}"
                                f"{r.elapsed:<7.2f}{r.answer!r}"
                            )
                            results.append(r)
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"{budget:<8}{needle['id']:<10}{depth_pct:<7}"
                                f"{name:<18}ERROR: {exc}"
                            )
                    print("-" * len(header))
    finally:
        engine.close()

    print()
    print("Aggregate pass-rate per (budget, strategy)")
    print(f"{'budget':<8}{'strategy':<20}{'pass_rate':<12}{'n_cells':<8}")
    print("-" * 50)
    for (budget, strategy), (passed, total) in sorted(_aggregate(results).items()):
        rate = passed / total if total > 0 else 0.0
        print(f"{budget:<8}{strategy:<20}{rate:<12.2%}{total:<8}")

    if out_json:
        Path(out_json).write_text(
            json.dumps([asdict(r) for r in results], indent=2),
            encoding="utf-8",
        )
        print(f"\nresults JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
