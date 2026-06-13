"""Multi-fact session probe with seed variance.

Plants five distinct facts (password, capital, amount, code, date) at randomized
depths inside a long haystack, then probes the model on all five at the end of
the session via per-fact probes. Each fact has multiple paraphrase templates
and a seed-specific value, so the same fact-id across two seeds is a different
question with a different answer (no value leakage between seeds).

Scoring is hybrid: a string-match-set on multiple value paraphrases is the
primary metric, and an optional local LLM judge (Gemma 4 E4B by default, loaded
sequentially after the SUT to avoid VRAM contention) breaks ambiguous answers
where the model paraphrased the value rather than emitting it verbatim. The
judge is configurable; setting EVOKE_MFB_JUDGE=none disables it and falls back
to string-match-set only.

Per-(seed, budget, policy) cell evaluates five facts and emits a per-fact PASS
or fail plus the aggregate cell pass-rate (passes/5). Aggregating over seeds
produces a mean pass-rate per (budget, policy) with a 95% Wilson confidence
interval, which is the variance reviewers asked for.

Environment overrides:
- EVOKE_MFB_SEEDS=5
- EVOKE_BUDGETS=512,1024,2048
- EVOKE_MFB_PARAGRAPHS=40
- EVOKE_MFB_JUDGE=gemma|none  (default gemma, falls back to none on load error)
- EVOKE_MFB_JUDGE_PATH (path to judge GGUF; default: gpuhost gemma-4 E4B)
- EVOKE_MFB_JSON (path to write per-cell raw JSON)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evoke.attention_scorer import AttentionScorer
from evoke.config import EvokeConfig
from evoke.embed import RetrievalEmbedder
from evoke.judge import LLMJudge
from evoke.llama_engine import LlamaCppEngine
from evoke.manager import EvokeManager

from niah_bench import (
    STRATEGIES,
    _smart_recover,
    _think_close_for,
    build_haystack,
)

from arkvale_policy import ArkValePolicy, block_cuboid


@dataclass
class Fact:
    fact_id: str
    plant_text: str
    probe: str
    expected_substrings: list[str]
    semantic_keywords: list[str]


NEEDLE_FAMILIES: dict[str, dict] = {
    "password": {
        "plant_templates": [
            "Important administrative note: the secret password for the laboratory vault is {value}. Personnel are reminded not to write it on shared notepads.",
            "Lab access record: the current passphrase for the central vault is {value}. Rotate quarterly per security policy.",
            "Internal memo (security): vault password set to {value}. Treat as confidential and do not log to chat transcripts.",
            "Operational note for the duty officer: the vault password is {value}. Confirm receipt before the next handover.",
        ],
        "value_sets": [
            [
                "icarus-pinwheel-43",
                "fjord-ember-77",
                "azure-mineral-12",
                "tundra-orchid-58",
                "delta-rover-99",
            ],
        ],
        "probe": "What is the secret password for the laboratory vault?",
        "semantic_keywords": ["password", "vault", "laboratory", "passphrase"],
    },
    "capital": {
        "plant_templates": [
            "Geographic footnote for the next briefing: the capital city of the small inland country of Polopia is the historic settlement of {value}, located near the western foothills.",
            "Country profile (Polopia): capital city is {value}. Population centred in the western valley district.",
            "Atlas correction: Polopia's capital is {value}, not the previously listed northern city.",
            "Diplomatic note: official correspondence to Polopia should be addressed to {value}, the capital.",
        ],
        "value_sets": [
            ["Zithrand", "Vellora", "Marathon-Veld", "Quarnith", "Eblantar"],
        ],
        "probe": "What is the capital city of the country called Polopia?",
        "semantic_keywords": ["capital", "Polopia", "city"],
    },
    "amount": {
        "plant_templates": [
            "Lab record: Dr. Halvorsen's calibration run on the spectrometer used exactly {value} milligrams of cesium chloride dissolved in buffered ammonium acetate.",
            "Experimental log: cesium chloride mass for the Halvorsen calibration was {value} milligrams, well within tolerance.",
            "Reagent inventory: {value} mg of cesium chloride consumed in the Halvorsen run on Tuesday.",
            "Notebook entry: cesium chloride dose for Halvorsen's pass was {value} milligrams; record for the audit log.",
        ],
        "value_sets": [
            ["47", "23", "108", "61", "84"],
        ],
        "probe": "How many milligrams of cesium chloride did Dr. Halvorsen use in the calibration run?",
        "semantic_keywords": [
            "cesium",
            "chloride",
            "milligrams",
            "Halvorsen",
            "calibration",
        ],
    },
    "code": {
        "plant_templates": [
            "Operational note for the duty officer: the activation code for the orbital relay station is {value}. Confirm receipt before the next handover window.",
            "Mission control: the orbital relay activation code is {value}, valid for the current orbit.",
            "Crew briefing: when activating the orbital relay, the authorization code is {value}.",
            "Daily ops log: relay station activation code rotated to {value}.",
        ],
        "value_sets": [
            [
                "BLUE-MOUNTAIN-7-DELTA",
                "RED-FALCON-3-OMEGA",
                "GREEN-LANTERN-9-SIGMA",
                "BLACK-HORIZON-5-KAPPA",
                "GOLDEN-CRESCENT-2-TAU",
            ],
        ],
        "probe": "What is the activation code for the orbital relay station?",
        "semantic_keywords": ["activation", "code", "orbital", "relay", "station"],
    },
    "date": {
        "plant_templates": [
            "Archival entry: the Treaty of Vrenholm was signed on the twenty-third of October, {value}, at a quarter past four in the afternoon.",
            "Historical record: the Treaty of Vrenholm was concluded on 23 October {value}, sealing the western alliance.",
            "Diplomatic timeline: Vrenholm Treaty ratification date is {value}; ceremony held in the merchants' guild hall.",
            "History note: the Treaty of Vrenholm dates to {value} (October), often cited as the start of the regional accord.",
        ],
        "value_sets": [
            ["1786", "1842", "1907", "1623", "1759"],
        ],
        "probe": "In what year was the Treaty of Vrenholm signed?",
        "semantic_keywords": ["Treaty", "Vrenholm", "year", "signed"],
    },
}


def build_fact_set(seed: int) -> list[Fact]:
    rng = random.Random(seed)
    facts: list[Fact] = []
    for fact_id, family in NEEDLE_FAMILIES.items():
        values = family["value_sets"][0]
        value = values[seed % len(values)]
        template = rng.choice(family["plant_templates"])
        plant_text = template.format(value=value)
        # Expected substrings: the value verbatim plus the value with common
        # punctuation/quote variations that don't change semantic content.
        # The judge handles paraphrase; this set handles formatting variance.
        expected = [value, value.lower(), value.upper()]
        if "-" in value:
            expected.append(value.replace("-", " "))
        facts.append(
            Fact(
                fact_id=fact_id,
                plant_text=plant_text,
                probe=family["probe"],
                expected_substrings=sorted(set(expected)),
                semantic_keywords=family["semantic_keywords"],
            )
        )
    return facts


def assign_depths(n_facts: int, n_paragraphs: int, seed: int) -> list[int]:
    # Spread facts evenly across the haystack with a small random jitter per
    # seed so the same fact doesn't land in the same paragraph across seeds.
    rng = random.Random(seed + 17)
    base = [int((i + 1) * n_paragraphs / (n_facts + 1)) for i in range(n_facts)]
    jitter = [rng.randint(-2, 2) for _ in range(n_facts)]
    return [max(1, min(n_paragraphs - 1, b + j)) for b, j in zip(base, jitter)]


def insert_facts(haystack: list[str], facts: list[Fact], depths: list[int]) -> str:
    # Sort by depth so positions stay stable across insertions.
    by_depth = sorted(zip(depths, facts), key=lambda x: x[0])
    pieces: list[str] = []
    last_idx = 0
    for depth, fact in by_depth:
        pieces.extend(haystack[last_idx:depth])
        pieces.append(fact.plant_text)
        last_idx = depth
    pieces.extend(haystack[last_idx:])
    return "\n\n".join(pieces)


@dataclass
class CellResult:
    seed: int
    budget: int
    strategy: str
    fact_results: dict[str, bool] = field(default_factory=dict)
    fact_answers: dict[str, str] = field(default_factory=dict)
    # Per-fact "needs LLM judge" flag. True when string-match-set failed
    # AND the answer contains semantic keywords (suggesting the model
    # engaged with the topic but produced a paraphrase/partial match the
    # string matcher cannot see). False on clean match or clean miss.
    fact_ambiguous: dict[str, bool] = field(default_factory=dict)
    fact_plant_texts: dict[str, str] = field(default_factory=dict)
    evictions: int = 0
    recoveries: int = 0
    active_tokens: int = 0
    elapsed: float = 0.0

    @property
    def pass_count(self) -> int:
        return sum(1 for v in self.fact_results.values() if v)

    @property
    def total(self) -> int:
        return len(self.fact_results)


def _string_match(answer: str, expected: list[str]) -> bool:
    lower = answer.lower()
    return any(e.lower() in lower for e in expected)


def _semantically_engaged(answer: str, keywords: list[str], threshold: int = 1) -> bool:
    # The answer engaged with the fact's topic when at least `threshold`
    # of the fact's semantic keywords appear in it; we then route to the
    # LLM judge to decide whether the model recalled the right value or
    # confabulated. Threshold 1 is permissive; raise it if the judge is
    # firing on too many shallow matches.
    lower = answer.lower()
    return sum(1 for k in keywords if k.lower() in lower) >= threshold


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


_RETRIEVAL_EMBEDDER = RetrievalEmbedder()


def _run_cell_arkvale(
    engine: LlamaCppEngine,
    facts: list[Fact],
    depths: list[int],
    haystack: list[str],
    overrides: dict,
    budget: int,
    seed: int,
) -> CellResult:
    # Faithful ArkVale: build per-block key cuboids from the key-capture as the document is
    # read block-by-block (same tokens/positions as a bulk add, so the cache is identical),
    # then at each probe score every block by q.cuboid and run the bounded top-k recall-and-evict
    # at original positions. The query is the last probe token's q at the scoring layer.
    engine.reset()
    config = EvokeConfig(
        max_active_tokens=budget,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
        **overrides,
    )
    mgr = EvokeManager(engine, config)

    engine.attn_capture_set_layer(config.attention_capture_layer)
    qbuf = np.zeros(4_000_000, dtype=np.float32)
    kbuf = np.zeros(4_000_000, dtype=np.float32)
    engine.query_capture_set_buffer(qbuf)
    engine.key_capture_set_buffer(kbuf)
    policy = ArkValePolicy(budget_blocks=max(1, budget // config.block_size))

    document = insert_facts(haystack, facts, depths)
    toks = engine.tokenize(document)
    bs = config.block_size
    for ci in range(0, len(toks), bs):
        chunk = toks[ci : ci + bs]
        # The key-capture is the FULL (padded) cache; the new chunk's real keys sit at
        # decode-time positions [start, start+len). Slice exactly those -- slicing the
        # padded tail would grab masked cells and yield garbage cuboids.
        start = engine.next_write_pos
        bkey = f"doc_s{seed}_b{budget}_c{ci // bs}"
        mgr.add_context_tokens(chunk, key=bkey)
        kcap = engine.read_key_capture()
        if kcap is not None and kcap.shape[0] >= start + len(chunk):
            policy.set_cuboid(
                f"{bkey}#0", *block_cuboid(kcap[start : start + len(chunk)])
            )

    result = CellResult(seed=seed, budget=budget, strategy="arkvale")
    think_close = _think_close_for(os.environ.get("EVOKE_MODEL_PATH", ""))
    t0 = time.perf_counter()
    for fact in facts:
        probe = f"\n\nQuestion: {fact.probe}\nAnswer:"
        mgr.process_user_message(probe)
        qcap = engine.read_query_capture()
        if qcap is not None and qcap.shape[0] > 0:
            policy.recall_and_evict(mgr, qcap[-1])
        if think_close:
            answer = mgr.generate(
                512, think_close=think_close, thinking_budget=2048, answer_budget=256
            )
        else:
            answer = mgr.generate(128)
        matched = _string_match(answer, fact.expected_substrings)
        result.fact_results[fact.fact_id] = matched
        result.fact_answers[fact.fact_id] = answer.strip().replace("\n", " ")[:120]
        result.fact_plant_texts[fact.fact_id] = fact.plant_text
        result.fact_ambiguous[fact.fact_id] = (
            False if matched else _semantically_engaged(answer, fact.semantic_keywords)
        )
    result.elapsed = time.perf_counter() - t0
    engine.query_capture_set_buffer(None)
    engine.key_capture_set_buffer(None)
    engine.attn_capture_set_layer(-1)
    stats = mgr.get_stats()
    result.evictions = stats.total_evictions
    result.recoveries = stats.total_recoveries
    result.active_tokens = stats.active_tokens
    return result


def run_cell(
    engine: LlamaCppEngine,
    facts: list[Fact],
    depths: list[int],
    haystack: list[str],
    strategy: str,
    overrides: dict,
    budget: int,
    seed: int,
) -> CellResult:
    if strategy == "arkvale":
        return _run_cell_arkvale(
            engine, facts, depths, haystack, overrides, budget, seed
        )
    engine.reset()
    config_kwargs: dict = dict(
        max_active_tokens=budget,
        block_size=64,
        high_watermark=0.95,
        low_watermark=0.75,
    )
    config_kwargs.update(overrides)
    config = EvokeConfig(**config_kwargs)
    attn_scorer = _build_scorer(engine, config)
    retrieval = _RETRIEVAL_EMBEDDER if config.use_retrieval_embeddings else None
    mgr = EvokeManager(
        engine,
        config,
        attention_scorer=attn_scorer,
        retrieval_embedder=retrieval,
    )

    document = insert_facts(haystack, facts, depths)
    mgr.add_context(document, key=f"doc_s{seed}_b{budget}")

    result = CellResult(seed=seed, budget=budget, strategy=strategy)
    think_close = _think_close_for(os.environ.get("EVOKE_MODEL_PATH", ""))
    t0 = time.perf_counter()
    for fact in facts:
        probe = f"\n\nQuestion: {fact.probe}\nAnswer:"
        if overrides.get("recovery_mode") != "discard":
            mgr._last_user_text = probe
            _smart_recover(mgr, k=config.smart_recover_k)
        mgr.process_user_message(probe)
        if think_close:
            answer = mgr.generate(
                512,
                think_close=think_close,
                thinking_budget=2048,
                answer_budget=256,
            )
        else:
            answer = mgr.generate(128)
        matched = _string_match(answer, fact.expected_substrings)
        result.fact_results[fact.fact_id] = matched
        result.fact_answers[fact.fact_id] = answer.strip().replace("\n", " ")[:120]
        result.fact_plant_texts[fact.fact_id] = fact.plant_text
        if not matched:
            result.fact_ambiguous[fact.fact_id] = _semantically_engaged(
                answer, fact.semantic_keywords
            )
        else:
            result.fact_ambiguous[fact.fact_id] = False
    result.elapsed = time.perf_counter() - t0
    stats = mgr.get_stats()
    result.evictions = stats.total_evictions
    result.recoveries = stats.total_recoveries
    result.active_tokens = stats.active_tokens
    return result


def _wilson_interval(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = passes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("set EVOKE_MODEL_PATH")
        return 1
    seeds = int(os.environ.get("EVOKE_MFB_SEEDS", "5"))
    budgets = [int(b) for b in os.environ.get("EVOKE_BUDGETS", "1024").split(",")]
    n_paragraphs = int(os.environ.get("EVOKE_MFB_PARAGRAPHS", "40"))
    out_json = os.environ.get("EVOKE_MFB_JSON")

    kv_quant = os.environ.get("EVOKE_KV_QUANT", "").lower().strip()
    engine_kwargs: dict = {}
    if kv_quant and kv_quant not in ("f16", "none"):
        engine_kwargs["type_k"] = kv_quant
        engine_kwargs["type_v"] = kv_quant
    engine = LlamaCppEngine(
        model, n_ctx=16384, n_gpu_layers=-1, verbose=False, **engine_kwargs
    )
    print(f"multifact eval | model={Path(model).stem}")
    if kv_quant and kv_quant not in ("f16", "none"):
        print(f"kv cache quantization: type_k=type_v={kv_quant}")
    print(f"kv_block primitives available: {engine.supports_kv_block}")
    print(
        f"haystack: {n_paragraphs} paragraphs, {seeds} seeds, "
        f"{len(NEEDLE_FAMILIES)} facts per session"
    )
    print(
        f"{'seed':<5}{'budget':<8}{'strategy':<18}{'pass':<6}"
        f"{'evict':<7}{'recov':<7}{'sec':<7}details"
    )
    print("-" * 90)

    all_results: list[CellResult] = []
    try:
        for seed in range(seeds):
            facts = build_fact_set(seed)
            depths = assign_depths(len(facts), n_paragraphs, seed)
            haystack = build_haystack(n_paragraphs, seed)
            only = os.environ.get("EVOKE_STRATEGIES", "").strip()
            selected = set(only.split(",")) if only else None
            for budget in budgets:
                for name, overrides in STRATEGIES.items():
                    if selected is not None and name not in selected:
                        continue
                    needs_kv_block = name in (
                        "evoke_kv_restore",
                        "evoke_recovery_aware",
                        "evoke_attention",
                        "h2o",
                        "snapkv",
                        "infllm",
                        "arkvale",
                        "sparse_importance",
                    )
                    if needs_kv_block and not engine.supports_kv_block:
                        continue
                    try:
                        r = run_cell(
                            engine,
                            facts,
                            depths,
                            haystack,
                            name,
                            overrides,
                            budget,
                            seed,
                        )
                        per_fact = ",".join(
                            f"{fid}:{int(ok)}" for fid, ok in r.fact_results.items()
                        )
                        print(
                            f"{seed:<5}{budget:<8}{name:<18}"
                            f"{r.pass_count}/{r.total:<3}"
                            f"{r.evictions:<7}{r.recoveries:<7}"
                            f"{r.elapsed:<7.2f}{per_fact}"
                        )
                        all_results.append(r)
                    except Exception as exc:  # noqa: BLE001
                        print(f"{seed:<5}{budget:<8}{name:<18}ERROR: {exc}")
                print("-" * 90)
    finally:
        engine.close()

    judge_mode = os.environ.get("EVOKE_MFB_JUDGE", "gemma").lower()
    if judge_mode != "none":
        ambiguous_cells = [
            (r, fact_id)
            for r in all_results
            for fact_id, ambiguous in r.fact_ambiguous.items()
            if ambiguous
        ]
        if ambiguous_cells:
            print(
                f"\nLoading judge to break {len(ambiguous_cells)} ambiguous "
                "cases (string-match failed, semantic keywords present)..."
            )
            try:
                with LLMJudge() as judge:
                    for r, fact_id in ambiguous_cells:
                        verdict = judge.score(
                            r.fact_plant_texts[fact_id],
                            r.fact_answers[fact_id],
                        )
                        if verdict:
                            r.fact_results[fact_id] = True
                print("Judge pass complete.")
            except FileNotFoundError as exc:
                print(f"Judge unavailable, falling back to string-match-only: {exc}")
            except (OSError, RuntimeError) as exc:
                print(f"Judge load failed, falling back to string-match-only: {exc}")
        else:
            print("\nNo ambiguous cases; skipping judge pass.")

    print()
    print("Aggregate per (budget, strategy) over seeds")
    print(
        f"{'budget':<8}{'strategy':<20}{'pass_rate':<11}{'95% CI':<22}{'n_facts':<10}"
    )
    print("-" * 75)
    by_cell: dict[tuple[int, str], tuple[int, int]] = {}
    for r in all_results:
        key = (r.budget, r.strategy)
        passes, total = by_cell.get(key, (0, 0))
        by_cell[key] = (passes + r.pass_count, total + r.total)
    for (budget, strategy), (passes, total) in sorted(by_cell.items()):
        rate = passes / total if total > 0 else 0.0
        lo, hi = _wilson_interval(passes, total)
        print(
            f"{budget:<8}{strategy:<20}{rate:<11.2%}"
            f"[{lo:.2%}, {hi:.2%}]      {passes}/{total}"
        )

    if out_json:
        Path(out_json).write_text(
            json.dumps([asdict(r) for r in all_results], indent=2),
            encoding="utf-8",
        )
        print(f"\nresults JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
