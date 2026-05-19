from __future__ import annotations

import dataclasses
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from evoke.chat_template import ChatTemplate, PassthroughTemplate, strip_thinking
from evoke.config import EvokeConfig
from evoke.engine import InferenceEngine
from evoke.manager import EvokeManager


@dataclass
class QAPair:
    question: str
    expected_answer: str
    position_hint: str = ""


@dataclass
class BenchmarkCase:
    name: str
    document: str
    qa_pairs: list[QAPair]
    description: str = ""


@dataclass
class TrialResult:
    strategy: str
    case_name: str
    budget: int
    question: str
    expected: str
    generated: str
    f1_score: float
    exact_match: bool
    generation_time_s: float
    active_tokens: int
    archive_tokens: int
    demotions: int
    promotions: int


@dataclass
class BenchmarkReport:
    results: list[TrialResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary_table(self) -> str:
        groups: dict[tuple[str, int], list[TrialResult]] = defaultdict(list)
        for r in self.results:
            groups[(r.strategy, r.budget)].append(r)

        lines = [
            f"{'Strategy':<20} {'Budget':>8} {'Avg F1':>8} {'EM':>6} "
            f"{'Demote':>8} {'Promote':>8} {'Time(s)':>8}"
        ]
        lines.append("-" * len(lines[0]))

        for (strategy, budget), trials in sorted(groups.items()):
            avg_f1 = sum(t.f1_score for t in trials) / len(trials)
            em_rate = sum(t.exact_match for t in trials) / len(trials)
            avg_demotions = sum(t.demotions for t in trials) / len(trials)
            avg_promotions = sum(t.promotions for t in trials) / len(trials)
            avg_time = sum(t.generation_time_s for t in trials) / len(trials)
            lines.append(
                f"{strategy:<20} {budget:>8} {avg_f1:>8.3f} {em_rate:>6.1%} "
                f"{avg_demotions:>8.0f} {avg_promotions:>8.0f} {avg_time:>8.2f}"
            )

        return "\n".join(lines)

    def to_json(self, path: Path) -> None:
        data = {
            "metadata": self.metadata,
            "results": [dataclasses.asdict(r) for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2))


STRATEGIES: dict[str, EvokeConfig] = {
    "truncate": EvokeConfig(
        block_size=32,
        w_recency=1.0,
        w_sink=0.0,
        w_coherence=0.0,
        retrieval_threshold=2.0,
        demotion_policy="watermark",
        high_watermark=0.95,
        low_watermark=0.75,
    ),
    "streaming_llm": EvokeConfig(
        block_size=32,
        w_recency=1.0,
        w_sink=1.0,
        w_coherence=0.0,
        retrieval_threshold=2.0,
        demotion_policy="watermark",
        high_watermark=0.95,
        low_watermark=0.75,
    ),
    "evoke": EvokeConfig(
        block_size=32,
        w_recency=0.4,
        w_sink=1.0,
        w_coherence=0.6,
        retrieval_threshold=0.85,
        max_retrieve_blocks=4,
        demotion_policy="watermark",
        high_watermark=0.95,
        low_watermark=0.75,
    ),
}


def compute_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, reference: str) -> bool:
    return _normalize(prediction) == _normalize(reference)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    for punct in ".,;:!?\"'()[]{}":
        text = text.replace(punct, " ")
    return " ".join(text.split())


def run_benchmark(
    engine: InferenceEngine,
    cases: list[BenchmarkCase],
    budgets: list[int],
    strategies: list[str] | None = None,
    max_gen_tokens: int = 256,
    chat_template: ChatTemplate | None = None,
) -> BenchmarkReport:
    template = chat_template or PassthroughTemplate()
    strategy_names = strategies or list(STRATEGIES.keys())
    report = BenchmarkReport(
        metadata={
            "budgets": budgets,
            "strategies": strategy_names,
            "max_gen_tokens": max_gen_tokens,
            "n_ctx": engine.n_ctx,
        }
    )

    for case in cases:
        for budget in budgets:
            full_context_config = EvokeConfig(
                max_active_tokens=engine.n_ctx,
                retrieval_threshold=2.0,
            )
            full_results = _run_case(
                engine,
                case,
                full_context_config,
                "full",
                budget,
                max_gen_tokens,
                template,
            )
            report.results.extend(full_results)

            for strat_name in strategy_names:
                if strat_name == "full":
                    continue
                config = _make_config(strat_name, budget)
                results = _run_case(
                    engine,
                    case,
                    config,
                    strat_name,
                    budget,
                    max_gen_tokens,
                    template,
                )
                report.results.extend(results)

    return report


def _make_config(strategy: str, budget: int) -> EvokeConfig:
    base = STRATEGIES[strategy]
    return EvokeConfig(
        max_active_tokens=budget,
        block_size=base.block_size,
        sink_count=base.sink_count,
        score_interval=base.score_interval,
        recency_decay=base.recency_decay,
        w_recency=base.w_recency,
        w_sink=base.w_sink,
        w_coherence=base.w_coherence,
        demotion_policy=base.demotion_policy,
        high_watermark=base.high_watermark,
        low_watermark=base.low_watermark,
        retrieval_threshold=base.retrieval_threshold,
        max_retrieve_blocks=base.max_retrieve_blocks,
        max_archive_blocks=base.max_archive_blocks,
        pin_generated=base.pin_generated,
    )


def _run_case(
    engine: InferenceEngine,
    case: BenchmarkCase,
    config: EvokeConfig,
    strategy: str,
    budget: int,
    max_gen_tokens: int,
    template: ChatTemplate,
) -> list[TrialResult]:
    results: list[TrialResult] = []

    stop_token_ids: set[int] = set()
    for s in template.stop_strings:
        toks = engine.tokenize(s)
        if len(toks) == 1:
            stop_token_ids.add(toks[0])

    engine.reset()
    mgr = EvokeManager(engine, config)

    doc_prefix = template.wrap_document_prefix(case.document)
    mgr.load_document(doc_prefix)

    for qa in case.qa_pairs:
        question_suffix = template.wrap_question_suffix(qa.question)
        mgr.process_user_message(question_suffix)

        t0 = time.monotonic()
        raw_answer = mgr.generate(max_gen_tokens, stop_token_ids=stop_token_ids)
        gen_time = time.monotonic() - t0

        answer = template.extract_answer(raw_answer)

        stats = mgr.get_stats()
        f1 = compute_f1(answer, qa.expected_answer)
        em = compute_exact_match(answer, qa.expected_answer)

        results.append(
            TrialResult(
                strategy=strategy,
                case_name=case.name,
                budget=budget,
                question=qa.question,
                expected=qa.expected_answer,
                generated=raw_answer,
                f1_score=f1,
                exact_match=em,
                generation_time_s=gen_time,
                active_tokens=stats.active_tokens,
                archive_tokens=stats.archive_tokens,
                demotions=stats.total_demotions,
                promotions=stats.total_promotions,
            )
        )

    return results


def make_needle_case(
    document_text: str,
    needle: str,
    needle_position: float,
    question: str,
    expected: str,
) -> BenchmarkCase:
    words = document_text.split()
    insert_idx = int(len(words) * needle_position)
    words.insert(insert_idx, needle)
    doc = " ".join(words)

    return BenchmarkCase(
        name=f"needle@{needle_position:.0%}",
        document=doc,
        qa_pairs=[QAPair(question=question, expected_answer=expected)],
        description=f"Needle at {needle_position:.0%} of document",
    )


def load_cases_from_json(path: Path) -> list[BenchmarkCase]:
    data = json.loads(path.read_text())
    cases = []
    for item in data:
        qa_pairs = [
            QAPair(
                question=qa["question"],
                expected_answer=qa["expected"],
                position_hint=qa.get("position_hint", ""),
            )
            for qa in item["qa_pairs"]
        ]
        cases.append(
            BenchmarkCase(
                name=item["name"],
                document=item["document"],
                qa_pairs=qa_pairs,
                description=item.get("description", ""),
            )
        )
    return cases
