from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.benchmark import (
    BenchmarkCase,
    QAPair,
    make_multi_turn_case,
    make_needle_case,
    run_benchmark,
)
from evoke.chat_template import detect_template, strip_thinking
from evoke.llama_engine import LlamaCppEngine

MODEL = os.environ.get(
    "EVOKE_MODEL_PATH",
    str(
        Path(__file__).resolve().parents[1]
        / "models"
        / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
)

FILLER = (
    "This is a passage about various topics. The weather today is partly cloudy "
    "with a chance of rain in the afternoon. Markets showed mixed performance "
    "with technology stocks leading gains while energy shares declined. "
    "Scientists have discovered a new species of deep-sea fish in the Pacific "
    "Ocean that displays bioluminescent properties. "
)


def build_needle_cases(filler: str) -> list[BenchmarkCase]:
    needle = "The secret password for the vault is CRYSTALLINE-HORIZON-42."
    question = "What is the secret password for the vault?"
    expected = "CRYSTALLINE-HORIZON-42"
    positions = [0.1, 0.25, 0.5, 0.75, 0.9]
    return [
        make_needle_case(filler, needle, pos, question, expected) for pos in positions
    ]


def build_multi_turn_cases(filler: str) -> list[BenchmarkCase]:
    cases = []

    cases.append(
        make_multi_turn_case(
            name="recall-code",
            document=filler,
            inject_turns=[
                "The secret project codename is AURORA-SEVEN and its budget is $4.2 million.",
                "What color is the sky on a clear day?",
                "Tell me about recent developments in renewable energy.",
                "How deep can the new species of fish be found?",
            ],
            eval_question="What is the codename of the secret project and what is its budget?",
            expected_answer="AURORA-SEVEN $4.2 million",
            description="Recall a fact planted 4 turns ago through filler conversation",
        )
    )

    cases.append(
        make_multi_turn_case(
            name="recall-date",
            document=filler,
            inject_turns=[
                "The quarterly review meeting is scheduled for March 15th at 2pm in Room 407.",
                "What are the latest stock market trends?",
                "Tell me about bioluminescent deep-sea creatures.",
                "What is the capital of France?",
            ],
            eval_question="When and where is the quarterly review meeting?",
            expected_answer="March 15th 2pm Room 407",
            description="Recall a date/location planted 4 turns ago",
        )
    )

    cases.append(
        make_multi_turn_case(
            name="recall-name",
            document=filler,
            inject_turns=[
                "Dr. Elena Vasquez is the lead researcher on the deep-sea bioluminescence project.",
                "What is the weather forecast for tomorrow?",
                "Tell me about recent advances in quantum computing.",
                "How do solar panels work?",
                "What are the benefits of meditation?",
            ],
            eval_question="Who is the lead researcher on the deep-sea bioluminescence project?",
            expected_answer="Dr. Elena Vasquez",
            description="Recall a name planted 5 turns ago through filler",
        )
    )

    return cases


def print_results(report, keyword: str = "CRYSTALLINE"):
    groups: dict[tuple[str, int, str], list] = defaultdict(list)
    for r in report.results:
        case_type = "needle" if r.case_name.startswith("needle") else "recall"
        groups[(r.strategy, r.budget, case_type)].append(r)

    header = (
        f"{'Strategy':<16} {'Budget':>6} {'Type':<8} "
        f"{'F1':>6} {'Contains':>8} {'Promo':>5} {'Demo':>5} {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    for (strategy, budget, case_type), trials in sorted(groups.items()):
        avg_f1 = sum(t.f1_score for t in trials) / len(trials)
        if case_type == "needle":
            contains = sum(
                1 for t in trials if keyword in strip_thinking(t.generated)
            ) / len(trials)
        else:
            contains = sum(
                1
                for t in trials
                if any(
                    w in strip_thinking(t.generated).upper()
                    for w in t.expected.upper().split()[:2]
                )
            ) / len(trials)
        avg_promo = sum(t.promotions for t in trials) / len(trials)
        avg_demo = sum(t.demotions for t in trials) / len(trials)
        avg_time = sum(t.generation_time_s for t in trials) / len(trials)
        print(
            f"{strategy:<16} {budget:>6} {case_type:<8} "
            f"{avg_f1:>6.3f} {contains:>8.0%} "
            f"{avg_promo:>5.1f} {avg_demo:>5.0f} {avg_time:>6.2f}"
        )

    print()
    print("=== Per-case detail ===")
    for r in report.results:
        answer = strip_thinking(r.generated)
        print(
            f"  {r.strategy:<14} budget={r.budget:>5} {r.case_name:15s} "
            f"f1={r.f1_score:.3f} promo={r.promotions} demo={r.demotions}"
        )
        print(f"    expected: {r.expected!r}")
        print(f"    answer:   {answer[:150]!r}")


def main():
    model_name = Path(MODEL).stem
    n_ctx = int(os.environ.get("EVOKE_N_CTX", "131072"))
    engine = LlamaCppEngine(MODEL, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
    template = detect_template(model_name)

    print(f"Model: {model_name}")
    print(f"Template: {type(template).__name__}")
    print(f"Thinking: {template.think_close is not None}")
    print(f"n_ctx={engine.n_ctx}, n_embd={engine.n_embd}")

    filler = FILLER * 50
    doc_tokens = len(engine.tokenize(filler))
    print(f"Filler document: ~{doc_tokens} tokens")
    print()

    needle_cases = build_needle_cases(filler)
    multi_turn_cases = build_multi_turn_cases(filler)
    all_cases = needle_cases + multi_turn_cases

    budgets = [512, 1024, 2048]
    strategies = ["truncate", "streaming_llm", "evoke_no_ret", "evoke"]

    max_gen = 512 if template.think_close else 128

    print(f"Cases: {len(needle_cases)} needle + {len(multi_turn_cases)} multi-turn")
    print(f"Budgets: {budgets}")
    print(f"Strategies: {strategies} + full")
    print(f"Max gen tokens: {max_gen}")
    print()

    report = run_benchmark(
        engine,
        cases=all_cases,
        budgets=budgets,
        strategies=strategies,
        max_gen_tokens=max_gen,
        inject_gen_tokens=64,
        chat_template=template,
    )

    print_results(report)

    output_path = Path(__file__).resolve().parents[1] / "results" / "benchmark.json"
    output_path.parent.mkdir(exist_ok=True)
    report.to_json(output_path)
    print(f"\nResults saved to {output_path}")

    engine.close()


if __name__ == "__main__":
    main()
