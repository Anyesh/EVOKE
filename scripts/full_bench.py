from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.benchmark import make_needle_case, run_benchmark
from evoke.chat_template import ChatTemplate, detect_template, strip_thinking
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


def main():
    model_name = Path(MODEL).stem
    engine = LlamaCppEngine(MODEL, n_ctx=4096, n_gpu_layers=-1, verbose=False)
    template = detect_template(model_name)

    is_thinking = "qwen3" in model_name.lower()
    max_gen = 1024 if is_thinking else 128

    filler = FILLER * 50
    needle = "The secret password for the vault is CRYSTALLINE-HORIZON-42."
    question = "What is the secret password for the vault?"
    expected = "CRYSTALLINE-HORIZON-42"

    positions = [0.1, 0.25, 0.5, 0.75, 0.9]
    cases = [
        make_needle_case(filler, needle, pos, question, expected) for pos in positions
    ]

    budgets = [512, 1024, 2048]
    strategies = ["truncate", "streaming_llm", "evoke"]

    print(f"Model: {model_name}")
    print(f"Template: {type(template).__name__}")
    print(f"Thinking model: {is_thinking} (max_gen={max_gen})")
    print(f"n_ctx={engine.n_ctx}, n_embd={engine.n_embd}")
    doc_tokens = len(engine.tokenize(filler + " " + needle))
    print(f"Document: ~{doc_tokens} tokens")
    print(f"Positions: {positions}")
    print(f"Budgets: {budgets}")
    print(f"Strategies: {strategies}")
    print()

    report = run_benchmark(
        engine,
        cases=cases,
        budgets=budgets,
        strategies=strategies,
        max_gen_tokens=max_gen,
        chat_template=template,
    )

    groups: dict[tuple[str, int], list] = defaultdict(list)
    for r in report.results:
        groups[(r.strategy, r.budget)].append(r)

    header = f"{'Strategy':<20} {'Budget':>6}  {'F1':>6}  {'Contains':>8}  {'Promo':>5}  {'Demo':>5}  {'Time':>6}"
    print(header)
    print("-" * len(header))

    for (strategy, budget), trials in sorted(groups.items()):
        avg_f1 = sum(t.f1_score for t in trials) / len(trials)
        contains = sum(
            1 for t in trials if "CRYSTALLINE" in strip_thinking(t.generated)
        ) / len(trials)
        avg_promo = sum(t.promotions for t in trials) / len(trials)
        avg_demo = sum(t.demotions for t in trials) / len(trials)
        avg_time = sum(t.generation_time_s for t in trials) / len(trials)
        print(
            f"{strategy:<20} {budget:>6}  {avg_f1:>6.3f}  {contains:>8.0%}  "
            f"{avg_promo:>5.1f}  {avg_demo:>5.0f}  {avg_time:>6.2f}"
        )

    print()
    print("=== Per-case breakdown (evoke strategy) ===")
    for r in report.results:
        if r.strategy != "evoke":
            continue
        answer = strip_thinking(r.generated)
        has_answer = "CRYSTALLINE" in answer
        print(
            f"  {r.case_name:15s} budget={r.budget:>5} "
            f"f1={r.f1_score:.3f} contains={has_answer} "
            f"promo={r.promotions} demo={r.demotions}"
        )
        print(f"    answer: {answer[:120]!r}")

    output_path = Path(__file__).resolve().parents[1] / "results" / "benchmark.json"
    output_path.parent.mkdir(exist_ok=True)
    report.to_json(output_path)
    print(f"\nResults saved to {output_path}")

    engine.close()


if __name__ == "__main__":
    main()
