from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.benchmark import make_needle_case, run_benchmark
from evoke.llama_engine import LlamaCppEngine

import os

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
    engine = LlamaCppEngine(MODEL, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    filler = FILLER * 50
    needle = "The secret password for the vault is CRYSTALLINE-HORIZON-42."
    question = "What is the secret password for the vault?"
    expected = "CRYSTALLINE-HORIZON-42"

    cases = [
        make_needle_case(filler, needle, pos, question, expected)
        for pos in [0.25, 0.5, 0.75]
    ]

    budgets = [512, 1024]
    strategies = ["truncate", "streaming_llm", "evoke"]

    print(f"n_ctx={engine.n_ctx}, n_embd={engine.n_embd}")
    print(f"Document ~{len(engine.tokenize(filler))} tokens, budget={budgets}")
    print(f"Running strategies: {strategies}\n")

    report = run_benchmark(
        engine,
        cases=cases,
        budgets=budgets,
        strategies=strategies,
        max_gen_tokens=64,
    )

    print(report.summary_table())
    print()

    for r in report.results:
        print(
            f"[{r.strategy:15s}] budget={r.budget} "
            f"f1={r.f1_score:.3f} em={r.exact_match} "
            f"demote={r.demotions} promote={r.promotions} "
            f"active={r.active_tokens} archive={r.archive_tokens}"
        )
        print(f"  generated: {r.generated[:120]!r}")
        print()

    engine.close()


if __name__ == "__main__":
    main()
