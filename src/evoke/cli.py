from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evoke.benchmark import (
    BenchmarkCase,
    QAPair,
    load_cases_from_json,
    make_needle_case,
    run_benchmark,
)
from evoke.llama_engine import LlamaCppEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="EVOKE benchmark runner")
    parser.add_argument("model_path", help="Path to GGUF model file")
    parser.add_argument(
        "--cases",
        type=Path,
        help="Path to benchmark cases JSON file",
    )
    parser.add_argument(
        "--needle",
        action="store_true",
        help="Run needle-in-haystack benchmark instead of cases file",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="2048,4096,8192",
        help="Comma-separated list of token budgets (default: 2048,4096,8192)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="truncate,streaming_llm,evoke",
        help="Comma-separated strategies (default: truncate,streaming_llm,evoke)",
    )
    parser.add_argument("--n-ctx", type=int, default=131072)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, help="Path to write JSON results")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]

    if args.needle:
        cases = _build_needle_cases(args.n_ctx)
    elif args.cases:
        cases = load_cases_from_json(args.cases)
    else:
        print("Specify --cases or --needle", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model_path}")
    with LlamaCppEngine(
        args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.verbose,
    ) as engine:
        print(f"Model loaded. n_ctx={engine.n_ctx}, n_embd={engine.n_embd}")
        print(f"Budgets: {budgets}")
        print(f"Strategies: {strategies}")
        print(f"Cases: {len(cases)}")
        print()

        report = run_benchmark(
            engine,
            cases=cases,
            budgets=budgets,
            strategies=strategies,
            max_gen_tokens=args.max_gen_tokens,
        )

    print(report.summary_table())
    print()

    if args.output:
        report.to_json(args.output)
        print(f"Results written to {args.output}")


FILLER = (
    "This is a passage about various topics. The weather today is partly cloudy "
    "with a chance of rain in the afternoon. Markets showed mixed performance "
    "with technology stocks leading gains while energy shares declined. "
    "Scientists have discovered a new species of deep-sea fish in the Pacific "
    "Ocean that displays bioluminescent properties. "
)


def _build_needle_cases(n_ctx: int = 4096) -> list[BenchmarkCase]:
    target_tokens = int(n_ctx * 0.8)
    tokens_per_filler = 50
    reps = max(10, target_tokens // tokens_per_filler)
    filler = FILLER * reps
    needle = "The secret password for the vault is CRYSTALLINE-HORIZON-42."
    question = "What is the secret password for the vault?"
    expected = "CRYSTALLINE-HORIZON-42"

    return [
        make_needle_case(filler, needle, pos, question, expected)
        for pos in [0.1, 0.25, 0.5, 0.75, 0.9]
    ]


if __name__ == "__main__":
    main()
