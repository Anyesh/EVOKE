from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evoke.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    QAPair,
    TrialResult,
    compute_exact_match,
    compute_f1,
    load_cases_from_json,
    make_multi_turn_case,
    make_needle_case,
    run_benchmark,
)
from evoke.chat_template import ChatMLThinkingTemplate
from evoke.mock_engine import MockEngine


class TestF1:
    def test_perfect_match(self):
        assert compute_f1("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert compute_f1("dog runs", "cat sits") == 0.0

    def test_partial_overlap(self):
        f1 = compute_f1("the big cat", "the small cat")
        assert 0.5 < f1 < 1.0

    def test_ignores_punctuation(self):
        assert compute_f1("hello, world!", "hello world") == 1.0

    def test_case_insensitive(self):
        assert compute_f1("Hello World", "hello world") == 1.0

    def test_empty_reference(self):
        assert compute_f1("", "") == 1.0
        assert compute_f1("something", "") == 0.0

    def test_empty_prediction(self):
        assert compute_f1("", "expected") == 0.0


class TestExactMatch:
    def test_exact(self):
        assert compute_exact_match("hello world", "hello world")

    def test_case_insensitive(self):
        assert compute_exact_match("Hello World", "hello world")

    def test_punctuation_ignored(self):
        assert compute_exact_match("hello, world!", "hello world")

    def test_mismatch(self):
        assert not compute_exact_match("hello", "world")


class TestNeedleCase:
    def test_creates_case_with_needle(self):
        doc = "word " * 100
        case = make_needle_case(
            doc.strip(),
            "SECRET_NEEDLE",
            0.5,
            "What is the secret?",
            "SECRET_NEEDLE",
        )
        assert "SECRET_NEEDLE" in case.document
        assert case.name == "needle@50%"
        assert len(case.qa_pairs) == 1


class TestLoadCases:
    def test_round_trip(self):
        data = [
            {
                "name": "test_case",
                "document": "some document text",
                "qa_pairs": [{"question": "what?", "expected": "answer"}],
                "description": "a test",
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cases = load_cases_from_json(Path(f.name))

        assert len(cases) == 1
        assert cases[0].name == "test_case"
        assert cases[0].qa_pairs[0].expected_answer == "answer"


class TestBenchmarkReport:
    def test_summary_table(self):
        report = BenchmarkReport(
            results=[
                TrialResult(
                    strategy="full",
                    case_name="test",
                    budget=1024,
                    question="q",
                    expected="a",
                    generated="a",
                    f1_score=1.0,
                    exact_match=True,
                    generation_time_s=0.1,
                    active_tokens=500,
                    archive_tokens=0,
                    demotions=0,
                    promotions=0,
                ),
            ]
        )
        table = report.summary_table()
        assert "full" in table
        assert "1.000" in table

    def test_to_json(self):
        report = BenchmarkReport(
            results=[
                TrialResult(
                    strategy="test",
                    case_name="c",
                    budget=100,
                    question="q",
                    expected="e",
                    generated="g",
                    f1_score=0.5,
                    exact_match=False,
                    generation_time_s=1.0,
                    active_tokens=50,
                    archive_tokens=10,
                    demotions=2,
                    promotions=1,
                ),
            ],
            metadata={"test": True},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report.to_json(Path(f.name))
            data = json.loads(Path(f.name).read_text())

        assert len(data["results"]) == 1
        assert data["metadata"]["test"] is True


class TestRunBenchmark:
    def test_runs_with_mock_engine(self):
        engine = MockEngine(n_ctx=2048)
        case = BenchmarkCase(
            name="simple",
            document="a " * 200,
            qa_pairs=[QAPair(question="what?", expected_answer="a")],
        )
        report = run_benchmark(
            engine,
            cases=[case],
            budgets=[512],
            strategies=["truncate"],
            max_gen_tokens=8,
        )
        assert len(report.results) >= 2
        strategies_seen = {r.strategy for r in report.results}
        assert "full" in strategies_seen
        assert "truncate" in strategies_seen

    def test_multiple_strategies(self):
        engine = MockEngine(n_ctx=2048)
        case = BenchmarkCase(
            name="multi",
            document="hello " * 100,
            qa_pairs=[QAPair(question="q", expected_answer="hello")],
        )
        report = run_benchmark(
            engine,
            cases=[case],
            budgets=[256],
            strategies=["truncate", "streaming_llm", "evoke"],
            max_gen_tokens=4,
        )
        strategies_seen = {r.strategy for r in report.results}
        assert strategies_seen == {"full", "truncate", "streaming_llm", "evoke"}

    def test_multi_turn_case_inject_turns_not_scored(self):
        engine = MockEngine(n_ctx=4096)
        case = make_multi_turn_case(
            name="recall",
            document="filler " * 100,
            inject_turns=[
                "The secret code is ZEBRA-NINE.",
                "What is the weather like today?",
                "Tell me about something else.",
            ],
            eval_question="What was the secret code?",
            expected_answer="ZEBRA-NINE",
        )
        assert len(case.qa_pairs) == 4
        assert sum(1 for qa in case.qa_pairs if not qa.is_eval) == 3
        assert sum(1 for qa in case.qa_pairs if qa.is_eval) == 1

        report = run_benchmark(
            engine,
            cases=[case],
            budgets=[512],
            strategies=["evoke"],
            max_gen_tokens=8,
            inject_gen_tokens=4,
        )
        evoke_results = [r for r in report.results if r.strategy == "evoke"]
        assert len(evoke_results) == 1
        assert evoke_results[0].question == "What was the secret code?"

    def test_empty_document_case(self):
        engine = MockEngine(n_ctx=4096)
        case = BenchmarkCase(
            name="no_doc",
            document="",
            qa_pairs=[QAPair(question="hello", expected_answer="hi")],
        )
        report = run_benchmark(
            engine,
            cases=[case],
            budgets=[512],
            strategies=["evoke"],
            max_gen_tokens=4,
        )
        assert len(report.results) >= 1

    def test_thinking_template_generates(self):
        engine = MockEngine(n_ctx=4096)
        template = ChatMLThinkingTemplate()
        think_text = "<think>reasoning</think>the answer"
        for _ in range(10):
            engine.queue_tokens([ord(c) for c in think_text])

        case = BenchmarkCase(
            name="think",
            document="doc " * 50,
            qa_pairs=[QAPair(question="what?", expected_answer="answer")],
        )
        report = run_benchmark(
            engine,
            cases=[case],
            budgets=[2048],
            strategies=["evoke"],
            max_gen_tokens=32,
            chat_template=template,
        )
        evoke_results = [r for r in report.results if r.strategy == "evoke"]
        assert len(evoke_results) == 1
        assert "</think>" in evoke_results[0].generated
