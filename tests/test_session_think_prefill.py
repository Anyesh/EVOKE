"""The empty <think>\\n\\n</think>\\n\\n pair that Qwen3 templates inject into the
generation prompt when enable_thinking=false must not persist in the cache.

The template adds the pair only at the generation position; the same assistant
turn re-renders WITHOUT it in the next request's history, so a cached pair
diverges the prefix at the newest assistant message on every turn. The pair
must be decoded (the model needs to see the closed block to answer directly)
but evicted at end of turn, exactly like a generated thinking trace.
"""

from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import Session

THINK_PAIR = "<think>\n\n</think>\n\n"


def _make_session() -> Session:
    engine = MockEngine()
    cfg = EvokeConfig(
        max_active_tokens=100_000,
        block_size=16,
        sink_count=0,
        recovery_mode="discard",
    )
    return Session(engine, config=cfg)


class TestPromptThinkPrefill:
    def test_injected_pair_evicted_after_generate(self):
        session = _make_session()
        engine = session._engine
        body = "system stuff<|im_start|>assistant\n"
        prompt = engine.tokenize(body + THINK_PAIR)
        answer = "The answer is 42."

        session.sync_prefix(prompt)
        engine.queue_tokens(engine.tokenize(answer) + [engine.eos_token])
        session.generate(max_tokens=64)

        cached = engine.detokenize(session._cached_tokens)
        assert "<think>" not in cached
        assert cached.startswith(body)
        assert answer in cached

    def test_next_turn_extends_without_divergence(self):
        session = _make_session()
        engine = session._engine
        body = "system stuff<|im_start|>assistant\n"
        answer = "The answer is 42."

        session.sync_prefix(engine.tokenize(body + THINK_PAIR))
        answer_tokens = engine.tokenize(answer)
        engine.queue_tokens(answer_tokens + [engine.eos_token])
        session.generate(max_tokens=64)

        # The client echoes the stripped history: no pair on the previous
        # assistant turn, a fresh pair on the new generation prompt.
        turn2 = engine.tokenize(body + answer) + [engine.eos_token]
        user2 = engine.tokenize("\nuser follow-up<|im_start|>assistant\n")
        pair = engine.tokenize(THINK_PAIR)
        stats = session.sync_prefix(turn2 + user2 + pair)

        assert stats.new_tokens_decoded == len(user2) + len(pair), (
            "turn 2 re-decoded history: the cached pair diverged the prefix "
            f"(new_tokens_decoded={stats.new_tokens_decoded})"
        )
