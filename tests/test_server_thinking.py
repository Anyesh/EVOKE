"""create_app(enable_thinking=...) must reach the template render even when the
request carries no tools; the C template path cannot carry the flag, so a
tool-less request that skips the with_tools renderer silently re-enables
thinking on Qwen3 models."""

from __future__ import annotations

from fastapi.testclient import TestClient

from evoke.mock_engine import MockEngine
from evoke.server import create_app


class _RecordingEngine(MockEngine):
    def __init__(self):
        super().__init__(n_ctx=16384)
        self.with_tools_calls: list[bool | None] = []

    def apply_chat_template_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        self.with_tools_calls.append(enable_thinking)
        return "".join(m.get("content") or "" for m in messages) + "\nassistant:"


def test_enable_thinking_false_reaches_render_without_tools():
    engine = _RecordingEngine()
    app = create_app(engine, "mock-model", enable_thinking=False)
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 8,
        },
    )
    assert resp.status_code == 200
    assert engine.with_tools_calls == [False]


def test_request_chat_template_kwargs_cannot_turn_thinking_back_on():
    engine = _RecordingEngine()
    client = TestClient(create_app(engine, "mock-model", enable_thinking=False))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    assert resp.status_code == 200
    assert engine.with_tools_calls == [False]
