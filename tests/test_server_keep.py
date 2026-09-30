"""Per-message evoke_keep marks over the wire: validation, the pin cap, and a
pinned message restored after eviction."""

from __future__ import annotations

from fastapi.testclient import TestClient

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.server import create_app

SESSION = {"X-Evoke-Session": "acme/keep"}
BLOB = "line of tool output. " * 20


def _config(budget: int) -> EvokeConfig:
    return EvokeConfig(
        max_active_tokens=budget,
        eviction_policy="threshold",
        block_size=16,
        sink_count=0,
        position_mode="sparse",
        recovery_mode="kv_restore",
        recovery_match="identity",
        gap_fill_budget_aware=True,
    )


def _client(budget: int, **kwargs) -> TestClient:
    engine = MockEngine(n_ctx=16384)
    return TestClient(create_app(engine, "evoke-mock", _config(budget), **kwargs))


def _post(client: TestClient, messages: list[dict]) -> dict:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "evoke-mock", "messages": messages, "max_tokens": 4},
        headers=SESSION,
    )
    return resp


def _history(turns: int) -> list[dict]:
    return [{"role": "system", "content": "You are an agent."}] + [
        {"role": "user", "content": f"tool {i}: {BLOB}"} for i in range(turns)
    ]


def test_unknown_keep_value_is_400():
    client = _client(4000)
    messages = _history(1)
    messages[1]["evoke_keep"] = "always"
    resp = _post(client, messages)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "evoke_keep_invalid"


def test_marked_request_reports_pinned_tokens():
    client = _client(4000)
    messages = _history(2)
    messages[1]["evoke_keep"] = "pin"
    resp = _post(client, messages)
    assert resp.status_code == 200
    evoke = resp.json()["usage"]["evoke"]
    assert evoke["pinned_tokens"] >= len(BLOB)


def test_pin_past_the_cap_is_refused_with_the_numbers():
    client = _client(200, pin_cap=0.6)
    messages = _history(2)
    messages[1]["evoke_keep"] = "pin"
    resp = _post(client, messages)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "evoke_pin_budget_exceeded"
    assert error["pin_cap"] == 120
    assert error["pinned_tokens"] > 120


def test_prefer_is_not_counted_against_the_cap():
    client = _client(200, pin_cap=0.6)
    messages = _history(2)
    messages[1]["evoke_keep"] = "prefer"
    assert _post(client, messages).status_code == 200


def test_pinned_message_is_restored_after_it_was_evicted():
    client = _client(1000)
    messages = _history(1)
    for i in range(1, 6):
        assert _post(client, messages).status_code == 200
        messages.append({"role": "user", "content": f"tool {i}: {BLOB}"})
    evicted = _post(client, messages).json()["usage"]["evoke"]
    assert evicted["blocks_evicted"] + evicted["tokens_evicted"] > 0

    messages[1]["evoke_keep"] = "pin"
    restored = _post(client, messages).json()["usage"]["evoke"]
    assert restored["blocks_recovered"] > 0
    assert restored["pinned_tokens"] >= len(BLOB)
