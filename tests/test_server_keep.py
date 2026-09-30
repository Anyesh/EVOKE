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


def test_cap_is_a_share_of_the_budget_left_after_the_system_pin():
    client = _client(2000, pin_cap=0.5, pin_system_prompt=True)
    system = {"role": "system", "content": "You are an agent. " * 60}
    small = {"role": "user", "content": "x" * 300, "evoke_keep": "pin"}
    assert _post(client, [system, small]).status_code == 200
    large = {"role": "user", "content": "x" * 600, "evoke_keep": "pin"}
    resp = _post(client, [system, large])
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert 400 < error["pin_cap"] < 460
    assert error["pinned_tokens"] > error["pin_cap"]


def _rendered_len(messages: list[dict]) -> int:
    from evoke.templates import format_qwen_chat

    plain = [{k: v for k, v in m.items() if k != "evoke_keep"} for m in messages]
    return len(format_qwen_chat(plain, add_generation_prompt=False))


def assert_covers_body(pinned: int, whole_message_tokens: int) -> None:
    # A span runs from just after the next-turn opener of the previous message to
    # just after the opener of the following one, so it may drop the role header
    # (up to "user\n") and it takes in the following "\n<|im_start|>".
    assert whole_message_tokens - 20 <= pinned <= whole_message_tokens + 20


def _tool_conversation() -> list[dict]:
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "read main.py"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "print('hi')\n" * 30},
        {"role": "user", "content": "now fix it"},
    ]


def test_pinned_span_is_exactly_the_marked_tool_message():
    client = _client(4000)
    messages = _tool_conversation()
    messages[3]["evoke_keep"] = "pin"
    whole = _rendered_len(messages[:4]) - _rendered_len(messages[:3])
    evoke = _post(client, messages).json()["usage"]["evoke"]
    assert_covers_body(evoke["pinned_tokens"], whole)


def test_adjacent_marked_messages_add_up():
    client = _client(4000)
    messages = _tool_conversation()
    messages[3]["evoke_keep"] = "pin"
    messages[4]["evoke_keep"] = "pin"
    whole = _rendered_len(messages[:5]) - _rendered_len(messages[:3])
    evoke = _post(client, messages).json()["usage"]["evoke"]
    assert_covers_body(evoke["pinned_tokens"], whole)


def test_streaming_request_honours_marks_and_the_cap():
    client = _client(4000)
    messages = _tool_conversation()
    messages[3]["evoke_keep"] = "pin"
    body = {
        "model": "evoke-mock",
        "messages": messages,
        "max_tokens": 4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = client.post("/v1/chat/completions", json=body, headers=SESSION)
    assert resp.status_code == 200
    usage_chunks = [
        line
        for line in resp.text.splitlines()
        if line.startswith("data: {") and '"usage"' in line
    ]
    assert usage_chunks
    import json

    evoke = json.loads(usage_chunks[-1][len("data: ") :])["usage"]["evoke"]
    assert evoke["pinned_tokens"] > 0

    tight = _client(200, pin_cap=0.1)
    refused = tight.post("/v1/chat/completions", json=body, headers=SESSION)
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "evoke_pin_budget_exceeded"


def test_mark_on_a_message_that_is_not_resident_yet_and_a_repeat_do_not_double_count():
    client = _client(4000)
    messages = _tool_conversation()
    messages[3]["evoke_keep"] = "pin"
    first = _post(client, messages).json()["usage"]["evoke"]
    again = _post(client, messages).json()["usage"]["evoke"]
    assert again["pinned_tokens"] == first["pinned_tokens"]


def test_healthz_says_whether_marks_reach_resident_blocks():
    assert _client(4000).get("/healthz").json()["keep_marks_on_resident"] is True
    compact = EvokeConfig(max_active_tokens=4000, position_mode="compact")
    engine = MockEngine(n_ctx=16384)
    client = TestClient(create_app(engine, "evoke-mock", compact))
    assert client.get("/healthz").json()["keep_marks_on_resident"] is False
