"""Multi-tenant serving mode: bearer keys, key-bound sessions, mandatory session
header, per-turn usage.evoke metrics and the lock-free /healthz probe."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from evoke.auth import KeyRing
from evoke.mock_engine import MockEngine
from evoke.server import create_app

LOOM = "loom-token-0123"
OTHER = "other-token-0123"
OPS = "ops-token-012345"

EVOKE_FIELDS = {
    "session",
    "logical_tokens",
    "logical_window",
    "kv_cells",
    "peak_resident_tokens",
    "resident_tokens_end",
    "pinned_tokens",
    "budget",
    "blocks_evicted",
    "tokens_evicted",
    "blocks_recovered",
    "tokens_recovered",
    "prompt_tokens_decoded",
    "prompt_tokens_reused",
    "queue_wait_ms",
}


def _keyring() -> KeyRing:
    return KeyRing.from_tokens(
        {"loom": (LOOM, False), "other": (OTHER, False), "ops": (OPS, True)}
    )


def _client(**kwargs) -> TestClient:
    kwargs.setdefault("keyring", _keyring())
    return TestClient(create_app(MockEngine(n_ctx=16384), "evoke-mock", **kwargs))


def _auth(token: str, session: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if session is not None:
        headers["X-Evoke-Session"] = session
    return headers


def _body(**extra) -> dict:
    return {
        "model": "evoke-mock",
        "messages": [
            {"role": "system", "content": "You are a careful agent. " * 20},
            {"role": "user", "content": "list the files"},
        ],
        "max_tokens": 8,
        **extra,
    }


def _sse_events(text: str) -> list[dict | str]:
    events: list[dict | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_missing_or_bad_key_is_401_with_openai_error_shape():
    client = _client()
    for headers in ({}, {"Authorization": "Bearer wrong-token-xyz"}):
        resp = client.post("/v1/chat/completions", json=_body(), headers=headers)
        assert resp.status_code == 401
        err = resp.json()["error"]
        assert err["type"] == "authentication_error"
        assert err["code"] == "invalid_api_key"
    assert client.get("/v1/models").status_code == 401


def test_headerless_request_is_rejected_not_routed():
    client = _client()
    resp = client.post("/v1/chat/completions", json=_body(), headers=_auth(LOOM))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "evoke_session_required"


def test_malformed_session_header_is_rejected():
    client = _client()
    resp = client.post(
        "/v1/chat/completions", json=_body(), headers=_auth(LOOM, "bad value")
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "evoke_session_invalid"


def test_non_stream_turn_reports_usage_evoke_and_headers():
    client = _client()
    resp = client.post(
        "/v1/chat/completions", json=_body(), headers=_auth(LOOM, "acme/s1")
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    evoke = usage["evoke"]
    assert set(evoke) == EVOKE_FIELDS
    assert evoke["session"] == "acme/s1"
    assert (
        evoke["logical_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    )
    assert evoke["kv_cells"] == 16384
    assert evoke["peak_resident_tokens"] >= evoke["resident_tokens_end"]
    assert evoke["prompt_tokens_decoded"] == usage["prompt_tokens"]
    assert resp.headers["X-EVOKE-Session"] == "acme/s1"
    assert int(resp.headers["X-EVOKE-Logical-Tokens"]) == evoke["logical_tokens"]
    assert (
        int(resp.headers["X-EVOKE-Peak-Resident-Tokens"])
        == evoke["peak_resident_tokens"]
    )


def test_second_turn_reuses_the_resident_prefix():
    client = _client()
    first = client.post(
        "/v1/chat/completions", json=_body(), headers=_auth(LOOM, "acme/s1")
    ).json()
    reply = first["choices"][0]["message"]["content"]
    body = _body()
    body["messages"] += [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "now read main.py"},
    ]
    second = client.post(
        "/v1/chat/completions", json=body, headers=_auth(LOOM, "acme/s1")
    ).json()
    evoke = second["usage"]["evoke"]
    assert evoke["prompt_tokens_reused"] > 0
    assert evoke["prompt_tokens_decoded"] < second["usage"]["prompt_tokens"]


def test_stream_with_include_usage_ends_with_usage_chunk():
    client = _client()
    resp = client.post(
        "/v1/chat/completions",
        json=_body(stream=True, stream_options={"include_usage": True}),
        headers=_auth(LOOM, "acme/s1"),
    )
    assert resp.status_code == 200
    assert resp.headers["X-EVOKE-Session"] == "acme/s1"
    events = _sse_events(resp.text)
    assert events[-1] == "[DONE]"
    usage_chunk = events[-2]
    assert usage_chunk["choices"] == []
    assert set(usage_chunk["usage"]["evoke"]) == EVOKE_FIELDS
    finish = events[-3]
    assert finish["choices"][0]["finish_reason"] is not None


def test_stream_without_include_usage_has_no_usage_chunk():
    client = _client()
    resp = client.post(
        "/v1/chat/completions",
        json=_body(stream=True),
        headers=_auth(LOOM, "acme/s1"),
    )
    events = _sse_events(resp.text)
    assert all("usage" not in e for e in events if isinstance(e, dict))


def test_same_session_header_under_different_keys_is_two_sessions():
    client = _client()
    for token in (LOOM, OTHER):
        resp = client.post(
            "/v1/chat/completions", json=_body(), headers=_auth(token, "acme/s1")
        )
        assert resp.status_code == 200
    mine = client.get("/v1/sessions", headers=_auth(LOOM)).json()
    assert mine["sessions"] == ["acme/s1"]
    everything = client.get("/v1/sessions", headers=_auth(OPS)).json()
    assert sorted(everything["sessions"]) == ["loom|acme/s1", "other|acme/s1"]


def test_delete_only_reaches_the_callers_own_session():
    client = _client()
    client.post("/v1/chat/completions", json=_body(), headers=_auth(OTHER, "acme/s1"))
    resp = client.delete("/v1/sessions/acme%2Fs1", headers=_auth(LOOM))
    assert resp.json()["status"] == "not_found"
    assert client.get("/v1/sessions", headers=_auth(OTHER)).json()["sessions"] == [
        "acme/s1"
    ]


def test_admin_endpoints_need_an_admin_key():
    client = _client()
    resp = client.post("/admin/set_budget", json={"tokens": 1024}, headers=_auth(LOOM))
    assert resp.status_code == 403
    assert resp.json()["error"]["type"] == "permission_error"
    ok = client.post("/admin/set_budget", json={"tokens": 1024}, headers=_auth(OPS))
    assert ok.status_code == 200


def test_healthz_is_open_and_leaks_no_session_data():
    client = _client()
    client.post("/v1/chat/completions", json=_body(), headers=_auth(LOOM, "acme/s1"))
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["auth_required"] is True
    assert body["session_required"] is True
    assert body["kv_cells"] == 16384
    assert body["queue"] == {"running": False, "waiting": 0}
    assert "acme" not in json.dumps(body)


def test_pinning_covers_the_system_prompt():
    client = _client(pin_system_prompt=True)
    resp = client.post(
        "/v1/chat/completions", json=_body(), headers=_auth(LOOM, "acme/s1")
    )
    evoke = resp.json()["usage"]["evoke"]
    system_chars = len("You are a careful agent. " * 20)
    assert evoke["pinned_tokens"] >= system_chars
    assert evoke["pinned_tokens"] < resp.json()["usage"]["prompt_tokens"]


def test_open_mode_keeps_headerless_routing():
    client = TestClient(create_app(MockEngine(n_ctx=16384), "evoke-mock"))
    resp = client.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200
    assert resp.json()["usage"]["evoke"]["session"].startswith("auto-")
    assert client.get("/healthz").json()["auth_required"] is False


def test_models_entry_advertises_the_context_window():
    client = _client()
    entry = client.get("/v1/models", headers=_auth(LOOM)).json()["data"][0]
    assert entry["context_length"] == 16384
    assert entry["evoke"]["kv_cells"] == 16384
    assert entry["evoke"]["logical_window"] == 16384


class _BrokenEngine(MockEngine):
    def process_tokens(self, tokens: list[int]) -> None:
        raise RuntimeError("llama_decode failed with code -3")


def test_engine_failure_is_an_openai_shaped_500():
    app = create_app(_BrokenEngine(n_ctx=16384), "evoke-mock", keyring=_keyring())
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions", json=_body(), headers=_auth(LOOM, "acme/s1")
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "evoke_engine_error"


def test_engine_failure_mid_stream_ends_with_an_error_event():
    app = create_app(_BrokenEngine(n_ctx=16384), "evoke-mock", keyring=_keyring())
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json=_body(stream=True),
        headers=_auth(LOOM, "acme/s1"),
    )
    events = _sse_events(resp.text)
    assert events[-1] == "[DONE]"
    assert events[-2]["error"]["code"] == "evoke_engine_error"


def test_validation_error_carries_an_openai_error_object():
    client = _client()
    resp = client.post(
        "/v1/chat/completions", json={"model": "m"}, headers=_auth(LOOM, "acme/s1")
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "invalid_request_error"
