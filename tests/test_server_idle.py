import time
from pathlib import Path

from fastapi.testclient import TestClient

from evoke.mock_engine import MockEngine
from evoke.server import create_app

MODEL = "evoke-qwen3-8b"

CHAT_BODY = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 4,
    "stream": False,
}


def _build_app(tmp_path: Path, engines: list[MockEngine], idle_timeout=None):
    (tmp_path / f"{MODEL}.gguf").touch()

    def factory(path: str) -> MockEngine:
        engine = MockEngine(n_ctx=16384)
        engines.append(engine)
        return engine

    first = MockEngine(n_ctx=16384)
    engines.append(first)
    return create_app(
        first,
        MODEL,
        model_dir=tmp_path,
        model_path=str(tmp_path / f"{MODEL}.gguf"),
        engine_factory=factory,
        idle_timeout=idle_timeout,
    )


def test_unload_endpoint_closes_engine(tmp_path):
    engines: list[MockEngine] = []
    client = TestClient(_build_app(tmp_path, engines))
    resp = client.post("/models/unload")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": MODEL, "unloaded": True}
    assert engines[0].closed
    health = client.get("/health").json()
    assert health["model_loaded"] is False
    assert health["n_ctx"] == 16384


def test_unload_when_already_unloaded_is_noop(tmp_path):
    engines: list[MockEngine] = []
    client = TestClient(_build_app(tmp_path, engines))
    client.post("/models/unload")
    resp = client.post("/models/unload")
    assert resp.status_code == 200
    assert resp.json()["unloaded"] is False


def test_unload_unconfigured_404():
    client = TestClient(create_app(MockEngine(n_ctx=16384), MODEL))
    assert client.post("/models/unload").status_code == 404


def test_chat_reloads_after_unload(tmp_path):
    engines: list[MockEngine] = []
    client = TestClient(_build_app(tmp_path, engines))
    client.post("/models/unload")
    resp = client.post("/v1/chat/completions", json=CHAT_BODY)
    assert resp.status_code == 200
    assert len(engines) == 2
    assert not engines[1].closed
    assert client.get("/health").json()["model_loaded"] is True


def test_readonly_endpoints_do_not_reload(tmp_path):
    engines: list[MockEngine] = []
    client = TestClient(_build_app(tmp_path, engines))
    client.post("/models/unload")
    assert client.get("/health").status_code == 200
    assert client.get("/models").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/sessions").status_code == 200
    assert len(engines) == 1
    assert client.get("/health").json()["model_loaded"] is False


def test_native_models_reports_ctx_while_unloaded(tmp_path):
    # Calcifer parses --ctx-size from the active /models entry to size its
    # prompts; that must not disappear while the model is idle-unloaded,
    # because a reload restores exactly the same context window.
    engines: list[MockEngine] = []
    client = TestClient(_build_app(tmp_path, engines))
    client.post("/models/unload")
    entry = next(m for m in client.get("/models").json()["data"] if m["id"] == MODEL)
    args = entry["status"]["args"]
    assert args[args.index("--ctx-size") + 1] == "16384"


def test_models_load_same_name_reloads_when_unloaded(tmp_path):
    engines: list[MockEngine] = []
    client = TestClient(_build_app(tmp_path, engines))
    client.post("/models/unload")
    resp = client.post("/models/load", json={"model": MODEL})
    assert resp.status_code == 200
    assert resp.json()["loaded"] is True
    assert len(engines) == 2
    assert client.get("/health").json()["model_loaded"] is True


def test_admin_reset_while_unloaded_is_noop(tmp_path):
    engines: list[MockEngine] = []
    client = TestClient(_build_app(tmp_path, engines))
    client.post("/models/unload")
    resp = client.post("/admin/reset")
    assert resp.status_code == 200
    assert len(engines) == 1


def test_idle_watcher_unloads_after_timeout(tmp_path):
    engines: list[MockEngine] = []
    app = _build_app(tmp_path, engines, idle_timeout=0.2)
    # The watcher task only runs under the app lifespan, so the context
    # manager form is required here.
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if client.get("/health").json()["model_loaded"] is False:
                break
            time.sleep(0.05)
        assert client.get("/health").json()["model_loaded"] is False
        assert engines[0].closed
        resp = client.post("/v1/chat/completions", json=CHAT_BODY)
        assert resp.status_code == 200
        assert client.get("/health").json()["model_loaded"] is True
        assert len(engines) == 2
