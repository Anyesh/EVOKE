from pathlib import Path

from fastapi.testclient import TestClient

from evoke.mock_engine import MockEngine
from evoke.server import create_app


def _seed_model_dir(tmp_path: Path) -> Path:
    for name in ("evoke-qwen3-8b", "other-model", "mmproj-F16", "qwen-mmproj-F16"):
        (tmp_path / f"{name}.gguf").touch()
    return tmp_path


def _client(tmp_path: Path, factory_paths: list[str] | None = None) -> TestClient:
    model_dir = _seed_model_dir(tmp_path)

    def factory(path: str) -> MockEngine:
        if factory_paths is not None:
            factory_paths.append(path)
        return MockEngine(n_ctx=16384)

    app = create_app(
        MockEngine(n_ctx=16384),
        "evoke-qwen3-8b",
        model_dir=model_dir,
        model_path=str(model_dir / "evoke-qwen3-8b.gguf"),
        engine_factory=factory,
    )
    return TestClient(app)


def test_native_models_reports_ctx_size(tmp_path):
    # Clients built against a llama.cpp server manager (calcifer among them)
    # probe GET /models and parse --ctx-size from the active entry's launch
    # args to learn the real context window; without this endpoint they fall
    # back to a guessed default and overrun n_ctx.
    client = _client(tmp_path)
    resp = client.get("/models")
    assert resp.status_code == 200
    entry = next(m for m in resp.json()["data"] if m["id"] == "evoke-qwen3-8b")
    args = entry["status"]["args"]
    assert args[args.index("--ctx-size") + 1] == "16384"


def test_native_models_lists_dir_without_projectors(tmp_path):
    client = _client(tmp_path)
    ids = [m["id"] for m in client.get("/models").json()["data"]]
    assert ids == ["evoke-qwen3-8b", "other-model"]


def test_models_load_switches_engine(tmp_path):
    factory_paths: list[str] = []
    client = _client(tmp_path, factory_paths)
    resp = client.post("/models/load", json={"model": "other-model"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "other-model", "loaded": True}
    assert factory_paths == [str(tmp_path / "other-model.gguf")]
    assert client.get("/v1/models").json()["data"][0]["id"] == "other-model"
    entry = next(
        m for m in client.get("/models").json()["data"] if m["id"] == "other-model"
    )
    assert "--ctx-size" in entry["status"]["args"]


def test_models_load_same_model_is_noop(tmp_path):
    factory_paths: list[str] = []
    client = _client(tmp_path, factory_paths)
    resp = client.post("/models/load", json={"model": "evoke-qwen3-8b"})
    assert resp.status_code == 200
    assert resp.json()["loaded"] is False
    assert factory_paths == []


def test_models_load_unknown_model_404(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/models/load", json={"model": "no-such-model"})
    assert resp.status_code == 404


def test_models_load_unconfigured_404():
    app = create_app(MockEngine(n_ctx=16384), "evoke-qwen3-8b")
    client = TestClient(app)
    resp = client.post("/models/load", json={"model": "anything"})
    assert resp.status_code == 404


def test_openai_models_unchanged(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "evoke-qwen3-8b"
