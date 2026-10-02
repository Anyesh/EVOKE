import json

from fastapi.testclient import TestClient

from evoke.mock_engine import MockEngine
from evoke.request_dump import RequestDumper, from_env
from evoke.server import create_app


def test_dumper_writes_first_n_and_every_kth(tmp_path):
    d = RequestDumper(tmp_path, first=2, every=3)
    wrote = [d.maybe_dump({"i": i}) for i in range(1, 9)]
    assert wrote == [True, True, True, False, False, True, False, False]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "request-000001.json",
        "request-000002.json",
        "request-000003.json",
        "request-000006.json",
    ]


def test_every_zero_dumps_only_the_first_n(tmp_path):
    d = RequestDumper(tmp_path, first=1, every=0)
    for i in range(5):
        d.maybe_dump({"i": i})
    assert [p.name for p in tmp_path.iterdir()] == ["request-000001.json"]


def test_from_env_is_off_by_default(monkeypatch):
    monkeypatch.delenv("EVOKE_DUMP_REQUESTS_DIR", raising=False)
    assert from_env() is None


def test_from_env_reads_dir_first_and_every(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOKE_DUMP_REQUESTS_DIR", str(tmp_path))
    monkeypatch.setenv("EVOKE_DUMP_FIRST", "4")
    monkeypatch.setenv("EVOKE_DUMP_EVERY", "10")
    d = from_env()
    assert (d.first, d.every) == (4, 10)


def test_server_dumps_the_request_and_the_rendered_prompt(tmp_path):
    app = create_app(
        MockEngine(n_ctx=16384),
        "mock-model",
        request_dump=RequestDumper(tmp_path, 1, 0),
    )
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "it's <b> & done"}],
            "stream": False,
            "max_tokens": 4,
        },
    )
    assert resp.status_code == 200
    data = json.loads((tmp_path / "request-000001.json").read_text())
    assert data["request"]["messages"][0]["content"] == "it's <b> & done"
    assert "it's <b> & done" in data["rendered_prompt"]
    assert data["prompt_tokens"] > 0
