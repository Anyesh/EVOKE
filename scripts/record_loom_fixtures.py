"""Record wire fixtures of the EVOKE server for Loom's provider integration.

Against a live server:  --base-url http://host:port --admin-key TOKEN
The admin key issues a per-person key the way loomd does, every turn runs
under that key, and the key is revoked at the end. Without --base-url the script serves create_app over a MockEngine on a local
port, so the fixtures carry the real server's framing and headers while the
token counts come from the mock's one-char-per-token tokenizer.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evoke.auth import IssuedKeyStore, KeyRing
from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.server import create_app

MOCK_ADMIN_TOKEN = "loomd-fixture-admin-token"
FIXTURE_KEY_ID = "loom@tenant-acme@user-0001"
SYSTEM_PROMPT = (
    "You are Loom's coding agent. Work in /work, call tools to read and edit "
    "files, and answer briefly. " * 12
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
TOOL_CALL_TEXT = (
    '<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>'
)
REPLY_TEXT = "main.py defines the CLI entry point and one helper."


class _StallingMockEngine(MockEngine):
    def __init__(self, n_ctx: int) -> None:
        super().__init__(n_ctx=n_ctx)
        self.stall_seconds = 0.0

    def process_tokens(self, tokens: list[int]) -> None:
        if self.stall_seconds:
            stall, self.stall_seconds = self.stall_seconds, 0.0
            time.sleep(stall)
        super().process_tokens(tokens)


class MockServer:
    def __init__(self, budget_aware: bool = False) -> None:
        self.engine = _StallingMockEngine(n_ctx=16384)
        config = EvokeConfig(
            max_active_tokens=2560,
            block_size=64,
            sink_count=0,
            recovery_mode="kv_restore",
            recovery_match="identity",
            position_mode="sparse",
            gap_fill_budget_aware=budget_aware,
        )
        self._store_dir = tempfile.TemporaryDirectory()
        keyring = KeyRing.from_tokens(
            {"loomd": (MOCK_ADMIN_TOKEN, True)},
            issued_store=IssuedKeyStore(Path(self._store_dir.name) / "issued.json"),
        )
        app = create_app(
            self.engine,
            "evoke-qwen3-8b",
            config=config,
            keyring=keyring,
            pin_system_prompt=True,
            pin_cap=0.9,
            queue_timeout=300.0,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "MockServer":
        self._thread.start()
        while not self._server.started:
            time.sleep(0.02)
        return self

    def __exit__(self, *_exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
        self._store_dir.cleanup()

    def script(self, text: str) -> None:
        self.engine.queue_tokens([ord(c) for c in text] + [self.engine.eos_token])


def _write(out: Path, name: str, payload) -> None:
    path = out / name
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def _evoke_headers(resp: httpx.Response) -> dict[str, str]:
    keep = {"content-type", "retry-after"}
    return {
        k: v for k, v in resp.headers.items() if k.startswith("x-evoke-") or k in keep
    }


def _record(resp: httpx.Response, body) -> dict:
    return {"status": resp.status_code, "headers": _evoke_headers(resp), "body": body}


def issue_key(
    client: httpx.Client, base_url: str, admin_key: str, key_id: str
) -> httpx.Response:
    return client.post(
        f"{base_url}/admin/keys",
        json={"key_id": key_id},
        headers={"Authorization": f"Bearer {admin_key}"},
    )


def _redacted(record: dict) -> dict:
    body = dict(record["body"])
    if "token" in body:
        body["token"] = body["token"][:4] + "<redacted>"
    return {**record, "body": body}


def _messages(*extra: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What does main.py do?"},
        *extra,
    ]


def _blob(tokens: int, chars_per_token: int, label: str) -> str:
    unit = f"{label} output line. "
    return (unit * (tokens * chars_per_token // len(unit) + 1))[
        : tokens * chars_per_token
    ]


def record_keep(
    client: httpx.Client,
    base_url: str,
    session: dict,
    out: Path,
    mock: MockServer | None,
) -> None:
    chat = f"{base_url}/v1/chat/completions"
    chars_per_token = 1 if mock else 4
    k_session = {**session, "X-Evoke-Session": "tenant-acme/session-keep"}

    def post(messages: list[dict]) -> httpx.Response:
        if mock:
            mock.script(REPLY_TEXT)
        body = {"model": "evoke-qwen3-8b", "messages": messages, "max_tokens": 64}
        return client.post(chat, json=body, headers=k_session)

    health = client.get(f"{base_url}/healthz").json()
    probe = post(_messages())
    pinned_sys = probe.json()["usage"]["evoke"]["pinned_tokens"]
    available = health["budget"] - pinned_sys
    cap = int(health["pin_cap_fraction"] * available)
    blob_tokens = available // 3

    def tool(i: int, keep: str | None = None, tokens: int = blob_tokens) -> dict:
        message = {
            "role": "user",
            "content": f"Tool result {i}:\n" + _blob(tokens, chars_per_token, f"t{i}"),
        }
        if keep:
            message["evoke_keep"] = keep
        return message

    r = post(_messages(tool(0, "pin")))
    _write(out, "keep_marked_request.json", _record(r, r.json()))

    r = post(_messages(tool(1, "pin", tokens=cap * 3 // 2)))
    _write(out, "keep_pin_cap_refusal.json", _record(r, r.json()))

    history = [tool(i) for i in range(6)]
    for upto in range(1, len(history) + 1):
        r = post(_messages(*history[:upto]))
    last = r.json()["usage"]["evoke"]
    print(f"keep: blocks_evicted={last['blocks_evicted']} before the marked replay")
    history[0]["evoke_keep"] = "pin"
    r = post(_messages(*history))
    _write(out, "keep_pinned_restored.json", _record(r, r.json()))
    recovered = r.json()["usage"]["evoke"]["blocks_recovered"]
    print(f"keep: blocks_recovered={recovered} on the marked replay")


def record(base_url: str, admin_key: str, out: Path, mock: MockServer | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    admin = {"Authorization": f"Bearer {admin_key}"}
    chat = f"{base_url}/v1/chat/completions"
    with httpx.Client(timeout=600) as client:
        _write(
            out,
            "healthz.json",
            _record(r := client.get(f"{base_url}/healthz"), r.json()),
        )

        r = issue_key(client, base_url, admin_key, FIXTURE_KEY_ID)
        issued = _record(r, r.json())
        _write(out, "admin_issue_key.json", _redacted(issued))
        auth = {"Authorization": f"Bearer {issued['body']['token']}"}
        session = {"X-Evoke-Session": "tenant-acme/session-0001", **auth}

        r = client.get(f"{base_url}/v1/models", headers=auth)
        _write(out, "models.json", _record(r, r.json()))

        body = {"model": "evoke-qwen3-8b", "messages": _messages(), "max_tokens": 512}
        r = client.post(chat, json=body, headers=auth)
        _write(out, "headerless_rejection.json", _record(r, r.json()))
        r = client.post(chat, json=body, headers={"X-Evoke-Session": "tenant-acme/s"})
        _write(out, "unauthenticated_rejection.json", _record(r, r.json()))

        if mock:
            mock.script(TOOL_CALL_TEXT)
        turn1 = {
            **body,
            "tools": TOOLS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        with client.stream("POST", chat, json=turn1, headers=session) as r:
            raw = "".join(r.iter_text())
        _write(out, "stream_turn1_tool_call.sse", raw)
        _write(out, "stream_turn1_tool_call.headers.json", _evoke_headers(r))

        if mock:
            mock.script(REPLY_TEXT)
        turn2 = {
            **turn1,
            "messages": _messages(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "main.py"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "content": "import sys\n\ndef main():\n    ...\n" * 40,
                },
            ),
        }
        with client.stream("POST", chat, json=turn2, headers=session) as r:
            raw = "".join(r.iter_text())
        _write(out, "stream_turn2_text.sse", raw)
        _write(out, "stream_turn2_text.headers.json", _evoke_headers(r))

        if mock:
            mock.script(REPLY_TEXT)
        turn3 = {**turn2, "stream": False}
        turn3.pop("stream_options")
        turn3["messages"] = turn2["messages"] + [
            {"role": "assistant", "content": REPLY_TEXT},
            {"role": "user", "content": "Thanks."},
        ]
        r = client.post(chat, json=turn3, headers=session)
        _write(out, "nonstream_turn3.json", _record(r, r.json()))

        if not mock:
            record_keep(client, base_url, auth, out, None)

        r = client.delete(f"{base_url}/admin/keys/{FIXTURE_KEY_ID}", headers=admin)
        _write(out, "admin_revoke_key.json", _record(r, r.json()))
        r = client.post(chat, json=turn3, headers=session)
        _write(out, "revoked_key_rejection.json", _record(r, r.json()))


def record_keep_standalone(
    base_url: str, admin_key: str, out: Path, mock: MockServer
) -> None:
    admin = {"Authorization": f"Bearer {admin_key}"}
    with httpx.Client(timeout=600) as client:
        r = issue_key(client, base_url, admin_key, FIXTURE_KEY_ID)
        auth = {"Authorization": f"Bearer {r.json()['token']}"}
        record_keep(client, base_url, auth, out, mock)
        client.delete(f"{base_url}/admin/keys/{FIXTURE_KEY_ID}", headers=admin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--admin-key", default=MOCK_ADMIN_TOKEN)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.base_url:
        record(args.base_url.rstrip("/"), args.admin_key, args.out, None)
        return 0
    with MockServer() as mock:
        record(f"http://127.0.0.1:{mock.port}", MOCK_ADMIN_TOKEN, args.out, mock)
    # The keep fixtures need budget-aware gap-fill, or every evicted block is
    # restored on the marked replay and the pin shows nothing.
    with MockServer(budget_aware=True) as mock:
        record_keep_standalone(
            f"http://127.0.0.1:{mock.port}", MOCK_ADMIN_TOKEN, args.out, mock
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
