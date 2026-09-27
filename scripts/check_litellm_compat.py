"""Check that litellm and the openai SDK consume the EVOKE server's wire format:
streamed tool-call deltas, SSE keepalive comments during a queue wait, the
include_usage chunk carrying usage.evoke, and X-EVOKE-* response headers.

Run with: uv run --with litellm python scripts/check_litellm_compat.py
Against a live server add --base-url and --api-key; the keepalive check needs
the mock (it stalls one session's prefill to queue a second session).
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
import threading
import time

import httpx
import litellm
import openai

from record_loom_fixtures import (
    MOCK_ADMIN_TOKEN,
    REPLY_TEXT,
    SYSTEM_PROMPT,
    TOOL_CALL_TEXT,
    TOOLS,
    MockServer,
    issue_key,
)

MODEL = "evoke-qwen3-8b"
MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What does main.py do?"},
]
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def litellm_stream_tool_call(base: str, key: str, mock: MockServer | None) -> None:
    if mock:
        mock.script(TOOL_CALL_TEXT)
    chunks = list(
        litellm.completion(
            model=f"openai/{MODEL}",
            api_base=f"{base}/v1",
            api_key=key,
            messages=MESSAGES,
            tools=TOOLS,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers={"X-Evoke-Session": "tenant-acme/litellm-stream"},
            max_tokens=512,
        )
    )
    built = litellm.stream_chunk_builder(chunks, messages=MESSAGES)
    calls = built.choices[0].message.tool_calls or []
    check("litellm stream: one tool call assembled", len(calls) == 1)
    if calls:
        check(
            "litellm stream: tool name and JSON arguments",
            calls[0].function.name == "read_file"
            and isinstance(json.loads(calls[0].function.arguments).get("path"), str),
        )
    check(
        "litellm stream: finish_reason tool_calls",
        built.choices[0].finish_reason == "tool_calls",
    )
    usage_chunks = [c for c in chunks if getattr(c, "usage", None)]
    evoke = None
    if usage_chunks:
        usage = usage_chunks[-1].usage
        evoke = getattr(usage, "evoke", None) or (usage.model_extra or {}).get("evoke")
    check(
        "litellm stream: usage.evoke survives on the usage chunk",
        isinstance(evoke, dict),
        f"usage chunks={len(usage_chunks)}",
    )


def litellm_nonstream_headers(base: str, key: str, mock: MockServer | None) -> None:
    if mock:
        mock.script(REPLY_TEXT)
    resp = litellm.completion(
        model=f"openai/{MODEL}",
        api_base=f"{base}/v1",
        api_key=key,
        messages=MESSAGES,
        extra_headers={"X-Evoke-Session": "tenant-acme/litellm-plain"},
        max_tokens=512,
    )
    usage = resp.usage
    evoke = getattr(usage, "evoke", None) or (usage.model_extra or {}).get("evoke")
    check("litellm non-stream: usage.evoke present", isinstance(evoke, dict))
    headers = resp._hidden_params.get("additional_headers") or {}
    evoke_headers = {k: v for k, v in headers.items() if "x-evoke-" in k.lower()}
    check(
        "litellm non-stream: X-EVOKE-* headers exposed",
        bool(evoke_headers),
        ", ".join(sorted(evoke_headers))[:120],
    )


def openai_stream_usage(base: str, key: str, mock: MockServer | None) -> None:
    if mock:
        mock.script(REPLY_TEXT)
    client = openai.OpenAI(base_url=f"{base}/v1", api_key=key)
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=MESSAGES,
        stream=True,
        stream_options={"include_usage": True},
        extra_headers={"X-Evoke-Session": "tenant-acme/openai-stream"},
        max_tokens=512,
    )
    check(
        "openai SDK: X-EVOKE-Session header on the stream",
        raw.headers.get("x-evoke-session") == "tenant-acme/openai-stream",
    )
    chunks = list(raw.parse())
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    expected_ok = text == REPLY_TEXT if mock else bool(text.strip())
    check("openai SDK: streamed text intact", expected_ok, repr(text[:60]))
    usage = chunks[-1].usage
    check(
        "openai SDK: usage.evoke on the final chunk",
        usage is not None and isinstance((usage.model_extra or {}).get("evoke"), dict),
    )


def keepalive_during_queue_wait(base: str, key: str, mock: MockServer) -> None:
    # Session A's prefill stalls past the server's 5s keepalive interval, so
    # session B's stream sits in the turn queue and receives keepalive comments.
    mock.engine.stall_seconds = 6.0
    mock.script(REPLY_TEXT)
    mock.script(REPLY_TEXT)

    def run_a() -> None:
        litellm.completion(
            model=f"openai/{MODEL}",
            api_base=f"{base}/v1",
            api_key=key,
            messages=MESSAGES,
            extra_headers={"X-Evoke-Session": "tenant-acme/slow-a"},
            max_tokens=512,
        )

    a = threading.Thread(target=run_a)
    a.start()
    time.sleep(0.5)
    start = time.monotonic()
    chunks = list(
        litellm.completion(
            model=f"openai/{MODEL}",
            api_base=f"{base}/v1",
            api_key=key,
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers={"X-Evoke-Session": "tenant-acme/queued-b"},
            max_tokens=512,
        )
    )
    waited = time.monotonic() - start
    a.join()
    text = "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)
    check(
        "litellm: queued stream survives keepalives and completes",
        text == REPLY_TEXT and waited > 5.0,
        f"waited {waited:.1f}s",
    )
    usage = [c for c in chunks if getattr(c, "usage", None)]
    wait_ms = None
    if usage:
        evoke = getattr(usage[-1].usage, "evoke", None) or (
            usage[-1].usage.model_extra or {}
        ).get("evoke")
        wait_ms = (evoke or {}).get("queue_wait_ms")
    check(
        "litellm: queue_wait_ms reflects the wait",
        isinstance(wait_ms, int) and wait_ms > 4000,
        f"queue_wait_ms={wait_ms}",
    )


def run(base: str, key: str, mock: MockServer | None) -> None:
    litellm_stream_tool_call(base, key, mock)
    litellm_nonstream_headers(base, key, mock)
    openai_stream_usage(base, key, mock)
    if mock:
        keepalive_during_queue_wait(base, key, mock)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", help="a non-admin key, for --base-url runs")
    args = parser.parse_args()
    print(f"litellm {version('litellm')}  openai {openai.__version__}")
    if args.base_url:
        run(args.base_url.rstrip("/"), args.api_key, None)
    else:
        with MockServer() as mock:
            base = f"http://127.0.0.1:{mock.port}"
            with httpx.Client() as client:
                token = issue_key(client, base, MOCK_ADMIN_TOKEN, "loom@acme@litellm")
            run(base, token.json()["token"], mock)
    print(f"{len(failures)} failure(s)" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
