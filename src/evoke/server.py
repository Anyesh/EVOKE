"""FastAPI server exposing EVOKE as an OpenAI-compatible /v1/chat/completions
endpoint. The persistent KV cache survives between requests; only the new tail
of the message history is decoded each turn (see Session.sync_prefix). This
lets any OpenAI-compatible agent harness (opencode, Aider, OpenHands) drive
EVOKE without code changes on their side.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from evoke.llama_engine import LlamaCppEngine
from evoke.session import Session
from evoke.templates import ParsedResponse, format_qwen_chat, parse_qwen_response


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    max_tokens: int | None = Field(default=2048)
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None
    stream: bool = False


def _normalize_stops(stop: list[str] | str | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _build_choice_message(parsed: ParsedResponse) -> dict[str, Any]:
    if parsed.tool_calls:
        return {
            "role": "assistant",
            "content": parsed.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in parsed.tool_calls
            ],
        }
    return {"role": "assistant", "content": parsed.content}


def _completion_payload(
    completion_id: str,
    created: int,
    model: str,
    parsed: ParsedResponse,
    prompt_token_count: int,
    completion_token_count: int,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": _build_choice_message(parsed),
                "finish_reason": (
                    "tool_calls" if parsed.tool_calls else parsed.finish_reason
                ),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": prompt_token_count + completion_token_count,
        },
    }


def _chunk_payload(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def create_app(engine: LlamaCppEngine, model_name: str) -> FastAPI:
    app = FastAPI(title="EVOKE", version="0.1.0")
    session = Session(engine)
    lock = asyncio.Lock()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "evoke",
                }
            ],
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "cached_tokens": session.cached_token_count,
            "n_ctx": engine.n_ctx,
        }

    @app.post("/admin/reset")
    async def reset_session() -> dict[str, Any]:
        async with lock:
            session.reset()
        return {"status": "reset"}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        msgs = [m.model_dump(exclude_none=True) for m in req.messages]
        prompt = format_qwen_chat(msgs, tools=req.tools, add_generation_prompt=True)
        prompt_tokens = engine.tokenize(prompt)
        prompt_n = len(prompt_tokens)

        stops = _normalize_stops(req.stop)
        if "<|im_end|>" not in stops:
            stops.append("<|im_end|>")

        max_new = req.max_tokens or 2048
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream_completion(
                    session,
                    engine,
                    lock,
                    prompt_tokens,
                    stops,
                    max_new,
                    completion_id,
                    created,
                    model_name,
                ),
                media_type="text/event-stream",
            )

        async with lock:
            session.sync_prefix(prompt_tokens)
            result = session.generate(max_tokens=max_new, stop_strings=stops)

        parsed = parse_qwen_response(result.text)
        return _completion_payload(
            completion_id,
            created,
            model_name,
            parsed,
            prompt_n,
            len(result.output_tokens),
        )

    return app


async def _stream_completion(
    session: Session,
    engine: LlamaCppEngine,
    lock: asyncio.Lock,
    prompt_tokens: list[int],
    stops: list[str],
    max_new: int,
    completion_id: str,
    created: int,
    model_name: str,
):
    async with lock:
        session.sync_prefix(prompt_tokens)
        # generation runs synchronously inside the lock; this is fine for a
        # single-session server and matches how llama.cpp wants to be driven.
        result = session.generate(max_tokens=max_new, stop_strings=stops)

    parsed = parse_qwen_response(result.text)

    yield _sse(
        _chunk_payload(completion_id, created, model_name, {"role": "assistant"})
    )
    if parsed.content:
        yield _sse(
            _chunk_payload(
                completion_id, created, model_name, {"content": parsed.content}
            )
        )
    if parsed.tool_calls:
        tool_deltas = [
            {
                "index": i,
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for i, tc in enumerate(parsed.tool_calls)
        ]
        yield _sse(
            _chunk_payload(
                completion_id, created, model_name, {"tool_calls": tool_deltas}
            )
        )
    final_reason = "tool_calls" if parsed.tool_calls else parsed.finish_reason
    yield _sse(
        _chunk_payload(
            completion_id, created, model_name, {}, finish_reason=final_reason
        )
    )
    yield "data: [DONE]\n\n"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
