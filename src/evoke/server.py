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

from evoke.config import EvokeConfig
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
    # EVOKE extensions for harness-aware scoring. These are not part of the
    # OpenAI spec; a harness like opencode or Claude Code can set them to
    # signal which turns are central to the current task and which are
    # ephemeral. Default 1.0 / False = no harness signal, scorer falls back
    # to attention + recency only.
    evoke_priority: float = Field(default=1.0)
    evoke_pinned: bool = Field(default=False)
    # When true, the scorer treats this request as the start of a new task:
    # the task-focus embedding snaps to the new user message, and blocks
    # coherent with the prior task lose their coherence score. Use this when
    # the harness explicitly transitions between unrelated tasks in the same
    # session (e.g. "investigate auth bug" -> "implement feature X").
    evoke_task_boundary: bool = Field(default=False)


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


def create_app(
    engine: LlamaCppEngine,
    model_name: str,
    config: EvokeConfig | None = None,
) -> FastAPI:
    app = FastAPI(title="EVOKE", version="0.1.0")
    session = Session(engine, config=config)
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
        stats = session.manager.get_stats()
        return {
            "status": "ok",
            "cached_tokens": session.cached_token_count,
            "n_ctx": engine.n_ctx,
            "active_tokens": stats.active_tokens,
            "active_blocks": stats.active_blocks,
            "budget": stats.budget,
            "budget_utilization": round(stats.budget_utilization, 3),
            "total_evictions": stats.total_evictions,
            "total_recoveries": stats.total_recoveries,
            "kv_block_primitives": engine.supports_kv_block,
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
        if req.tools:
            # Render via Python jinja2 against the GGUF's own chat template,
            # which understands tools. Falls back to our handwritten
            # format_qwen_chat only if the model has no embedded template or
            # the render fails outright.
            try:
                prompt = engine.apply_chat_template_with_tools(
                    msgs, tools=req.tools, add_generation_prompt=True
                )
            except RuntimeError:
                prompt = format_qwen_chat(
                    msgs, tools=req.tools, add_generation_prompt=True
                )
        else:
            try:
                prompt = engine.apply_chat_template(msgs, add_generation_prompt=True)
            except RuntimeError:
                # Model has no embedded chat template; fall back to ours.
                prompt = format_qwen_chat(msgs, tools=None, add_generation_prompt=True)
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
                    req.evoke_priority,
                    req.evoke_pinned,
                    req.evoke_task_boundary,
                ),
                media_type="text/event-stream",
            )

        async with lock:
            session.sync_prefix(
                prompt_tokens,
                priority=req.evoke_priority,
                pinned=req.evoke_pinned,
                task_boundary=req.evoke_task_boundary,
            )
            result = session.generate(max_tokens=max_new, stop_strings=stops)

        parsed = parse_qwen_response(
            result.text,
            strip_thinking=not session._config.suppress_thinking_strip,
        )
        return _completion_payload(
            completion_id,
            created,
            model_name,
            parsed,
            prompt_n,
            len(result.output_tokens),
        )

    return app


_MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_end|>")
_LOOKBACK = max(len(m) for m in _MARKERS)


def _safe_emit_end(full_text: str) -> int:
    # Hold back the last _LOOKBACK chars so a marker that is mid-formation
    # ("<thin") never ships as content. The held tail is released either when
    # the marker resolves or at end-of-generation.
    return max(0, len(full_text) - _LOOKBACK)


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
    evoke_priority: float = 1.0,
    evoke_pinned: bool = False,
    evoke_task_boundary: bool = False,
):
    yield _sse(
        _chunk_payload(completion_id, created, model_name, {"role": "assistant"})
    )

    in_think = False
    tool_locked = False
    emit_end = 0
    full_text = ""
    finish_reason: str | None = None

    async with lock:
        session.sync_prefix(
            prompt_tokens,
            priority=evoke_priority,
            pinned=evoke_pinned,
            task_boundary=evoke_task_boundary,
        )
        for chunk in session.stream_generate(max_tokens=max_new, stop_strings=stops):
            full_text = chunk.full_text
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

            if tool_locked:
                continue

            # On hybrid memory models with suppress_thinking_strip set, the
            # client must echo the full assistant content (including the
            # thinking trace) back to keep the cached state aligned. Skip
            # the in_think gating so <think>...</think> streams through
            # verbatim alongside the answer.
            if not session._config.suppress_thinking_strip:
                if in_think:
                    close_idx = full_text.find("</think>", emit_end)
                    if close_idx == -1:
                        continue
                    in_think = False
                    emit_end = close_idx + len("</think>")

                think_idx = full_text.find("<think>", emit_end)
            else:
                think_idx = -1

            tc_idx = full_text.find("<tool_call>", emit_end)

            if tc_idx != -1 and (think_idx == -1 or tc_idx < think_idx):
                pre = full_text[emit_end:tc_idx]
                if pre.strip():
                    yield _sse(
                        _chunk_payload(
                            completion_id, created, model_name, {"content": pre}
                        )
                    )
                tool_locked = True
                emit_end = tc_idx
                continue

            if think_idx != -1:
                pre = full_text[emit_end:think_idx]
                if pre.strip():
                    yield _sse(
                        _chunk_payload(
                            completion_id, created, model_name, {"content": pre}
                        )
                    )
                in_think = True
                emit_end = think_idx
                continue

            safe_end = _safe_emit_end(full_text)
            if safe_end > emit_end:
                delta = full_text[emit_end:safe_end]
                emit_end = safe_end
                yield _sse(
                    _chunk_payload(
                        completion_id, created, model_name, {"content": delta}
                    )
                )

    parsed = parse_qwen_response(
        full_text, strip_thinking=not session._config.suppress_thinking_strip
    )
    if not tool_locked and not in_think:
        tail = full_text[emit_end:]
        for tok in ("<|im_end|>", "<|endoftext|>"):
            if tok in tail:
                tail = tail.split(tok, 1)[0]
                break
        if tail:
            yield _sse(
                _chunk_payload(completion_id, created, model_name, {"content": tail})
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

    final_reason = (
        "tool_calls" if parsed.tool_calls else (finish_reason or parsed.finish_reason)
    )
    yield _sse(
        _chunk_payload(
            completion_id, created, model_name, {}, finish_reason=final_reason
        )
    )
    yield "data: [DONE]\n\n"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
