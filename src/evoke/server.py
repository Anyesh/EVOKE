"""FastAPI server exposing EVOKE as an OpenAI-compatible /v1/chat/completions
endpoint. The persistent KV cache survives between requests; only the new tail
of the message history is decoded each turn (see Session.sync_prefix). This
lets any OpenAI-compatible agent harness (opencode, Aider, OpenHands) drive
EVOKE without code changes on their side.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from fastapi import Header

from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.session import Session, SessionPool
from evoke.templates import ParsedResponse, format_qwen_chat, parse_qwen_response

DEFAULT_SESSION_ID = "default"


class ChatMessage(BaseModel):
    role: str
    # Modern OpenAI-compatible clients (opencode, the openai Python SDK on
    # multimodal/tool turns, Aider in some modes) send content as either a
    # plain string or as a list of content parts: [{"type": "text", "text":
    # "..."}], possibly with image_url parts. The flatten_content validator
    # below collapses the list-of-text-parts form into a single string so
    # downstream code (templating, tokenizing) works uniformly.
    content: str | list[Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # Optional reasoning trace some clients (opencode, OpenHands) include on
    # assistant messages. We accept and ignore it; the cached state holds the
    # full assistant emit including the <think> trace already, when
    # suppress_thinking_strip is set on the session.
    reasoning: str | None = None
    reasoning_content: str | None = None

    @model_validator(mode="after")
    def flatten_content(self) -> "ChatMessage":
        # Collapse a list-of-content-parts into a single string. Multimodal
        # image parts are stubbed as "[image]" since the backend is text-only.
        if isinstance(self.content, list):
            parts: list[str] = []
            for part in self.content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    elif part.get("type") in {"image_url", "image"}:
                        parts.append("[image]")
                    elif "text" in part and isinstance(part["text"], str):
                        parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            self.content = "".join(parts) if parts else ""
        return self


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
    *,
    max_sessions: int = 8,
    enable_thinking: bool | None = None,
) -> FastAPI:
    app = FastAPI(title="EVOKE", version="0.1.0")
    if enable_thinking is None:
        enable_thinking = {"1": True, "true": True, "0": False, "false": False}.get(
            os.environ.get("EVOKE_ENABLE_THINKING", "").lower()
        )

    @app.exception_handler(RequestValidationError)
    async def _log_422(request: Request, exc: RequestValidationError):
        # Log the validation errors AND a truncated view of the body so we
        # can see exactly what an OpenAI-compatible client (opencode, etc.)
        # is sending that the schema rejected. Useful during integration
        # work; the body excerpt is bounded to keep logs sane.
        try:
            body_bytes = await request.body()
            body_excerpt = body_bytes[:2000].decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body_excerpt = "<unreadable>"
        print(
            f"[422] {request.method} {request.url.path} from {request.client.host if request.client else '?'}\n"
            f"  errors: {exc.errors()}\n"
            f"  body[:2000]: {body_excerpt}",
            flush=True,
        )
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "body_excerpt": body_excerpt[:500]},
        )

    pool = SessionPool(engine, config=config, max_sessions=max_sessions)
    # Single global lock: SessionPool swaps engine state on every
    # cross-session transition; concurrent requests against the same
    # engine context would race. Per-session concurrency would require
    # n_seq_max > 1 routing in every primitive (paper §9, future work).
    lock = asyncio.Lock()
    # Engine work runs in worker threads (so the event loop stays free to
    # serve /health and SSE keepalives during a long prefill); this second
    # lock serializes the threads themselves, because the asyncio lock is
    # released if a streaming client disconnects while the producer thread
    # is still driving the engine.
    engine_lock = threading.Lock()

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
    async def health(
        x_evoke_session: str | None = Header(default=None),
    ) -> dict[str, Any]:
        async with lock:
            # peek, never get(): a poll must not create a session or swap
            # engine state away from a generation in flight.
            sid = x_evoke_session or pool.active_session_id or DEFAULT_SESSION_ID
            session = pool.peek(sid)
            base = {
                "status": "ok",
                "n_ctx": engine.n_ctx,
                "kv_block_primitives": engine.supports_kv_block,
                "session_id": sid,
                "n_sessions": pool.n_sessions,
                "sessions_evicted": pool.evicted_count,
            }
            if session is None:
                return {
                    **base,
                    "cached_tokens": 0,
                    "active_tokens": 0,
                    "active_blocks": 0,
                    "budget": 0,
                    "budget_utilization": 0.0,
                    "total_evictions": 0,
                    "total_recoveries": 0,
                    "peak_active": 0,
                    "total_prompt_tokens": 0,
                    "total_new_decoded": 0,
                    "identity_recovered": 0,
                    "identity_mismatch": 0,
                }
            stats = session.manager.get_stats()
            return {
                **base,
                "cached_tokens": session.cached_token_count,
                "active_tokens": stats.active_tokens,
                "active_blocks": stats.active_blocks,
                "budget": stats.budget,
                "budget_utilization": round(stats.budget_utilization, 3),
                "total_evictions": stats.total_evictions,
                "total_recoveries": stats.total_recoveries,
                "peak_active": session.manager.peak_active_tokens,
                "total_prompt_tokens": session.total_prompt_tokens,
                "total_new_decoded": session.total_new_decoded,
                "identity_recovered": session.gapfill_recovered,
                "identity_mismatch": session.gapfill_mismatch,
            }

    @app.get("/v1/sessions")
    async def list_sessions() -> dict[str, Any]:
        async with lock:
            return {
                "active": pool.active_session_id,
                "sessions": pool.session_ids(),
                "n_sessions": pool.n_sessions,
                "max_sessions": max_sessions,
                "evicted_count": pool.evicted_count,
            }

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        async with lock:
            dropped = pool.drop(session_id)
        return {
            "status": "dropped" if dropped else "not_found",
            "session_id": session_id,
        }

    @app.post("/admin/reset")
    async def reset_session(
        x_evoke_session: str | None = Header(default=None),
    ) -> dict[str, Any]:
        async with lock:
            sid = x_evoke_session or pool.active_session_id or DEFAULT_SESSION_ID
            session = pool.get(sid)
            session.reset()
        return {"status": "reset", "session_id": sid}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        x_evoke_session: str | None = Header(default=None),
    ):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        # Resolve session under the pool lock so a concurrent request to
        # another session_id doesn't swap the engine state out from under
        # us mid-request. The lock is held for the full prompt-tokenize +
        # decode + generate cycle.
        msgs = [m.model_dump(exclude_none=True) for m in req.messages]
        if req.tools:
            # Render via Python jinja2 against the GGUF's own chat template,
            # which understands tools. Falls back to our handwritten
            # format_qwen_chat only if the model has no embedded template or
            # the render fails outright.
            try:
                prompt = engine.apply_chat_template_with_tools(
                    msgs,
                    tools=req.tools,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except RuntimeError as exc:
                print(
                    f"[tmpl] gguf template render failed ({exc}); "
                    "falling back to format_qwen_chat",
                    flush=True,
                )
                prompt = format_qwen_chat(
                    msgs, tools=req.tools, add_generation_prompt=True
                )
        else:
            try:
                prompt = engine.apply_chat_template(msgs, add_generation_prompt=True)
            except RuntimeError as exc:
                # Model has no embedded chat template; fall back to ours.
                print(
                    f"[tmpl] C template path failed ({exc}); "
                    "falling back to format_qwen_chat",
                    flush=True,
                )
                prompt = format_qwen_chat(msgs, tools=None, add_generation_prompt=True)
        prompt_tokens = engine.tokenize(prompt)
        prompt_n = len(prompt_tokens)
        if prompt_n >= engine.n_ctx:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"prompt is {prompt_n} tokens but n_ctx is {engine.n_ctx}; "
                    "it cannot be decoded"
                ),
            )

        stops = _normalize_stops(req.stop)
        if "<|im_end|>" not in stops:
            stops.append("<|im_end|>")

        max_new = req.max_tokens or 2048
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        # Explicit header wins; otherwise route by longest shared token
        # prefix so an interleaved side-request (title generation) cannot
        # land on the agent session and reset away its recovery archive.
        if x_evoke_session:
            session_id = x_evoke_session
        else:
            async with lock:
                session_id = pool.route_id(prompt_tokens)
        print(
            f"[req {completion_id}] msgs={len(msgs)} tools={len(req.tools or [])} "
            f"stream={req.stream} max_new={max_new} prompt_chars={len(prompt)} "
            f"prompt_tokens={prompt_n} session={session_id} stops={stops}",
            flush=True,
        )

        if req.stream:
            return StreamingResponse(
                _stream_completion(
                    pool,
                    session_id,
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
                    req.tools,
                    engine_lock,
                ),
                media_type="text/event-stream",
            )

        def _run_turn():
            with engine_lock:
                session = pool.get(session_id)
                session.sync_prefix(
                    prompt_tokens,
                    priority=req.evoke_priority,
                    pinned=req.evoke_pinned,
                    task_boundary=req.evoke_task_boundary,
                )
                result = session.generate(max_tokens=max_new, stop_strings=stops)
                return result, session._config.suppress_thinking_strip

        async with lock:
            result, suppress = await asyncio.to_thread(_run_turn)

        parsed = parse_qwen_response(
            result.text,
            strip_thinking=not suppress,
            tools=req.tools,
        )
        if not parsed.tool_calls and "<tool_call>" in result.text:
            print(
                f"[req {completion_id}] tool_call parse failed: "
                f"head={result.text[:160]!r} tail={result.text[-160:]!r}",
                flush=True,
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
    pool: SessionPool,
    session_id: str,
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
    tools: list[dict[str, Any]] | None = None,
    engine_lock: threading.Lock | None = None,
):
    yield _sse(
        _chunk_payload(completion_id, created, model_name, {"role": "assistant"})
    )

    in_think = False
    tool_locked = False
    emit_end = 0
    full_text = ""
    finish_reason: str | None = None

    # The engine work (prefix sync, prefill, decode) is synchronous and can
    # run for minutes on a long prompt. Running it inline would freeze the
    # event loop (no /health, no SSE bytes) until the first token, and
    # agent clients abort streams that stay silent that long. A producer
    # thread drives the engine and feeds chunks through a queue; the
    # generator emits SSE keepalive comments while the queue is quiet.
    loop = asyncio.get_running_loop()
    chunk_q: asyncio.Queue[Any] = asyncio.Queue()
    _DONE = object()
    # Set when the client disconnects (or the stream finishes) so the
    # producer thread stops generating instead of grinding out the rest of
    # a response nobody will read while retries queue up behind the lock.
    abort = threading.Event()

    try:
        async with lock:
            # Resolve the session inside the lock so any other request that
            # raced us through the pool gets to swap us in cleanly.
            session = pool.get(session_id)

            def _produce():
                ctx = engine_lock if engine_lock is not None else threading.Lock()
                try:
                    with ctx:
                        session.sync_prefix(
                            prompt_tokens,
                            priority=evoke_priority,
                            pinned=evoke_pinned,
                            task_boundary=evoke_task_boundary,
                        )
                        for chunk in session.stream_generate(
                            max_tokens=max_new,
                            stop_strings=stops,
                            abort_event=abort,
                        ):
                            loop.call_soon_threadsafe(chunk_q.put_nowait, chunk)
                    loop.call_soon_threadsafe(chunk_q.put_nowait, _DONE)
                except BaseException as exc:  # noqa: BLE001
                    loop.call_soon_threadsafe(chunk_q.put_nowait, exc)

            producer = threading.Thread(target=_produce, daemon=True)
            producer.start()

            while True:
                try:
                    item = await asyncio.wait_for(chunk_q.get(), timeout=10.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                chunk = item
                full_text = chunk.full_text
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

                if tool_locked:
                    continue

                # On hybrid memory models with suppress_thinking_strip set,
                # the client must echo the full assistant content (including
                # the thinking trace) back to keep the cached state aligned.
                # Skip the in_think gating so <think>...</think> streams
                # through verbatim alongside the answer.
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
    finally:
        abort.set()

    parsed = parse_qwen_response(
        full_text,
        strip_thinking=not session._config.suppress_thinking_strip,
        tools=tools,
    )
    if not parsed.tool_calls and "<tool_call>" in full_text:
        print(
            f"[stream {completion_id}] tool_call parse failed: "
            f"head={full_text[:160]!r} tail={full_text[-160:]!r}",
            flush=True,
        )
    # A locked stream whose block failed to parse must still ship the raw
    # text; otherwise the client receives an empty message and agents bail.
    if not in_think and (not tool_locked or not parsed.tool_calls):
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
    print(
        f"[stream {completion_id}] full_chars={len(full_text)} "
        f"finish={final_reason} tool_calls={len(parsed.tool_calls)} "
        f"emitted_to={emit_end}",
        flush=True,
    )
    yield _sse(
        _chunk_payload(
            completion_id, created, model_name, {}, finish_reason=final_reason
        )
    )
    yield "data: [DONE]\n\n"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
