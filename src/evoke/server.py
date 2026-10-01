"""FastAPI server exposing EVOKE as an OpenAI-compatible /v1/chat/completions
endpoint. The persistent KV cache survives between requests; only the new tail
of the message history is decoded each turn (see Session.sync_prefix). This
lets any OpenAI-compatible agent harness (opencode, Aider, OpenHands) drive
EVOKE without code changes on their side.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path as FsPath
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from evoke.auth import (
    NAMESPACE_SEP,
    ApiKey,
    InvalidKeyId,
    KeyExists,
    KeyRing,
    StaticKey,
)
from evoke.config import EvokeConfig
from evoke.llama_engine import LlamaCppEngine
from evoke.scheduler import QueueTimeout, SessionBusy, TurnScheduler
from evoke.session import Session, SessionPool, SyncStats
from evoke.templates import ParsedResponse, format_qwen_chat, parse_qwen_response
from evoke.turn_metrics import TurnMetrics, measure_turn
from evoke.types import CacheStats, KeepSpan

DEFAULT_SESSION_ID = "default"
SESSION_HEADER_RE = re.compile(r"[A-Za-z0-9._:@/+=-]{1,256}")
# A user message no real prompt starts with, so the rendered system-only
# conversation diverges from the real prompt right after the system turn.
_PIN_SENTINEL = "\uffff"
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    501: "not_implemented_error",
    422: "invalid_request_error",
    503: "overloaded_error",
}


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers
        self.extra = extra


def _error_body(
    status: int,
    message: str,
    code: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": _ERROR_TYPES.get(status, "server_error"),
            "code": code,
            "param": None,
            **(extra or {}),
        }
    }


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
    # Per-request keep mark: "pin" or "prefer". Validated in the handler so an
    # unknown value gets an OpenAI-shaped 400 rather than a 422.
    evoke_keep: str | None = None

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
    stream_options: dict[str, Any] | None = None
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
    metrics: TurnMetrics | None = None,
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
        "usage": _usage(prompt_token_count, completion_token_count, metrics),
    }


def _usage(
    prompt_token_count: int,
    completion_token_count: int,
    metrics: TurnMetrics | None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": prompt_token_count + completion_token_count,
    }
    if metrics is not None:
        usage["evoke"] = metrics.as_usage()
    return usage


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


class LoadModelRequest(BaseModel):
    model: str


class IssueKeyRequest(BaseModel):
    key_id: str


def create_app(
    engine: LlamaCppEngine,
    model_name: str,
    config: EvokeConfig | None = None,
    *,
    max_sessions: int = 8,
    enable_thinking: bool | None = None,
    model_dir: str | FsPath | None = None,
    model_path: str | None = None,
    engine_factory: Callable[[str], LlamaCppEngine] | None = None,
    idle_timeout: float | None = None,
    keyring: KeyRing | None = None,
    pin_system_prompt: bool = False,
    queue_timeout: float | None = None,
    max_waiting_per_session: int = 1,
    max_sessions_per_key: int = 2,
    pin_cap: float = 0.6,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def _lifespan(_: FastAPI):
        watcher: asyncio.Task | None = None
        if idle_timeout:
            if engine_factory is None or model_path is None:
                print(
                    "[idle] idle_timeout set but model reload is not configured; "
                    "idle unload disabled",
                    flush=True,
                )
            else:
                watcher = asyncio.create_task(_idle_watcher())
        try:
            yield
        finally:
            if watcher is not None:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher

    app = FastAPI(title="EVOKE", version="0.1.0", lifespan=_lifespan)
    if enable_thinking is None:
        enable_thinking = {"1": True, "true": True, "0": False, "false": False}.get(
            os.environ.get("EVOKE_ENABLE_THINKING", "").lower()
        )
    # With keys configured, every session is namespaced by key id, so the
    # headerless prefix-affinity router would have to match across tenants;
    # it is disabled and the session header becomes mandatory.
    require_session = keyring is not None

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status,
            content=_error_body(exc.status, exc.message, exc.code, exc.extra),
            headers=exc.headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.status_code, str(exc.detail), None),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _engine_error(_: Request, exc: Exception):
        print(f"[500] {exc!r}", flush=True)
        return JSONResponse(
            status_code=500, content=_error_body(500, str(exc), "evoke_engine_error")
        )

    def _caller(authorization: str | None = Header(default=None)) -> ApiKey | None:
        if keyring is None:
            return None
        key = keyring.authenticate(authorization)
        if key is None:
            raise ApiError(401, "invalid_api_key", "missing or invalid API key")
        return key

    def _admin(caller: ApiKey | None = Depends(_caller)) -> ApiKey | None:
        if caller is not None and not caller.admin:
            raise ApiError(403, "admin_required", "this endpoint needs an admin key")
        return caller

    def _pool_id(caller: ApiKey | None, header: str) -> str:
        if caller is None:
            return header
        if not SESSION_HEADER_RE.fullmatch(header):
            raise ApiError(
                400,
                "evoke_session_invalid",
                "X-Evoke-Session must be 1-256 chars of [A-Za-z0-9._:@/+=-]",
            )
        return f"{caller.key_id}{NAMESPACE_SEP}{header}"

    def _visible_id(caller: ApiKey | None, pool_id: str) -> str | None:
        if caller is None or caller.admin:
            return pool_id
        prefix = f"{caller.key_id}{NAMESPACE_SEP}"
        return pool_id[len(prefix) :] if pool_id.startswith(prefix) else None

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
            content={
                **_error_body(422, str(exc.errors())[:1000], "invalid_request"),
                "detail": exc.errors(),
                "body_excerpt": body_excerpt[:500],
            },
        )

    def _owner_of(session_id: str) -> str | None:
        owner, sep, _ = session_id.partition(NAMESPACE_SEP)
        return owner if sep else None

    def _new_pool() -> SessionPool:
        return SessionPool(
            engine,
            config=config,
            max_sessions=max_sessions,
            max_sessions_per_owner=max_sessions_per_key if keyring else None,
            owner_of=_owner_of if keyring else None,
        )

    pool = _new_pool()
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
    scheduler = TurnScheduler(
        max_waiting_per_session=max_waiting_per_session, timeout=queue_timeout
    )
    pin_cache: dict[str, int] = {}

    # Idle unload state. n_ctx and supports_kv_block are cached at load time
    # because reading them off a closed engine dereferences a freed
    # llama_context; observability endpoints must keep answering while the
    # model is unloaded.
    engine_loaded = True
    active_n_ctx = engine.n_ctx
    active_kv_block = engine.supports_kv_block
    last_used = time.monotonic()
    inflight = 0

    def _close_engine_blocking() -> None:
        # engine_lock must be held around close: a streaming producer thread
        # can still be driving the engine after a client disconnect released
        # the asyncio lock, and freeing the context under it would crash.
        with engine_lock:
            engine.close()

    async def _unload_locked(reason: str) -> None:
        # Caller holds `lock`. Sessions and their archives die with the
        # engine, same contract as a /models/load hot swap; clients resend
        # full history, so a reloaded server rebuilds state by prefill.
        nonlocal pool, engine_loaded
        await asyncio.to_thread(_close_engine_blocking)
        pool = _new_pool()
        engine_loaded = False
        print(f"[models] unloaded {model_name} ({reason})", flush=True)

    async def _load_locked(path: str, name: str) -> None:
        # Caller holds `lock` and has already closed the previous engine.
        nonlocal engine, pool, model_name, model_path, engine_loaded
        nonlocal active_n_ctx, active_kv_block, last_used
        engine = await asyncio.to_thread(engine_factory, path)
        model_name = name
        model_path = path
        pool = _new_pool()
        active_n_ctx = engine.n_ctx
        active_kv_block = engine.supports_kv_block
        engine_loaded = True
        last_used = time.monotonic()

    async def _ensure_loaded() -> None:
        if engine_loaded:
            return
        async with lock:
            if engine_loaded:
                return
            try:
                await _load_locked(model_path, model_name)
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail=f"model reload failed: {exc}"
                ) from exc
            print(f"[models] reloaded {model_name} after unload", flush=True)

    async def _idle_watcher() -> None:
        interval = min(max(idle_timeout / 4.0, 0.05), 10.0)
        while True:
            await asyncio.sleep(interval)
            if not engine_loaded or inflight > 0:
                continue
            if time.monotonic() - last_used < idle_timeout:
                continue
            async with lock:
                # Re-check under the lock: a request may have raced in
                # between the unlocked check and acquisition.
                if (
                    engine_loaded
                    and inflight == 0
                    and time.monotonic() - last_used >= idle_timeout
                ):
                    await _unload_locked(f"idle {idle_timeout:g}s")

    def _budget() -> int:
        return (pool._config or Session._default_config(active_n_ctx)).max_active_tokens

    def _keep_marks_on_resident() -> bool:
        # Marks on already-decoded blocks need absolute positions to line up with
        # the prompt, which only sparse positions with identity recovery give;
        # otherwise a mark applies to content decoded on that request only.
        config = pool._config or Session._default_config(active_n_ctx)
        return config.position_mode == "sparse" and config.recovery_match == "identity"

    def _logical_window() -> int:
        config = pool._config or Session._default_config(active_n_ctx)
        return config.logical_window or active_n_ctx

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        # Lock-free on purpose: a stream holds the engine lock for its whole
        # duration, and a doctor probe must answer while a turn is running.
        # Carries no session ids, so it is safe without a key.
        return {
            "status": "ok",
            "model": model_name,
            "model_loaded": engine_loaded,
            "kv_cells": active_n_ctx,
            "logical_window": _logical_window(),
            "budget": _budget(),
            "pin_system_prompt": pin_system_prompt,
            "pin_cap_fraction": pin_cap,
            "keep_marks_on_resident": _keep_marks_on_resident(),
            "auth_required": keyring is not None,
            "session_required": require_session,
            "kv_block_primitives": active_kv_block,
            "queue": {
                "running": scheduler.running is not None,
                "waiting": scheduler.waiting,
            },
            "version": app.version,
        }

    @app.get("/v1/models", dependencies=[Depends(_caller)])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "evoke",
                    "context_length": active_n_ctx,
                    "evoke": {
                        "kv_cells": active_n_ctx,
                        "logical_window": _logical_window(),
                        "budget": _budget(),
                    },
                }
            ],
        }

    def _gguf_stems() -> list[str]:
        # mmproj-*.gguf files are vision projectors, not loadable models.
        if model_dir is None:
            return []
        return sorted(
            p.stem
            for p in FsPath(model_dir).glob("*.gguf")
            if "mmproj" not in p.stem.lower()
        )

    @app.get("/models", dependencies=[Depends(_caller)])
    async def native_models() -> dict[str, Any]:
        # llama-server manager shape: clients like calcifer read the model
        # list from data[].id and the live context window / vision support
        # from --ctx-size / --mmproj in the active entry's status.args.
        # Without this endpoint they fall back to a guessed context window
        # and can overrun n_ctx.
        stems = _gguf_stems()
        if model_name not in stems:
            stems.insert(0, model_name)
        return {
            "data": [
                {
                    "id": stem,
                    "object": "model",
                    "status": (
                        {"args": ["--ctx-size", str(active_n_ctx)]}
                        if stem == model_name
                        else {}
                    ),
                }
                for stem in stems
            ]
        }

    @app.post("/models/load", dependencies=[Depends(_admin)])
    async def load_model(req: LoadModelRequest) -> dict[str, Any]:
        nonlocal engine_loaded
        if model_dir is None or engine_factory is None:
            raise HTTPException(
                status_code=404, detail="model switching is not configured"
            )
        if req.model == model_name and engine_loaded:
            return {"status": "ok", "model": model_name, "loaded": False}
        candidate = FsPath(model_dir) / f"{req.model}.gguf"
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"unknown model: {req.model}")
        async with lock:
            # Close the active model before loading the replacement: two
            # resident models cannot fit in VRAM on a 16GB card. All
            # sessions and their archives die with the engine because saved
            # KV from one model is meaningless to another.
            prev_path, prev_name = model_path, model_name
            await asyncio.to_thread(_close_engine_blocking)
            engine_loaded = False
            try:
                await _load_locked(str(candidate), req.model)
            except Exception as exc:
                if prev_path:
                    try:
                        await _load_locked(prev_path, prev_name)
                    except Exception as recovery_exc:
                        print(
                            f"[models] recovery load of {prev_name} failed: "
                            f"{recovery_exc}",
                            flush=True,
                        )
                raise HTTPException(
                    status_code=500, detail=f"model load failed: {exc}"
                ) from exc
            print(f"[models] switched to {model_name} ({model_path})", flush=True)
        return {"status": "ok", "model": model_name, "loaded": True}

    @app.post("/models/unload", dependencies=[Depends(_admin)])
    async def unload_model() -> dict[str, Any]:
        if engine_factory is None or model_path is None:
            raise HTTPException(
                status_code=404, detail="model reload is not configured"
            )
        async with lock:
            if not engine_loaded:
                return {"status": "ok", "model": model_name, "unloaded": False}
            if inflight > 0:
                raise HTTPException(status_code=409, detail="requests in flight")
            await _unload_locked("explicit unload")
        return {"status": "ok", "model": model_name, "unloaded": True}

    @app.get("/health")
    async def health(
        x_evoke_session: str | None = Header(default=None),
        caller: ApiKey | None = Depends(_caller),
    ) -> dict[str, Any]:
        async with lock:
            # peek, never get(): a poll must not create a session or swap
            # engine state away from a generation in flight.
            if x_evoke_session:
                sid = _pool_id(caller, x_evoke_session)
            elif caller is None or caller.admin:
                sid = pool.active_session_id or DEFAULT_SESSION_ID
            else:
                raise ApiError(
                    400, "evoke_session_required", "X-Evoke-Session header is required"
                )
            session = pool.peek(sid)
            base = {
                "status": "ok",
                "model_loaded": engine_loaded,
                "n_ctx": active_n_ctx,
                "kv_block_primitives": active_kv_block,
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
    async def list_sessions(
        caller: ApiKey | None = Depends(_caller),
    ) -> dict[str, Any]:
        async with lock:
            visible = [_visible_id(caller, sid) for sid in pool.session_ids()]
            active = pool.active_session_id
            return {
                "active": _visible_id(caller, active) if active else None,
                "sessions": [sid for sid in visible if sid is not None],
                "n_sessions": pool.n_sessions,
                "max_sessions": max_sessions,
                "evicted_count": pool.evicted_count,
            }

    @app.delete("/v1/sessions/{session_id:path}")
    async def delete_session(
        session_id: str,
        caller: ApiKey | None = Depends(_caller),
    ) -> dict[str, Any]:
        target = (
            session_id
            if caller is None or caller.admin
            else _pool_id(caller, session_id)
        )
        async with lock:
            dropped = pool.drop(target)
        return {
            "status": "dropped" if dropped else "not_found",
            "session_id": session_id,
        }

    @app.post("/admin/reset")
    async def reset_session(
        x_evoke_session: str | None = Header(default=None),
        caller: ApiKey | None = Depends(_caller),
    ) -> dict[str, Any]:
        if caller is not None and not x_evoke_session:
            raise ApiError(
                400, "evoke_session_required", "X-Evoke-Session header is required"
            )
        async with lock:
            if x_evoke_session:
                sid = _pool_id(caller, x_evoke_session)
            else:
                sid = pool.active_session_id or DEFAULT_SESSION_ID
            if not engine_loaded:
                return {"status": "reset", "session_id": sid}
            session = pool.get(sid)
            session.reset()
        return {"status": "reset", "session_id": sid}

    def _issuing_keyring() -> KeyRing:
        if keyring is None or not keyring.issuance_enabled:
            raise ApiError(
                501,
                "evoke_key_issuance_disabled",
                "key issuance needs EVOKE_API_KEYS_FILE and EVOKE_ISSUED_KEYS_FILE",
            )
        return keyring

    @app.post("/admin/keys", status_code=201, dependencies=[Depends(_admin)])
    async def issue_key(req: IssueKeyRequest) -> dict[str, Any]:
        ring = _issuing_keyring()
        try:
            token = ring.issue(req.key_id)
        except InvalidKeyId as exc:
            raise ApiError(400, "evoke_key_id_invalid", str(exc)) from exc
        except KeyExists as exc:
            raise ApiError(
                409, "evoke_key_exists", f"key {req.key_id!r} exists"
            ) from exc
        created = next(k.created for k in ring.issued_keys() if k.key_id == req.key_id)
        return {"key_id": req.key_id, "token": token, "created": created}

    @app.get("/admin/keys", dependencies=[Depends(_admin)])
    async def list_keys() -> dict[str, Any]:
        ring = _issuing_keyring()
        return {
            "static": ring.static_key_ids(),
            "issued": [
                {"key_id": k.key_id, "created": k.created} for k in ring.issued_keys()
            ],
        }

    def _drop_owned_blocking(key_id: str) -> int:
        with engine_lock:
            owned = [sid for sid in pool.session_ids() if _owner_of(sid) == key_id]
            for sid in owned:
                pool.drop(sid)
            return len(owned)

    @app.delete("/admin/keys/{key_id}", dependencies=[Depends(_admin)])
    async def revoke_key(key_id: str) -> dict[str, Any]:
        ring = _issuing_keyring()
        try:
            revoked = ring.revoke(key_id)
        except StaticKey as exc:
            raise ApiError(
                409,
                "evoke_key_static",
                "static keys are revoked by editing the key file",
            ) from exc
        if not revoked:
            raise ApiError(404, "evoke_key_not_found", f"no issued key {key_id!r}")
        # Taking the engine lock waits out a turn already running for this key;
        # turns still queued re-check the key when they start and are refused.
        async with lock:
            dropped = await asyncio.to_thread(_drop_owned_blocking, key_id)
        return {"status": "revoked", "key_id": key_id, "sessions_dropped": dropped}

    def _still_authorized(caller: ApiKey | None) -> Callable[[], bool] | None:
        if caller is None or keyring is None:
            return None
        return lambda: keyring.is_active(caller.key_id)

    @app.post("/admin/set_budget", dependencies=[Depends(_admin)])
    async def set_budget(req: Request) -> dict[str, Any]:
        nonlocal config
        body = await req.json()
        tokens = int(body.get("tokens", _budget()))
        if tokens < 32:
            raise HTTPException(status_code=400, detail="budget must be >= 32")
        async with lock:
            # Without an explicit config every session derives its own from
            # n_ctx; materialise one so the new budget reaches new sessions
            # and survives a pool rebuild on model reload.
            if pool._config is None:
                pool._config = Session._default_config(active_n_ctx)
                config = pool._config
            pool._config.max_active_tokens = tokens
            for sid in pool.session_ids():
                s = pool.peek(sid)
                if s is not None:
                    s.reset()
        return {"status": "ok", "max_active_tokens": tokens}

    def _render(msgs: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        if tools or enable_thinking is not None:
            # Render via Python jinja2 against the GGUF's own chat
            # template, which understands tools and enable_thinking
            # (the C path cannot carry either, so an explicit thinking
            # flag must route here or it is silently dropped). Falls
            # back to our handwritten format_qwen_chat only if the
            # model has no embedded template or the render fails.
            try:
                return engine.apply_chat_template_with_tools(
                    msgs,
                    tools=tools,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except RuntimeError as exc:
                print(
                    f"[tmpl] gguf template render failed ({exc}); "
                    "falling back to format_qwen_chat",
                    flush=True,
                )
                return format_qwen_chat(msgs, tools=tools, add_generation_prompt=True)
        try:
            return engine.apply_chat_template(msgs, add_generation_prompt=True)
        except RuntimeError as exc:
            # Model has no embedded chat template; fall back to ours.
            print(
                f"[tmpl] C template path failed ({exc}); "
                "falling back to format_qwen_chat",
                flush=True,
            )
            return format_qwen_chat(msgs, tools=None, add_generation_prompt=True)

    def _pin_length(
        msgs: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        prompt_tokens: list[int],
    ) -> int:
        if not pin_system_prompt:
            return 0
        n_sys = 0
        while n_sys < len(msgs) and msgs[n_sys].get("role") == "system":
            n_sys += 1
        if n_sys == 0:
            return 0
        key = hashlib.sha256(
            json.dumps([msgs[:n_sys], tools], sort_keys=True).encode("utf-8")
        ).hexdigest()
        cached = pin_cache.get(key)
        if cached is None:
            probe = engine.tokenize(
                _render(
                    msgs[:n_sys] + [{"role": "user", "content": _PIN_SENTINEL}], tools
                )
            )
            cached = _common_prefix_len(probe, prompt_tokens)
            if len(pin_cache) >= 64:
                pin_cache.clear()
            pin_cache[key] = cached
        return cached

    def _keep_spans(
        marked: list[ChatMessage],
        msgs: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        prompt_tokens: list[int],
    ) -> list[KeepSpan]:
        # A message's token range is found the way the system prompt's is: the
        # messages before it plus a sentinel user turn render to the same tokens
        # as the prompt up to the boundary the sentinel starts at.
        boundaries: dict[int, int] = {}

        def boundary(count: int) -> int:
            if count == 0:
                return 0
            if count not in boundaries:
                probe = engine.tokenize(
                    _render(
                        msgs[:count] + [{"role": "user", "content": _PIN_SENTINEL}],
                        tools,
                    )
                )
                boundaries[count] = _common_prefix_len(probe, prompt_tokens)
            return boundaries[count]

        return [
            KeepSpan(boundary(i), boundary(i + 1), m.evoke_keep)
            for i, m in enumerate(marked)
            if m.evoke_keep
        ]

    def _check_pin_cap(keep_spans: list[KeepSpan], pin_n: int) -> None:
        # Only client pins count against the cap, and the cap is a share of the
        # budget left after the operator's pinned system prompt: a Loom system
        # prompt already fills most of the budget, so a share of the whole
        # budget would refuse every marked request.
        covered = sorted(
            (max(span.start, pin_n), span.end)
            for span in keep_spans
            if span.mode == "pin" and span.end > pin_n
        )
        pinned = 0
        end = 0
        for start, stop in covered:
            start = max(start, end)
            if stop > start:
                pinned += stop - start
                end = stop
        cap = int(pin_cap * max(0, _budget() - pin_n))
        if pinned > cap:
            raise ApiError(
                400,
                "evoke_pin_budget_exceeded",
                f"{pinned} pinned tokens exceed the pin cap of {cap} "
                f"({pin_cap:g} of the budget left after the system prompt)",
                extra={"pinned_tokens": pinned, "pin_cap": cap},
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        x_evoke_session: str | None = Header(default=None),
        caller: ApiKey | None = Depends(_caller),
    ):
        if not req.messages:
            raise ApiError(400, "messages_empty", "messages must not be empty")
        if x_evoke_session:
            session_id: str | None = _pool_id(caller, x_evoke_session)
        elif require_session:
            raise ApiError(
                400, "evoke_session_required", "X-Evoke-Session header is required"
            )
        else:
            session_id = None

        # In-flight accounting gates the idle watcher and the explicit
        # unload endpoint: the engine must not be closed between here and
        # the end of this request (or the end of its SSE stream).
        nonlocal inflight
        inflight += 1
        handed_off = False

        def _release() -> None:
            nonlocal inflight, last_used
            inflight -= 1
            last_used = time.monotonic()

        try:
            await _ensure_loaded()

            for m in req.messages:
                if m.evoke_keep not in (None, "pin", "prefer"):
                    raise ApiError(
                        400,
                        "evoke_keep_invalid",
                        f"evoke_keep must be 'pin' or 'prefer', got {m.evoke_keep!r}",
                    )
            msgs = [
                m.model_dump(exclude_none=True, exclude={"evoke_keep"})
                for m in req.messages
            ]
            prompt = _render(msgs, req.tools)
            prompt_tokens = engine.tokenize(prompt)
            prompt_n = len(prompt_tokens)
            if prompt_n >= _logical_window():
                raise ApiError(
                    400,
                    "context_length_exceeded",
                    f"prompt is {prompt_n} tokens but the context window is "
                    f"{_logical_window()}; it cannot be decoded",
                )
            pin_n = _pin_length(msgs, req.tools, prompt_tokens)
            keep_spans = _keep_spans(req.messages, msgs, req.tools, prompt_tokens)
            _check_pin_cap(keep_spans, pin_n)

            starts_in_think = _prompt_opens_think(prompt)

            stops = _normalize_stops(req.stop)
            if "<|im_end|>" not in stops:
                stops.append("<|im_end|>")

            max_new = req.max_tokens or 2048
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            created = int(time.time())
            if session_id is None:
                # Open mode only: route by longest shared token prefix so an
                # interleaved side-request (title generation) cannot land on
                # the agent session and reset away its recovery archive.
                async with lock:
                    session_id = pool.route_id(prompt_tokens)
            label = x_evoke_session or session_id
            if not scheduler.admissible(session_id):
                raise ApiError(
                    409,
                    "evoke_session_busy",
                    "a turn for this session is running and another is queued",
                )
            print(
                f"[req {completion_id}] msgs={len(msgs)} tools={len(req.tools or [])} "
                f"stream={req.stream} max_new={max_new} prompt_chars={len(prompt)} "
                f"prompt_tokens={prompt_n} pinned={pin_n} "
                f"keep={[(k.start, k.end, k.mode) for k in keep_spans]} "
                f"session={session_id} stops={stops}",
                flush=True,
            )

            if req.stream:
                stream = _stream_completion(
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
                    starts_in_think,
                    scheduler=scheduler,
                    owner=caller.key_id if caller else None,
                    authorized=_still_authorized(caller),
                    session_label=label,
                    pin_prefix=pin_n,
                    keep_spans=keep_spans,
                    include_usage=bool(
                        req.stream_options and req.stream_options.get("include_usage")
                    ),
                )

                async def _stream_with_release():
                    try:
                        async for chunk in stream:
                            yield chunk
                    finally:
                        # Close the inner generator now rather than at GC:
                        # its finally sets the abort event that stops the
                        # producer thread, and the engine must be quiet
                        # before inflight hits zero.
                        await stream.aclose()
                        _release()

                handed_off = True
                return StreamingResponse(
                    _stream_with_release(),
                    media_type="text/event-stream",
                    headers={
                        "X-EVOKE-Session": label,
                        "X-EVOKE-Request-Id": completion_id,
                    },
                )

            def _run_turn(waited: float):
                with engine_lock:
                    session = pool.get(session_id)
                    before = session.manager.get_stats()
                    sync = session.sync_prefix(
                        prompt_tokens,
                        priority=req.evoke_priority,
                        pinned=req.evoke_pinned,
                        task_boundary=req.evoke_task_boundary,
                        pin_prefix=pin_n,
                        reserve_tokens=max_new,
                        keep_spans=keep_spans,
                    )
                    result = session.generate(max_tokens=max_new, stop_strings=stops)
                    metrics = _measure(
                        session,
                        engine.n_ctx,
                        label,
                        before,
                        sync,
                        prompt_n,
                        len(result.output_tokens),
                        waited,
                    )
                    return result, session._config.suppress_thinking_strip, metrics

            try:
                async with scheduler.turn(
                    session_id, owner=caller.key_id if caller else None
                ) as waited:
                    async with lock:
                        authorized = _still_authorized(caller)
                        if authorized is not None and not authorized():
                            raise ApiError(
                                401, "invalid_api_key", "API key was revoked"
                            )
                        result, suppress, metrics = await asyncio.to_thread(
                            _run_turn, waited
                        )
            except SessionBusy as exc:
                raise ApiError(
                    409,
                    "evoke_session_busy",
                    "a turn for this session is running and another is queued",
                ) from exc
            except QueueTimeout as exc:
                raise ApiError(
                    503,
                    "evoke_queue_timeout",
                    "timed out waiting for the engine; another session is generating",
                    headers={"Retry-After": "5"},
                ) from exc

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
            return JSONResponse(
                _completion_payload(
                    completion_id,
                    created,
                    model_name,
                    parsed,
                    prompt_n,
                    len(result.output_tokens),
                    metrics,
                ),
                headers=metrics.as_headers(),
            )
        finally:
            if not handed_off:
                _release()

    return app


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _measure(
    session: Session,
    n_ctx: int,
    label: str,
    before: CacheStats,
    sync: SyncStats,
    prompt_n: int,
    completion_n: int,
    waited: float,
) -> TurnMetrics:
    manager = session.manager
    return measure_turn(
        session=label,
        before=before,
        after=manager.get_stats(),
        peak_resident_tokens=manager.turn_peak_active_tokens,
        pinned_tokens=manager.pinned_token_count,
        prompt_tokens=prompt_n,
        prompt_tokens_decoded=sync.new_tokens_decoded,
        completion_tokens=completion_n,
        kv_cells=n_ctx,
        logical_window=session.logical_window,
        queue_wait_s=waited,
    )


_MARKERS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_end|>")
_LOOKBACK = max(len(m) for m in _MARKERS)


def _safe_emit_end(full_text: str) -> int:
    # Hold back the last _LOOKBACK chars so a marker that is mid-formation
    # ("<thin") never ships as content. The held tail is released either when
    # the marker resolves or at end-of-generation.
    return max(0, len(full_text) - _LOOKBACK)


def _prompt_opens_think(prompt: str) -> bool:
    # Qwen3.x thinking templates append a bare "<think>\n" to the assistant
    # generation prompt, so generation begins INSIDE the reasoning block and
    # the model never emits its own opening <think>. The stream gate keys on
    # that opener, so without this signal it sees only the trailing </think>
    # and leaks the whole reasoning trace as content. Detect the open block by
    # the last think marker in the rendered prompt being an unclosed <think>.
    open_idx = prompt.rfind("<think>")
    if open_idx == -1:
        return False
    return prompt.rfind("</think>") < open_idx


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
    starts_in_think: bool = False,
    keepalive_interval: float = 5.0,
    *,
    scheduler: TurnScheduler | None = None,
    owner: str | None = None,
    authorized: Callable[[], bool] | None = None,
    session_label: str | None = None,
    pin_prefix: int = 0,
    keep_spans: list[KeepSpan] | None = None,
    include_usage: bool = False,
):
    yield _sse(
        _chunk_payload(completion_id, created, model_name, {"role": "assistant"})
    )
    last_emit = time.monotonic()

    in_think = starts_in_think
    tool_locked = False
    emit_end = 0
    # Set when a think block closes: drop the whitespace between </think> and
    # the first visible char (matches the non-stream parser's lstrip), even
    # when that whitespace and the answer arrive in separate chunks.
    lstrip_pending = False
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
    waited = 0.0
    metrics: TurnMetrics | None = None
    acquire: asyncio.Future[None] | None = None
    holds_turn = False

    try:
        if scheduler is not None:
            # Wait for this session's turn with keepalives on the wire: the
            # response has already started, and a silent queue wait behind
            # another session's generation would trip client read timeouts.
            wait_start = time.monotonic()
            acquire = asyncio.ensure_future(scheduler.acquire(session_id, owner=owner))
            while not acquire.done():
                await asyncio.wait({acquire}, timeout=keepalive_interval)
                if not acquire.done():
                    yield ": keepalive\n\n"
                    last_emit = time.monotonic()
            try:
                acquire.result()
            except SessionBusy:
                yield _sse_error(
                    409,
                    "evoke_session_busy",
                    "a turn for this session is running and another is queued",
                )
                yield "data: [DONE]\n\n"
                return
            except QueueTimeout:
                yield _sse_error(
                    503,
                    "evoke_queue_timeout",
                    "timed out waiting for the engine; another session is generating",
                )
                yield "data: [DONE]\n\n"
                return
            holds_turn = True
            waited = time.monotonic() - wait_start
        async with lock:
            if authorized is not None and not authorized():
                yield _sse_error(401, "invalid_api_key", "API key was revoked")
                yield "data: [DONE]\n\n"
                return
            # Resolve the session inside the lock so any other request that
            # raced us through the pool gets to swap us in cleanly.
            session = pool.get(session_id)
            # Suppress mode streams the trace verbatim, so the prompt-opened
            # think block must not put the gate into suppression: that would
            # also gate the end-of-stream tail flush and drop the answer's last
            # _LOOKBACK chars.
            if session._config.suppress_thinking_strip:
                in_think = False

            def _produce():
                ctx = engine_lock if engine_lock is not None else threading.Lock()
                try:
                    with ctx:
                        before = session.manager.get_stats()
                        sync = session.sync_prefix(
                            prompt_tokens,
                            priority=evoke_priority,
                            pinned=evoke_pinned,
                            task_boundary=evoke_task_boundary,
                            pin_prefix=pin_prefix,
                            reserve_tokens=max_new,
                            keep_spans=keep_spans or (),
                        )
                        n_out = 0
                        for chunk in session.stream_generate(
                            max_tokens=max_new,
                            stop_strings=stops,
                            abort_event=abort,
                        ):
                            n_out = len(chunk.output_tokens)
                            loop.call_soon_threadsafe(chunk_q.put_nowait, chunk)
                        turn = _measure(
                            session,
                            engine.n_ctx,
                            session_label or session_id,
                            before,
                            sync,
                            len(prompt_tokens),
                            n_out,
                            waited,
                        )
                        loop.call_soon_threadsafe(chunk_q.put_nowait, turn)
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
                    last_emit = time.monotonic()
                    continue
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    print(
                        f"[stream {completion_id}] engine error: {item!r}", flush=True
                    )
                    yield _sse_error(500, "evoke_engine_error", str(item))
                    yield "data: [DONE]\n\n"
                    return
                if isinstance(item, TurnMetrics):
                    metrics = item
                    continue
                chunk = item
                full_text = chunk.full_text
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

                # A suppressed think block consumes chunks without emitting
                # anything, and the queue-quiet keepalive above never fires
                # because the engine keeps producing. Clients enforcing a
                # read timeout (the HF demo app) would abort the stream, so
                # starved iterations must emit their own keepalive.
                if time.monotonic() - last_emit >= keepalive_interval:
                    yield ": keepalive\n\n"
                    last_emit = time.monotonic()

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
                        lstrip_pending = True

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
                        last_emit = time.monotonic()
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
                        last_emit = time.monotonic()
                    in_think = True
                    emit_end = think_idx
                    continue

                safe_end = _safe_emit_end(full_text)
                if lstrip_pending:
                    while emit_end < safe_end and full_text[emit_end] in " \t\r\n":
                        emit_end += 1
                    if emit_end < safe_end:
                        lstrip_pending = False
                if safe_end > emit_end:
                    delta = full_text[emit_end:safe_end]
                    emit_end = safe_end
                    yield _sse(
                        _chunk_payload(
                            completion_id, created, model_name, {"content": delta}
                        )
                    )
                    last_emit = time.monotonic()
    finally:
        abort.set()
        if acquire is not None:
            if holds_turn:
                scheduler.release()
            elif not acquire.done():
                acquire.cancel()
            elif not acquire.cancelled() and acquire.exception() is None:
                scheduler.release()

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
        if lstrip_pending:
            tail = tail.lstrip()
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
    if include_usage:
        prompt_n = len(prompt_tokens)
        completion_n = metrics.logical_tokens - prompt_n if metrics else 0
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [],
                "usage": _usage(prompt_n, completion_n, metrics),
            }
        )
    yield "data: [DONE]\n\n"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse_error(status: int, code: str, message: str) -> str:
    return _sse(_error_body(status, message, code))
