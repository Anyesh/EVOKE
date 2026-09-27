"""A queued stream waits for its turn with keepalives on the wire, and a queue
timeout is reported in-stream because the 200 has already been sent."""

from __future__ import annotations

import asyncio
import json

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.scheduler import TurnScheduler
from evoke.server import _stream_completion
from evoke.session import SessionPool


def _stream(pool, engine, scheduler, session_id="s1", include_usage=False):
    return _stream_completion(
        pool,
        session_id,
        engine,
        asyncio.Lock(),
        engine.tokenize("prompt"),
        ["<|im_end|>"],
        64,
        "cid",
        0,
        "model",
        keepalive_interval=0.01,
        scheduler=scheduler,
        session_label="acme/" + session_id,
        include_usage=include_usage,
    )


def _setup():
    engine = MockEngine(n_ctx=4096)
    cfg = EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=16,
        sink_count=0,
        recovery_mode="discard",
    )
    engine.queue_tokens([ord(c) for c in "hello"] + [engine.eos_token])
    return engine, SessionPool(engine, config=cfg)


def test_queued_stream_sends_keepalives_then_runs():
    engine, pool = _setup()

    async def main():
        sched = TurnScheduler()
        await sched.acquire("other")

        async def release_later():
            await asyncio.sleep(0.05)
            sched.release()

        releaser = asyncio.create_task(release_later())
        raw = [line async for line in _stream(pool, engine, sched, include_usage=True)]
        await releaser
        return raw, sched.running

    raw, running = asyncio.run(main())
    assert raw[1] == ": keepalive\n\n"
    assert raw[-1] == "data: [DONE]\n\n"
    usage = json.loads(raw[-2][len("data: ") :])["usage"]
    assert usage["evoke"]["session"] == "acme/s1"
    assert usage["evoke"]["queue_wait_ms"] >= 40
    assert running is None


def test_queue_timeout_is_reported_in_stream():
    engine, pool = _setup()

    async def main():
        sched = TurnScheduler(timeout=0.03)
        await sched.acquire("other")
        raw = [line async for line in _stream(pool, engine, sched)]
        return raw, sched.waiting

    raw, waiting = asyncio.run(main())
    error = json.loads(raw[-2][len("data: ") :])["error"]
    assert error["code"] == "evoke_queue_timeout"
    assert raw[-1] == "data: [DONE]\n\n"
    assert waiting == 0


def test_abandoned_queued_stream_leaves_no_waiter():
    engine, pool = _setup()

    async def main():
        sched = TurnScheduler()
        await sched.acquire("other")
        gen = _stream(pool, engine, sched)
        await gen.__anext__()
        await gen.__anext__()
        await gen.aclose()
        await asyncio.sleep(0)
        return sched.waiting, sched.running

    assert asyncio.run(main()) == (0, "other")


def test_turn_from_a_revoked_key_is_refused_when_it_starts():
    engine, pool = _setup()

    async def main():
        gen = _stream_completion(
            pool,
            "s1",
            engine,
            asyncio.Lock(),
            engine.tokenize("prompt"),
            ["<|im_end|>"],
            64,
            "cid",
            0,
            "model",
            scheduler=TurnScheduler(),
            authorized=lambda: False,
        )
        return [line async for line in gen]

    raw = asyncio.run(main())
    assert json.loads(raw[-2][len("data: ") :])["error"]["code"] == "invalid_api_key"
    assert pool.session_ids() == []
