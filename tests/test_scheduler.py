import asyncio

import pytest

from evoke.scheduler import QueueTimeout, SessionBusy, TurnScheduler


def _run(coro):
    return asyncio.run(coro)


def test_single_turn_runs_immediately():
    async def main():
        sched = TurnScheduler()
        async with sched.turn("a") as waited:
            assert waited < 0.05
            assert sched.running == "a"
        assert sched.running is None

    _run(main())


def test_round_robin_across_sessions():
    # Session a queues two turns before b queues one; b must run between them
    # so one caller cannot monopolise the engine by queueing.
    async def main():
        sched = TurnScheduler(max_waiting_per_session=2)
        order: list[str] = []
        gate = asyncio.Event()

        async def holder():
            async with sched.turn("x"):
                await gate.wait()

        async def worker(sid: str, tag: str):
            async with sched.turn(sid):
                order.append(tag)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        tasks = [asyncio.create_task(worker("a", "a1"))]
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(worker("a", "a2")))
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(worker("b", "b1")))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(h, *tasks)
        return order

    assert _run(main()) == ["a1", "b1", "a2"]


def test_same_session_overflow_is_rejected():
    async def main():
        sched = TurnScheduler(max_waiting_per_session=1)
        gate = asyncio.Event()

        async def holder():
            async with sched.turn("a"):
                await gate.wait()

        async def waiter():
            async with sched.turn("a"):
                pass

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        w = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        with pytest.raises(SessionBusy):
            async with sched.turn("a"):
                pass
        gate.set()
        await asyncio.gather(h, w)

    _run(main())


def test_queue_timeout_frees_slot_for_later_waiters():
    async def main():
        sched = TurnScheduler(timeout=0.05)
        gate = asyncio.Event()

        async def holder():
            async with sched.turn("a"):
                await gate.wait()

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        with pytest.raises(QueueTimeout):
            async with sched.turn("b"):
                pass
        assert sched.waiting == 0
        gate.set()
        await h
        async with sched.turn("b"):
            assert sched.running == "b"

    _run(main())


def test_cancelled_waiter_does_not_strand_the_queue():
    async def main():
        sched = TurnScheduler()
        gate = asyncio.Event()
        ran: list[str] = []

        async def holder():
            async with sched.turn("a"):
                await gate.wait()

        async def worker(sid: str):
            async with sched.turn(sid):
                ran.append(sid)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        doomed = asyncio.create_task(worker("b"))
        survivor = asyncio.create_task(worker("c"))
        await asyncio.sleep(0)
        doomed.cancel()
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(h, survivor)
        with pytest.raises(asyncio.CancelledError):
            await doomed
        return ran, sched.waiting, sched.running

    assert _run(main()) == (["c"], 0, None)


def test_counters_report_running_and_waiting():
    async def main():
        sched = TurnScheduler()
        gate = asyncio.Event()

        async def holder():
            async with sched.turn("a"):
                await gate.wait()

        async def worker(sid: str):
            async with sched.turn(sid):
                pass

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        w = asyncio.create_task(worker("b"))
        await asyncio.sleep(0)
        assert sched.waiting == 1
        assert sched.running == "a"
        gate.set()
        await asyncio.gather(h, w)

    _run(main())


def test_round_robin_is_across_owners_not_sessions():
    # Owner x opens three sessions, owner y one; y must not wait behind all of
    # x's sessions, so opening more sessions buys no extra share of the engine.
    async def main():
        sched = TurnScheduler()
        order: list[str] = []
        gate = asyncio.Event()

        async def holder():
            async with sched.turn("h", owner="z"):
                await gate.wait()

        async def worker(sid: str, owner: str):
            async with sched.turn(sid, owner=owner):
                order.append(sid)

        h = asyncio.create_task(holder())
        await asyncio.sleep(0)
        tasks = []
        for sid, owner in (("x1", "x"), ("x2", "x"), ("x3", "x"), ("y1", "y")):
            tasks.append(asyncio.create_task(worker(sid, owner)))
            await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(h, *tasks)
        return order

    assert _run(main()) == ["x1", "y1", "x2", "x3"]
