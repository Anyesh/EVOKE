"""Admission control for chat turns on a single engine context.

One llama_context decodes one session at a time, so turns cannot run
concurrently. The scheduler makes the wait fair instead: each session has a
FIFO of waiting turns, and when the engine frees up the next turn comes from
the session after the one just served (round-robin by session), so a caller
that queues many turns cannot starve another caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict, deque
from typing import AsyncIterator


class SessionBusy(Exception):
    pass


class QueueTimeout(Exception):
    pass


class TurnScheduler:
    def __init__(
        self,
        *,
        max_waiting_per_session: int = 1,
        timeout: float | None = None,
    ) -> None:
        self._max_waiting = max_waiting_per_session
        self._timeout = timeout
        self._running: str | None = None
        # Sessions with waiters, in service order; the head is served next.
        self._queues: OrderedDict[str, deque[asyncio.Future[None]]] = OrderedDict()

    @property
    def running(self) -> str | None:
        return self._running

    @property
    def waiting(self) -> int:
        return sum(len(q) for q in self._queues.values())

    @contextlib.asynccontextmanager
    async def turn(self, session_id: str) -> AsyncIterator[float]:
        start = time.monotonic()
        await self.acquire(session_id)
        try:
            yield time.monotonic() - start
        finally:
            self.release()

    def admissible(self, session_id: str) -> bool:
        if self._running is None and not self._queues:
            return True
        queue = self._queues.get(session_id)
        if queue is not None and len(queue) >= self._max_waiting:
            return False
        return not (self._running == session_id and self._max_waiting == 0)

    async def acquire(self, session_id: str) -> None:
        if self._running is None and not self._queues:
            self._running = session_id
            return
        if not self.admissible(session_id):
            raise SessionBusy(session_id)
        queue = self._queues.get(session_id)
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if queue is None:
            queue = self._queues[session_id] = deque()
        queue.append(fut)
        try:
            if self._timeout is None:
                await asyncio.shield(fut)
            else:
                await asyncio.wait_for(asyncio.shield(fut), self._timeout)
        except BaseException as exc:
            if fut.done() and not fut.cancelled():
                # Granted in the same tick the waiter gave up: hand the slot on
                # so the engine is not left owned by a caller that is gone.
                self.release()
            else:
                fut.cancel()
                self._discard(session_id, fut)
            if isinstance(exc, asyncio.TimeoutError):
                raise QueueTimeout(session_id) from exc
            raise

    def _discard(self, session_id: str, fut: asyncio.Future[None]) -> None:
        queue = self._queues.get(session_id)
        if queue is None:
            return
        with contextlib.suppress(ValueError):
            queue.remove(fut)
        if not queue:
            del self._queues[session_id]

    def release(self) -> None:
        self._running = None
        while self._queues:
            session_id, queue = next(iter(self._queues.items()))
            fut = queue.popleft()
            del self._queues[session_id]
            if queue:
                # Re-append so the session goes to the back of the service order.
                self._queues[session_id] = queue
            if fut.cancelled():
                continue
            self._running = session_id
            fut.set_result(None)
            return
