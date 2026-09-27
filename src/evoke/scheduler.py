"""Admission control for chat turns on a single engine context.

One llama_context decodes one session at a time, so turns cannot run
concurrently. The scheduler makes the wait fair instead: waiting turns are
grouped by owner (the API key; the session itself in open mode), and when the
engine frees up the next turn comes from the owner after the one just served.
Within an owner turns run in arrival order. Round-robin by owner rather than
by session means opening more sessions buys no extra share of the engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import Counter, OrderedDict, deque
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
        # Owners with waiters, in service order; the head is served next.
        self._owners: OrderedDict[str, deque[tuple[str, asyncio.Future[None]]]] = (
            OrderedDict()
        )
        self._waiting_by_session: Counter[str] = Counter()

    @property
    def running(self) -> str | None:
        return self._running

    @property
    def waiting(self) -> int:
        return sum(self._waiting_by_session.values())

    @contextlib.asynccontextmanager
    async def turn(
        self, session_id: str, *, owner: str | None = None
    ) -> AsyncIterator[float]:
        start = time.monotonic()
        await self.acquire(session_id, owner=owner)
        try:
            yield time.monotonic() - start
        finally:
            self.release()

    def admissible(self, session_id: str) -> bool:
        if self._running is None and not self._owners:
            return True
        if self._waiting_by_session[session_id] >= self._max_waiting:
            return False
        return not (self._running == session_id and self._max_waiting == 0)

    async def acquire(self, session_id: str, *, owner: str | None = None) -> None:
        if self._running is None and not self._owners:
            self._running = session_id
            return
        if not self.admissible(session_id):
            raise SessionBusy(session_id)
        owner = owner if owner is not None else session_id
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._owners.setdefault(owner, deque()).append((session_id, fut))
        self._waiting_by_session[session_id] += 1
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
                self._discard(owner, session_id, fut)
            if isinstance(exc, asyncio.TimeoutError):
                raise QueueTimeout(session_id) from exc
            raise

    def _discard(self, owner: str, session_id: str, fut: asyncio.Future[None]) -> None:
        queue = self._owners.get(owner)
        if queue is None:
            return
        with contextlib.suppress(ValueError):
            queue.remove((session_id, fut))
            self._forget_waiter(session_id)
        if not queue:
            del self._owners[owner]

    def _forget_waiter(self, session_id: str) -> None:
        self._waiting_by_session[session_id] -= 1
        if self._waiting_by_session[session_id] <= 0:
            del self._waiting_by_session[session_id]

    def release(self) -> None:
        self._running = None
        while self._owners:
            owner, queue = next(iter(self._owners.items()))
            session_id, fut = queue.popleft()
            del self._owners[owner]
            if queue:
                # Re-append so the owner goes to the back of the service order.
                self._owners[owner] = queue
            self._forget_waiter(session_id)
            if fut.cancelled():
                continue
            self._running = session_id
            fut.set_result(None)
            return
