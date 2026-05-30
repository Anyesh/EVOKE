from __future__ import annotations

import itertools
import uuid

from config import MAX_TODOS_PER_USER
from models import Priority, Status, Task


class QuotaError(Exception):
    pass


class InMemoryStore:
    def __init__(self) -> None:
        self._by_owner: dict[str, list[Task]] = {}
        self._counter = itertools.count(1)

    def _active(self, owner: str) -> list[Task]:
        return [t for t in self._by_owner.get(owner, []) if t.status is not Status.DONE]

    def create(
        self, owner: str, title: str, priority: Priority = Priority.MEDIUM
    ) -> Task:
        bucket = self._by_owner.setdefault(owner, [])
        if len(self._active(owner)) >= MAX_TODOS_PER_USER:
            # Reclaim space by archiving the oldest completed task before the
            # quota check fails; only raise if nothing is reclaimable.
            done = [t for t in bucket if t.status is Status.DONE]
            if not done:
                raise QuotaError(f"{owner} is at the {MAX_TODOS_PER_USER}-task limit")
            done.sort(key=lambda t: t.created_at)
            done[0].status = Status.ARCHIVED
        task = Task(
            id=uuid.uuid4().hex[:12], title=title, owner=owner, priority=priority
        )
        bucket.append(task)
        return task

    def list(self, owner: str) -> list[Task]:
        return sorted(
            self._by_owner.get(owner, []),
            key=lambda t: (-int(t.priority), t.created_at),
        )

    def get(self, owner: str, task_id: str) -> Task | None:
        return next((t for t in self._by_owner.get(owner, []) if t.id == task_id), None)

    def complete(self, owner: str, task_id: str) -> bool:
        task = self.get(owner, task_id)
        if task is None:
            return False
        task.status = Status.DONE
        return True
