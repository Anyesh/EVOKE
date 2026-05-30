from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class Priority(enum.IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Status(enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ARCHIVED = "archived"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Task:
    id: str
    title: str
    owner: str
    priority: Priority = Priority.MEDIUM
    status: Status = Status.OPEN
    created_at: datetime = field(default_factory=_now)
    due_date: datetime | None = None
    tags: list[str] = field(default_factory=list)

    def is_overdue(self) -> bool:
        return (
            self.due_date is not None
            and self.status is not Status.DONE
            and self.due_date < _now()
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "owner": self.owner,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "tags": list(self.tags),
        }
