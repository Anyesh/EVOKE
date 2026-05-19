from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evoke.types import ActiveBlock


@dataclass
class Breadcrumb:
    key: str
    token_count: int
    evicted_at_step: int


class RecoveryBackend(Protocol):
    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None: ...

    def list_evicted(self) -> list[Breadcrumb]: ...


class DiscardBackend:
    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        pass

    def list_evicted(self) -> list[Breadcrumb]:
        return []


class BreadcrumbBackend:
    def __init__(self) -> None:
        self._crumbs: dict[str, Breadcrumb] = {}

    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        for block in blocks:
            self._crumbs[block.key] = Breadcrumb(
                key=block.key,
                token_count=len(block.token_ids),
                evicted_at_step=step,
            )

    def list_evicted(self) -> list[Breadcrumb]:
        return list(self._crumbs.values())


def make_recovery_backend(mode: str) -> RecoveryBackend:
    if mode == "discard":
        return DiscardBackend()
    if mode == "breadcrumb":
        return BreadcrumbBackend()
    raise ValueError(f"unknown recovery_mode: {mode!r}")
