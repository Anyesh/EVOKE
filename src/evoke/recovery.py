from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from evoke.types import ActiveBlock, BlockSource


@dataclass
class Breadcrumb:
    key: str
    token_count: int
    evicted_at_step: int


@dataclass
class SavedBlock:
    key: str
    kv_bytes: bytes
    token_ids: list[int]
    source: BlockSource
    representative_embedding: np.ndarray | None
    saved_at_step: int


class RecoveryBackend(Protocol):
    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None: ...

    def list_evicted(self) -> list[Breadcrumb]: ...

    def take(self, key: str) -> SavedBlock | None: ...


class DiscardBackend:
    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        pass

    def list_evicted(self) -> list[Breadcrumb]:
        return []

    def take(self, key: str) -> SavedBlock | None:
        return None


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

    def take(self, key: str) -> SavedBlock | None:
        return None


class KVRestoreBackend:
    def __init__(self, engine: object) -> None:
        self._engine = engine
        self._saved: dict[str, SavedBlock] = {}

    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        for block in blocks:
            kv_bytes = self._engine.kv_block_save(
                block.logical_start, block.logical_end
            )
            self._saved[block.key] = SavedBlock(
                key=block.key,
                kv_bytes=kv_bytes,
                token_ids=list(block.token_ids),
                source=block.source,
                representative_embedding=block.representative_embedding,
                saved_at_step=step,
            )

    def list_evicted(self) -> list[Breadcrumb]:
        return [
            Breadcrumb(s.key, len(s.token_ids), s.saved_at_step)
            for s in self._saved.values()
        ]

    def take(self, key: str) -> SavedBlock | None:
        return self._saved.pop(key, None)


def make_recovery_backend(mode: str, engine: object | None = None) -> RecoveryBackend:
    if mode == "discard":
        return DiscardBackend()
    if mode == "breadcrumb":
        return BreadcrumbBackend()
    if mode == "kv_restore":
        if engine is None:
            raise ValueError("kv_restore recovery mode requires an engine")
        return KVRestoreBackend(engine)
    raise ValueError(f"unknown recovery_mode: {mode!r}")
