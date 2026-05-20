from __future__ import annotations

from collections import OrderedDict
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

    def peek_embedding(self, key: str) -> np.ndarray | None: ...


class DiscardBackend:
    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        pass

    def list_evicted(self) -> list[Breadcrumb]:
        return []

    def take(self, key: str) -> SavedBlock | None:
        return None

    def peek_embedding(self, key: str) -> np.ndarray | None:
        return None


class BreadcrumbBackend:
    def __init__(self) -> None:
        self._crumbs: dict[str, Breadcrumb] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        for block in blocks:
            self._crumbs[block.key] = Breadcrumb(
                key=block.key,
                token_count=len(block.token_ids),
                evicted_at_step=step,
            )
            if block.representative_embedding is not None:
                self._embeddings[block.key] = block.representative_embedding

    def list_evicted(self) -> list[Breadcrumb]:
        return list(self._crumbs.values())

    def take(self, key: str) -> SavedBlock | None:
        return None

    def peek_embedding(self, key: str) -> np.ndarray | None:
        return self._embeddings.get(key)


class KVRestoreBackend:
    def __init__(self, engine: object, ram_budget_bytes: int | None = None) -> None:
        self._engine = engine
        # OrderedDict preserves insertion order, which we use as LRU order:
        # popitem(last=False) drops the oldest entry. We move-to-end on
        # successful take() so a take-then-resave cycle treats the block as
        # freshly used. Total bytes is tracked separately so we don't have
        # to iterate the dict on every save.
        self._saved: OrderedDict[str, SavedBlock] = OrderedDict()
        self._breadcrumbs: dict[str, Breadcrumb] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._total_bytes = 0
        self._budget = ram_budget_bytes
        self._lru_evictions = 0

    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        for block in blocks:
            kv_bytes = self._engine.kv_block_save(
                block.logical_start, block.logical_end
            )
            # If the same key already has a saved blob (e.g. a re-evict after
            # a transient recover that didn't take), free the old bytes
            # before tracking the new ones.
            old = self._saved.pop(block.key, None)
            if old is not None:
                self._total_bytes -= len(old.kv_bytes)
            self._saved[block.key] = SavedBlock(
                key=block.key,
                kv_bytes=kv_bytes,
                token_ids=list(block.token_ids),
                source=block.source,
                representative_embedding=block.representative_embedding,
                saved_at_step=step,
            )
            self._total_bytes += len(kv_bytes)
            self._breadcrumbs[block.key] = Breadcrumb(
                key=block.key,
                token_count=len(block.token_ids),
                evicted_at_step=step,
            )
            if block.representative_embedding is not None:
                self._embeddings[block.key] = block.representative_embedding
            self._enforce_ram_budget()

    def _enforce_ram_budget(self) -> None:
        # Drop oldest saved K/V bytes until under budget, keeping at least one
        # entry alive (so a single oversized block doesn't get LRU'd out the
        # moment it lands). The breadcrumb + embedding survive demotion so
        # the scorer's peek_embedding still works and a caller asking
        # list_evicted still sees the block; only the K/V bytes are dropped.
        # After demotion, take(key) returns None — the caller falls through
        # to its breadcrumb / discard path naturally.
        if self._budget is None:
            return
        while self._total_bytes > self._budget and len(self._saved) > 1:
            _key, victim = self._saved.popitem(last=False)
            self._total_bytes -= len(victim.kv_bytes)
            self._lru_evictions += 1

    def list_evicted(self) -> list[Breadcrumb]:
        # Includes both still-saved blocks AND LRU-demoted ones. The
        # smart-recovery path checks peek_embedding for scoring; if a key
        # comes back with embedding but take() returns None, the caller
        # treats it as a breadcrumb-only entry.
        return list(self._breadcrumbs.values())

    def take(self, key: str) -> SavedBlock | None:
        saved = self._saved.pop(key, None)
        if saved is None:
            return None
        self._total_bytes -= len(saved.kv_bytes)
        return saved

    def peek_embedding(self, key: str) -> np.ndarray | None:
        return self._embeddings.get(key)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def lru_evictions(self) -> int:
        return self._lru_evictions


def make_recovery_backend(
    mode: str,
    engine: object | None = None,
    *,
    kv_restore_ram_budget_bytes: int | None = None,
) -> RecoveryBackend:
    if mode == "discard":
        return DiscardBackend()
    if mode == "breadcrumb":
        return BreadcrumbBackend()
    if mode == "kv_restore":
        if engine is None:
            raise ValueError("kv_restore recovery mode requires an engine")
        return KVRestoreBackend(engine, ram_budget_bytes=kv_restore_ram_budget_bytes)
    raise ValueError(f"unknown recovery_mode: {mode!r}")
