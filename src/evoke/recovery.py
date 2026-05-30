from __future__ import annotations

import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
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
    # The block's logical start position at eviction time. In sparse mode this
    # is the true absolute index the block must be spliced back into on
    # recovery (no re-anchoring); in compact mode it is informational
    # (recovery targets the tail instead).
    original_start: int = 0


class RecoveryBackend(Protocol):
    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None: ...

    def list_evicted(self) -> list[Breadcrumb]: ...

    def take(self, key: str) -> SavedBlock | None: ...

    def peek(self, key: str) -> SavedBlock | None: ...

    def peek_embedding(self, key: str) -> np.ndarray | None: ...


class DiscardBackend:
    def on_evict(self, blocks: list[ActiveBlock], step: int) -> None:
        pass

    def list_evicted(self) -> list[Breadcrumb]:
        return []

    def take(self, key: str) -> SavedBlock | None:
        return None

    def peek(self, key: str) -> SavedBlock | None:
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

    def peek(self, key: str) -> SavedBlock | None:
        return None

    def peek_embedding(self, key: str) -> np.ndarray | None:
        return self._embeddings.get(key)


class KVRestoreBackend:
    def __init__(
        self,
        engine: object,
        ram_budget_bytes: int | None = None,
        spill_path: str | None = None,
    ) -> None:
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
        # Disk spill tier: when an LRU-victim would otherwise drop its K/V
        # bytes, we instead write them to <spill_path>/<unique>.bin and
        # remember the filename. take() can read them back at NVMe speed.
        # spill_path=None disables the tier (preserves old "drop on LRU"
        # behavior).
        self._spill_path: Path | None = Path(spill_path) if spill_path else None
        if self._spill_path is not None:
            self._spill_path.mkdir(parents=True, exist_ok=True)
        self._spilled: dict[str, tuple[Path, SavedBlock]] = {}
        self._spill_evictions = 0

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
                original_start=block.logical_start,
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
        # When over budget, demote the oldest saved block: either spill its
        # K/V bytes to the disk tier (still recoverable via take(), just
        # slower) or — if no spill is configured — drop the bytes entirely
        # (the breadcrumb + embedding survive in either case). Keep at
        # least one entry in RAM so a single oversized block doesn't get
        # demoted the moment it lands.
        if self._budget is None:
            return
        while self._total_bytes > self._budget and len(self._saved) > 1:
            key, victim = self._saved.popitem(last=False)
            self._total_bytes -= len(victim.kv_bytes)
            if self._spill_path is not None:
                self._spill_to_disk(key, victim)
                self._spill_evictions += 1
            else:
                self._lru_evictions += 1

    def _spill_to_disk(self, key: str, block: SavedBlock) -> None:
        # Write the K/V bytes out under a unique filename. The SavedBlock
        # tuple (minus kv_bytes — those moved to disk) is held in memory
        # so take() can reconstruct the full block by re-reading the file.
        # Using a uuid hex prefix means concurrent backends pointed at the
        # same spill dir won't collide even without locking the directory.
        assert self._spill_path is not None
        fname = self._spill_path / f"evoke-spill-{uuid.uuid4().hex}.bin"
        with open(fname, "wb") as f:
            f.write(block.kv_bytes)
        meta = SavedBlock(
            key=block.key,
            kv_bytes=b"",  # bytes are on disk; placeholder here.
            token_ids=block.token_ids,
            source=block.source,
            representative_embedding=block.representative_embedding,
            saved_at_step=block.saved_at_step,
            original_start=block.original_start,
        )
        self._spilled[key] = (fname, meta)

    def list_evicted(self) -> list[Breadcrumb]:
        # Includes still-RAM blocks AND demoted (spilled or dropped) ones.
        # The smart-recovery path checks peek_embedding for scoring; if a
        # key comes back with embedding but take() returns None, the
        # caller treats it as a breadcrumb-only entry.
        return list(self._breadcrumbs.values())

    def take(self, key: str) -> SavedBlock | None:
        # RAM first.
        saved = self._saved.pop(key, None)
        if saved is not None:
            self._total_bytes -= len(saved.kv_bytes)
            return saved
        # Disk tier: read the bytes back, delete the file, hand back the
        # reconstructed block. The breadcrumb stays (caller may probe it
        # again).
        spilled = self._spilled.pop(key, None)
        if spilled is None:
            return None
        fname, meta = spilled
        try:
            with open(fname, "rb") as f:
                kv_bytes = f.read()
        except OSError:
            return None
        try:
            fname.unlink()
        except OSError:
            pass
        return SavedBlock(
            key=meta.key,
            kv_bytes=kv_bytes,
            token_ids=meta.token_ids,
            source=meta.source,
            representative_embedding=meta.representative_embedding,
            saved_at_step=meta.saved_at_step,
            original_start=meta.original_start,
        )

    def peek(self, key: str) -> SavedBlock | None:
        # Non-destructive lookup for identity gap-fill: return the SavedBlock
        # (token_ids + original_start) without consuming it, so the caller can
        # compare content identity before committing to recover()/take(). For a
        # spilled block the meta carries token_ids/original_start (kv_bytes is a
        # placeholder); take() reads the real bytes from disk on recovery.
        saved = self._saved.get(key)
        if saved is not None:
            return saved
        spilled = self._spilled.get(key)
        if spilled is not None:
            return spilled[1]
        return None

    def peek_embedding(self, key: str) -> np.ndarray | None:
        return self._embeddings.get(key)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def lru_evictions(self) -> int:
        return self._lru_evictions

    @property
    def spill_evictions(self) -> int:
        return self._spill_evictions

    @property
    def n_spilled(self) -> int:
        return len(self._spilled)


def make_recovery_backend(
    mode: str,
    engine: object | None = None,
    *,
    kv_restore_ram_budget_bytes: int | None = None,
    kv_restore_spill_path: str | None = None,
) -> RecoveryBackend:
    if mode == "discard":
        return DiscardBackend()
    if mode == "breadcrumb":
        return BreadcrumbBackend()
    if mode == "kv_restore":
        if engine is None:
            raise ValueError("kv_restore recovery mode requires an engine")
        return KVRestoreBackend(
            engine,
            ram_budget_bytes=kv_restore_ram_budget_bytes,
            spill_path=kv_restore_spill_path,
        )
    raise ValueError(f"unknown recovery_mode: {mode!r}")
