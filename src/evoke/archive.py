from __future__ import annotations

from collections import OrderedDict

import numpy as np

from evoke.config import EvokeConfig
from evoke.scorer import cosine_similarity
from evoke.types import ArchiveBlock


class ArchiveStore:
    def __init__(self, config: EvokeConfig):
        self._config = config
        self._blocks: OrderedDict[int, ArchiveBlock] = OrderedDict()
        self._next_id = 0

    def store(self, block: ArchiveBlock) -> None:
        self._blocks[block.block_id] = block
        self._blocks.move_to_end(block.block_id)
        self._evict_if_over_capacity()

    def retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        max_results: int,
        query_text: str = "",
        min_lexical_recall: float = 0.0,
    ) -> list[ArchiveBlock]:
        semantic_hits: list[tuple[float, ArchiveBlock]] = []
        for block in self._blocks.values():
            sim = cosine_similarity(query_embedding, block.representative_embedding)
            if sim >= threshold:
                semantic_hits.append((sim, block))
        semantic_hits.sort(key=lambda x: x[0], reverse=True)

        lexical_hits: list[tuple[float, ArchiveBlock]] = []
        if query_text and threshold <= 1.0:
            query_words = _tokenize_words(query_text)
            if query_words:
                for block in self._blocks.values():
                    lex = _lexical_recall(query_words, block.text)
                    if lex > min_lexical_recall:
                        lexical_hits.append((lex, block))
                lexical_hits.sort(key=lambda x: x[0], reverse=True)

        seen: set[int] = set()
        hits: list[ArchiveBlock] = []
        for _, block in lexical_hits[:max_results]:
            if block.block_id not in seen:
                seen.add(block.block_id)
                hits.append(block)
        for _, block in semantic_hits[:max_results]:
            if block.block_id not in seen:
                seen.add(block.block_id)
                hits.append(block)

        if self._config.expand_neighbors:
            results = self._expand_neighbors(hits, seen)
        else:
            results = hits
        results.sort(key=lambda b: b.pos_start)

        for block in results:
            block.access_count += 1
        return results

    def _expand_neighbors(
        self, hits: list[ArchiveBlock], seen: set[int]
    ) -> list[ArchiveBlock]:
        results = list(hits)
        for hit in hits:
            for block in self._blocks.values():
                if block.block_id in seen:
                    continue
                if block.pos_end == hit.pos_start or block.pos_start == hit.pos_end:
                    seen.add(block.block_id)
                    results.append(block)
        return results

    def remove(self, block_id: int) -> ArchiveBlock | None:
        return self._blocks.pop(block_id, None)

    def get(self, block_id: int) -> ArchiveBlock | None:
        return self._blocks.get(block_id)

    @property
    def size(self) -> int:
        return len(self._blocks)

    @property
    def total_tokens(self) -> int:
        return sum(b.size for b in self._blocks.values())

    def all_blocks(self) -> list[ArchiveBlock]:
        return list(self._blocks.values())

    def allocate_id(self) -> int:
        bid = self._next_id
        self._next_id += 1
        return bid

    def _evict_if_over_capacity(self) -> None:
        while len(self._blocks) > self._config.max_archive_blocks:
            self._blocks.popitem(last=False)


_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "and but or nor not no so yet both either neither each every all "
    "any few more most other some such than too very it its this that "
    "these those i me my we our you your he him his she her they them their "
    "what which who whom how when where why".split()
)


def _tokenize_words(text: str) -> set[str]:
    cleaned = text.lower()
    for ch in ".,;:!?\"'()[]{}/-":
        cleaned = cleaned.replace(ch, " ")
    return {w for w in cleaned.split() if len(w) > 1 and w not in _STOPWORDS}


def _lexical_recall(query_words: set[str], block_text: str) -> float:
    if not query_words:
        return 0.0
    block_words = _tokenize_words(block_text)
    hits = query_words & block_words
    return len(hits) / len(query_words)
