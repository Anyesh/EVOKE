"""Retrieval-quality text embeddings for smart-recovery scoring.

LM intermediate hidden states (what `get_embeddings` exposes) carry strong
common-mode similarity between any two pieces of fluent English written in
similar registers, which collapses cosine discrimination to a narrow band
(~0.85-0.93 on the NIAH haystack). That band is too tight for smart-recovery
to reliably pick the needle block over noise.

This module wraps fastembed (default model BAAI/bge-small-en-v1.5, ~30 MB,
trained specifically for retrieval) so block and query embeddings come from
the same retrieval-tuned space. Cosine discrimination on the same workload
widens to ~0.3-0.9 — enough that the resident-gate cleanly separates the
needle from noise.

The cache directory defaults to `<repo>/.fastembed_cache` so the model bytes
live with the project.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding


_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _default_cache_dir() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent / ".fastembed_cache")
    return str(Path.home() / ".cache" / "fastembed")


class RetrievalEmbedder:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir or _default_cache_dir()
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        os.makedirs(self._cache_dir, exist_ok=True)
        self._model = TextEmbedding(
            model_name=self._model_name,
            cache_dir=self._cache_dir,
        )

    def embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        assert self._model is not None
        for emb in self._model.embed([text]):
            return np.asarray(emb, dtype=np.float32)
        return np.zeros(self.dim, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_model()
        assert self._model is not None
        if not texts:
            return []
        return [np.asarray(e, dtype=np.float32) for e in self._model.embed(texts)]

    @property
    def dim(self) -> int:
        # bge-small-en-v1.5 emits 384-dim vectors. Hard-coded so callers can
        # pre-allocate without forcing model load just to read a constant.
        return 384
