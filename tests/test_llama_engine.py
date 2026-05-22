from __future__ import annotations

import os

import pytest

from evoke.engine import InferenceEngine

GGUF_PATH = os.environ.get("EVOKE_MODEL_PATH", "")
requires_model = pytest.mark.skipif(
    not GGUF_PATH or not os.path.isfile(GGUF_PATH),
    reason="Set EVOKE_MODEL_PATH to a .gguf file to run integration tests",
)


class TestResolveKvType:
    """Pure-function tests for the EVOKE_KV_QUANT env var entry point.

    These do not need a real model; they cover the string-to-ggml-enum
    translation that the bench harnesses go through when EVOKE_KV_QUANT
    is set. Without this coverage a typo in a strategy override or env
    var would silently fall back to F16 and the kv_quant comparison
    would mislabel a Q4 cell as F16.
    """

    def test_none_returns_default(self):
        from evoke.llama_engine import _resolve_kv_type

        assert _resolve_kv_type(None) == 1

    def test_none_returns_explicit_default(self):
        from evoke.llama_engine import _resolve_kv_type

        assert _resolve_kv_type(None, default=8) == 8

    def test_int_passthrough(self):
        from evoke.llama_engine import _resolve_kv_type

        assert _resolve_kv_type(8) == 8
        assert _resolve_kv_type(2) == 2

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("f16", 1),
            ("F16", 1),
            ("q4_0", 2),
            ("Q4_0", 2),
            ("q8_0", 8),
            ("q5_1", 7),
            ("f32", 0),
        ],
    )
    def test_known_names(self, name, expected):
        from evoke.llama_engine import _resolve_kv_type

        assert _resolve_kv_type(name) == expected

    def test_unknown_name_raises(self):
        from evoke.llama_engine import _resolve_kv_type

        with pytest.raises(ValueError, match="unknown kv cache type"):
            _resolve_kv_type("not_a_real_type")

    def test_whitespace_stripped(self):
        from evoke.llama_engine import _resolve_kv_type

        assert _resolve_kv_type("  q4_0  ") == 2


@requires_model
class TestLlamaCppEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from evoke.llama_engine import LlamaCppEngine

        eng = LlamaCppEngine(GGUF_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        yield eng
        eng.close()

    def test_satisfies_protocol(self, engine):
        assert isinstance(engine, InferenceEngine)

    def test_tokenize_roundtrip(self, engine):
        text = "Hello, world!"
        tokens = engine.tokenize(text)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
        decoded = engine.detokenize(tokens)
        assert "Hello" in decoded

    def test_process_and_generate(self, engine):
        engine.reset()
        tokens = engine.tokenize("The capital of France is")
        engine.process_tokens(tokens)
        assert engine.get_kv_cache_token_count() == len(tokens)

        generated = engine.generate_next()
        assert isinstance(generated, int)
        assert engine.get_kv_cache_token_count() == len(tokens) + 1

    def test_evict_ranges(self, engine):
        engine.reset()
        tokens = engine.tokenize("one two three four five")
        engine.process_tokens(tokens)
        count_before = engine.get_kv_cache_token_count()

        engine.evict_ranges([(0, 2)])
        assert engine.get_kv_cache_token_count() == count_before - 2
        assert engine.next_write_pos == count_before - 2

    def test_n_ctx(self, engine):
        assert engine.n_ctx == 2048

    def test_n_embd(self, engine):
        assert engine.n_embd > 0

    def test_eos_bos_tokens(self, engine):
        assert isinstance(engine.eos_token, int)
        assert isinstance(engine.bos_token, int)

    def test_embeddings_after_process(self, engine):
        import numpy as np

        engine.reset()
        tokens = engine.tokenize("Hello world test")
        engine.process_tokens(tokens)

        embeddings = engine.get_embeddings(list(range(len(tokens))))
        assert embeddings.shape == (len(tokens), engine.n_embd)
        assert np.linalg.norm(embeddings[0]) > 0


@requires_model
class TestLlamaCppWithManager:
    @pytest.fixture(scope="class")
    def engine(self):
        from evoke.llama_engine import LlamaCppEngine

        eng = LlamaCppEngine(GGUF_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
        yield eng
        eng.close()

    def test_load_document_and_generate(self, engine):
        from evoke.config import EvokeConfig
        from evoke.manager import EvokeManager

        config = EvokeConfig(max_active_tokens=1024, block_size=64)
        mgr = EvokeManager(engine, config)

        doc = "The quick brown fox jumps over the lazy dog. " * 50
        mgr.load_document(doc)

        stats = mgr.get_stats()
        assert stats.active_tokens <= config.max_active_tokens
        assert stats.active_tokens > 0

        answer = mgr.generate(16)
        assert isinstance(answer, str)
        assert len(answer) > 0
