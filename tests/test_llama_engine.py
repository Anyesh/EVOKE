from __future__ import annotations

import os

import pytest

from evoke.engine import InferenceEngine

GGUF_PATH = os.environ.get("EVOKE_MODEL_PATH", "")
requires_model = pytest.mark.skipif(
    not GGUF_PATH or not os.path.isfile(GGUF_PATH),
    reason="Set EVOKE_MODEL_PATH to a .gguf file to run integration tests",
)


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

    def test_kv_cache_seq_rm(self, engine):
        engine.reset()
        tokens = engine.tokenize("one two three four five")
        engine.process_tokens(tokens)
        count_before = engine.get_kv_cache_token_count()

        engine.kv_cache_seq_rm(0, 2)
        count_after = engine.get_kv_cache_token_count()
        assert count_after == count_before - 2

    def test_rebuild_kv(self, engine):
        engine.reset()
        tokens = engine.tokenize("alpha beta gamma")
        engine.process_tokens(tokens)
        count_before = engine.get_kv_cache_token_count()

        engine.rebuild_kv([tokens])
        assert engine.get_kv_cache_token_count() == count_before
        assert engine.next_write_pos == len(tokens)

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
