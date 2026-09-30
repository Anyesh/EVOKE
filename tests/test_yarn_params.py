from __future__ import annotations

import pytest

from evoke.llama_engine import ROPE_SCALING_YARN, yarn_context_params


def test_yarn_disabled_by_default():
    assert yarn_context_params(0.0, 0) == {}


def test_qwen3_factor_four_over_32k():
    params = yarn_context_params(4.0, 32768)
    assert params["rope_scaling_type"] == ROPE_SCALING_YARN
    assert params["rope_freq_scale"] == pytest.approx(0.25)
    assert params["yarn_orig_ctx"] == 32768
    assert params["yarn_ext_factor"] == 1.0
    assert params["yarn_beta_fast"] == 32.0
    assert params["yarn_beta_slow"] == 1.0


def test_factor_needs_original_context():
    with pytest.raises(ValueError):
        yarn_context_params(4.0, 0)


def test_factor_below_one_is_rejected():
    with pytest.raises(ValueError):
        yarn_context_params(0.5, 32768)
