from __future__ import annotations

import ctypes

import pytest

from evoke._engine_lib import context_params_class, detect_extra_u32


def _defaults(extra: int) -> bytes:
    params = context_params_class(extra)()
    params.n_ctx = 512
    params.n_batch = 2048
    params.n_threads = 8
    params.n_threads_batch = 8
    params.rope_scaling_type = -1
    params.pooling_type = -1
    params.attention_type = -1
    params.flash_attn_type = -1
    params.yarn_ext_factor = -1.0
    params.yarn_attn_factor = -1.0
    params.yarn_beta_fast = -1.0
    params.yarn_beta_slow = -1.0
    params.type_k = 1
    return bytes(params).ljust(1024, b"\xa5")


@pytest.mark.parametrize("extra", [0, 1, 2])
def test_each_fork_layout_is_recognised(extra):
    assert detect_extra_u32(_defaults(extra)) == extra


def test_unknown_layout_is_not_guessed():
    assert detect_extra_u32(b"\x00" * 1024) is None


def test_settings_land_on_their_own_fields():
    cls = context_params_class(1)
    params = cls()
    params.type_k = 8
    params.rope_freq_scale = 0.25
    raw = bytes(params)
    assert ctypes.sizeof(cls) == len(raw)
    other = context_params_class(0).from_buffer_copy(
        raw[: ctypes.sizeof(context_params_class(0))]
    )
    assert other.type_k != 8
