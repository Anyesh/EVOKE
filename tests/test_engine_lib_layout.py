from __future__ import annotations

import ctypes

import pytest

from evoke._engine_lib import (
    context_params_class,
    detect_context_layout,
    detect_model_layout,
    model_params_class,
)


def _defaults(extra: int, moe_cache: bool = False) -> bytes:
    params = context_params_class(extra, moe_cache)()
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
    params.offload_kqv = True
    params.no_perf = True
    params.op_offload = True
    params.swa_full = True
    return bytes(params).ljust(1024, b"\xa5")


def _model_defaults(load_mode: bool) -> bytes:
    params = model_params_class(load_mode)()
    params.n_gpu_layers = -1
    params.split_mode = 1
    params.use_extra_bufts = True
    if load_mode:
        params.load_mode = -1
        params.lazy_mode = 1
    else:
        params.use_mmap = True
    return bytes(params).ljust(1024, b"\xa5")


@pytest.mark.parametrize("moe_cache", [False, True])
@pytest.mark.parametrize("extra", [0, 1, 2])
def test_each_fork_layout_is_recognised(extra, moe_cache):
    assert detect_context_layout(_defaults(extra, moe_cache)) == (extra, moe_cache)


def test_unknown_layout_is_not_guessed():
    assert detect_context_layout(b"\x00" * 1024) is None


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


def test_moe_cache_field_shifts_the_flags():
    old = context_params_class(2, False)
    new = context_params_class(2, True)
    assert new.offload_kqv.offset - old.offload_kqv.offset == ctypes.sizeof(
        ctypes.c_size_t
    )


@pytest.mark.parametrize("load_mode", [False, True])
def test_each_model_layout_is_recognised(load_mode):
    assert detect_model_layout(_model_defaults(load_mode)) is load_mode


def test_unknown_model_layout_is_not_guessed():
    assert detect_model_layout(b"\x00" * 1024) is None


def test_n_gpu_layers_offset_is_shared_by_both_model_layouts():
    assert (
        model_params_class(True).n_gpu_layers.offset
        == model_params_class(False).n_gpu_layers.offset
    )
