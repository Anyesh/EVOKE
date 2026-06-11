from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.manager import EvokeManager
from evoke.mock_engine import MockEngine
from evoke.session import Session


class RecordingEngine(MockEngine):
    # Records every process_tokens call so a test can assert that an identity
    # splice was recompute-free (no decode of the spliced span).
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoded: list[list[int]] = []

    def process_tokens(self, tokens: list[int]) -> None:
        self.decoded.append(list(tokens))
        super().process_tokens(tokens)


def _identity_config(block_size: int = 4) -> EvokeConfig:
    # Budget far above content so the only eviction is the explicit force_evict;
    # sparse + kv_restore + identity is the live gap-fill configuration.
    return EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=block_size,
        sink_count=0,
        position_mode="sparse",
        recovery_mode="kv_restore",
        recovery_match="identity",
    )


def test_get_token_view_position_ordered_after_inplace_recover():
    engine = MockEngine()
    mgr = EvokeManager(engine, _identity_config())
    tokens = list(range(1, 13))  # 3 blocks of 4
    mgr.add_context_tokens(tokens, key="ctx")
    mid = next(b for b in mgr._positions.active_blocks if b.logical_start == 4)
    mgr.force_evict([mid.block_id])
    assert mgr.get_token_view() == [1, 2, 3, 4, 9, 10, 11, 12]
    crumb = next(c for c in mgr.get_breadcrumbs() if c.key == mid.key)
    assert mgr.recover(crumb.key)
    # The recovered block has the highest block_id but a mid logical_start;
    # block_id order would misorder the view. Position order is correct.
    assert mgr.get_token_view() == tokens


def test_identity_gap_fill_recompute_free_on_resend():
    engine = RecordingEngine()
    session = Session(engine, config=_identity_config())
    prompt = list(range(20))  # 5 blocks of 4
    s1 = session.sync_prefix(prompt)
    assert s1.new_tokens_decoded == 20
    assert engine.decoded == [prompt]
    # A middle block is evicted (as a budget pass would do on an earlier turn).
    mid = next(
        b for b in session._manager._positions.active_blocks if b.logical_start == 8
    )
    session._manager.force_evict([mid.block_id])
    # The agent re-sends full history; the evicted block reappears in place.
    s2 = session.sync_prefix(prompt)
    assert s2.blocks_recovered == 1
    assert s2.new_tokens_decoded == 0  # nothing re-decoded
    assert engine.decoded == [prompt]  # no new process_tokens call: recompute-free
    assert session._cached_tokens == prompt  # faithful full prefix restored


def test_changed_mid_history_tail_evicts_from_divergence():
    # Content changed at an evicted block's position: the matching prefix
    # below the divergence must survive and only the conflicting remainder
    # (changed hole + everything after it) re-decodes. The old behavior was
    # a full session reset because the tiling check demanded contiguity up
    # to the whole cached view, which trailing holes always fail.
    engine = RecordingEngine()
    session = Session(engine, config=_identity_config())
    prompt = list(range(20))
    session.sync_prefix(prompt)
    mgr = session._manager
    mid = next(b for b in mgr._positions.active_blocks if b.logical_start == 8)
    mgr.force_evict([mid.block_id])
    changed = list(range(8)) + [88, 89, 90, 91] + list(range(12, 20))
    s2 = session.sync_prefix(changed)
    assert s2.blocks_recovered == 0  # identity miss: content changed, no splice
    assert s2.new_tokens_decoded == 12  # only from the divergence on
    assert engine.decoded[-1] == changed[8:]
    assert session._manager is mgr  # no reset
    assert session._cached_tokens == changed
    assert engine._token_at_pos[8] == 88
    assert engine.next_write_pos == 20


def test_partial_gap_fill_keeps_recovered_prefix_on_later_mismatch():
    # Live pattern: a new request shares a long evicted prefix
    # (system prompt + tools), gap-fill splices it back block by block, then
    # hits changed content at a later hole. The spliced prefix must survive;
    # only the conflicting tail re-decodes. The old code reset the session
    # because unfilled holes past the mismatch failed the full-view tiling
    # check, discarding every block just recovered.
    engine = RecordingEngine()
    session = Session(engine, config=_identity_config())
    prompt = list(range(28))
    session.sync_prefix(prompt)
    mgr = session._manager
    for start in (8, 20):
        blk = next(b for b in mgr._positions.active_blocks if b.logical_start == start)
        mgr.force_evict([blk.block_id])
    changed = list(range(20)) + [200, 201, 202, 203] + list(range(24, 28))
    s2 = session.sync_prefix(changed)
    assert s2.blocks_recovered == 1  # the unchanged hole at 8 spliced back
    assert s2.new_tokens_decoded == 8  # changed hole at 20 + trailing block
    assert engine.decoded == [prompt, changed[20:]]
    assert session._manager is mgr  # no reset
    assert session._cached_tokens == changed
    assert engine._token_at_pos[8] == 8  # spliced, not re-decoded
    assert engine._token_at_pos[20] == 200
    assert engine._token_at_pos[24] == 24
    assert engine.next_write_pos == 28


def test_generated_tail_drift_preserves_gapfilled_prefix():
    # Live failure mode (qwen3-8b opencode run): gap-fill recovers the whole
    # evicted prefix, then the client's templated echo of the generated turn
    # drifts from the raw generation by a token, and the old code reset the
    # session, discarding every just-recovered block and re-decoding all of
    # it. The recovered prefix must survive; only the stale generated tail
    # plus the new content should decode.
    engine = RecordingEngine()
    session = Session(engine, config=_identity_config())
    prompt = list(range(20))
    session.sync_prefix(prompt)
    engine.queue_tokens([100, 101, 102, 103])
    session.generate(max_tokens=4)
    mid = next(
        b for b in session._manager._positions.active_blocks if b.logical_start == 8
    )
    session._manager.force_evict([mid.block_id])

    drifted_echo = [200, 201, 202, 203]
    turn2 = prompt + drifted_echo + list(range(30, 38))
    s2 = session.sync_prefix(turn2)
    assert s2.blocks_recovered == 1
    assert s2.new_tokens_decoded == 12
    assert engine.decoded[-1] == turn2[20:]
    assert session._cached_tokens == turn2
    # The tail must decode AT the dropped positions, not past them: a stale
    # engine write cursor after the tail-evict crashed llama_decode live.
    assert engine._token_at_pos[20] == 200
    assert engine._token_at_pos[24] == 30
    assert engine.next_write_pos == len(turn2)


def test_similarity_flag_redecodes_tail():
    cfg = EvokeConfig(
        max_active_tokens=1_000_000,
        block_size=4,
        sink_count=0,
        position_mode="compact",
        recovery_mode="kv_restore",
        recovery_match="similarity",
    )
    engine = RecordingEngine()
    session = Session(engine, config=cfg)
    prompt = list(range(20))
    session.sync_prefix(prompt)
    mid = next(
        b for b in session._manager._positions.active_blocks if b.logical_start == 8
    )
    session._manager.force_evict([mid.block_id])
    s2 = session.sync_prefix(prompt)
    # The similarity path does not identity-splice; the diverged tail is decoded.
    assert s2.new_tokens_decoded > 0
