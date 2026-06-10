"""Prefix-affinity session routing.

Stateless OpenAI clients (opencode) cannot send a session header, so every
request landed on the one default session. An interleaved side-request such
as opencode's title generation diverges the prefix, and the identity-match
divergence path resets the session, destroying the saved-KV archive that
identity recovery needs. Routing by longest shared token prefix keeps the
agent thread on its own session and side-requests on theirs.
"""

from __future__ import annotations

from evoke.config import EvokeConfig
from evoke.mock_engine import MockEngine
from evoke.session import SessionPool


def _make_pool(max_sessions: int = 8) -> SessionPool:
    cfg = EvokeConfig(
        max_active_tokens=2048,
        block_size=128,
        recovery_mode="discard",
    )
    return SessionPool(MockEngine(), config=cfg, max_sessions=max_sessions)


class TestPrefixAffinityRouting:
    def test_first_request_gets_fresh_auto_session(self):
        pool = _make_pool()
        sid = pool.route_id(list(range(200)))
        assert sid.startswith("auto-")

    def test_resent_conversation_routes_to_same_session(self):
        pool = _make_pool()
        prompt1 = list(range(500))
        sid1 = pool.route_id(prompt1)
        pool.get(sid1).sync_prefix(prompt1)
        prompt2 = prompt1 + list(range(1000, 1100))
        assert pool.route_id(prompt2) == sid1

    def test_unrelated_request_gets_new_session(self):
        pool = _make_pool()
        prompt1 = list(range(500))
        sid1 = pool.route_id(prompt1)
        pool.get(sid1).sync_prefix(prompt1)
        title_prompt = [1, 2, 3] + list(range(9000, 9100))
        sid2 = pool.route_id(title_prompt)
        assert sid2 != sid1

    def test_interleaved_side_request_does_not_steal_agent_session(self):
        pool = _make_pool()
        agent1 = list(range(500))
        agent_sid = pool.route_id(agent1)
        pool.get(agent_sid).sync_prefix(agent1)

        title = [1, 2, 3] + list(range(9000, 9100))
        title_sid = pool.route_id(title)
        assert title_sid != agent_sid
        pool.get(title_sid).sync_prefix(title)

        agent2 = agent1 + list(range(1000, 1100))
        assert pool.route_id(agent2) == agent_sid

    def test_short_shared_prefix_below_threshold_separates(self):
        pool = _make_pool()
        p1 = list(range(500))
        sid1 = pool.route_id(p1)
        pool.get(sid1).sync_prefix(p1)
        p2 = p1[:8] + list(range(7000, 7400))
        assert pool.route_id(p2) != sid1
