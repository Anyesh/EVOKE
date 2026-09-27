"""Per-turn EVOKE counters reported to clients as usage.evoke and X-EVOKE-*.

peak_resident_tokens is the high-water mark during the turn, so a client can
see when identity gap-fill transiently restored the whole working set before
end-of-turn eviction brought residency back down to resident_tokens_end.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from evoke.types import CacheStats

HEADER_FIELDS = (
    "logical_tokens",
    "logical_window",
    "kv_cells",
    "peak_resident_tokens",
    "resident_tokens_end",
    "pinned_tokens",
    "blocks_evicted",
    "blocks_recovered",
    "queue_wait_ms",
)


@dataclass(frozen=True)
class TurnMetrics:
    session: str
    logical_tokens: int
    logical_window: int
    kv_cells: int
    peak_resident_tokens: int
    resident_tokens_end: int
    pinned_tokens: int
    budget: int
    blocks_evicted: int
    tokens_evicted: int
    blocks_recovered: int
    tokens_recovered: int
    prompt_tokens_decoded: int
    prompt_tokens_reused: int
    queue_wait_ms: int

    def as_usage(self) -> dict[str, int | str]:
        return asdict(self)

    def as_headers(self) -> dict[str, str]:
        headers = {"X-EVOKE-Session": self.session}
        for name in HEADER_FIELDS:
            title = "-".join(part.capitalize() for part in name.split("_"))
            headers[f"X-EVOKE-{title}"] = str(getattr(self, name))
        return headers


def measure_turn(
    *,
    session: str,
    before: CacheStats,
    after: CacheStats,
    peak_resident_tokens: int,
    pinned_tokens: int,
    prompt_tokens: int,
    prompt_tokens_decoded: int,
    completion_tokens: int,
    kv_cells: int,
    logical_window: int,
    queue_wait_s: float,
) -> TurnMetrics:
    return TurnMetrics(
        session=session,
        logical_tokens=prompt_tokens + completion_tokens,
        logical_window=logical_window,
        kv_cells=kv_cells,
        peak_resident_tokens=max(peak_resident_tokens, after.active_tokens),
        resident_tokens_end=after.active_tokens,
        pinned_tokens=pinned_tokens,
        budget=after.budget,
        blocks_evicted=after.total_evictions - before.total_evictions,
        tokens_evicted=after.evicted_tokens - before.evicted_tokens,
        blocks_recovered=after.total_recoveries - before.total_recoveries,
        tokens_recovered=after.recovered_tokens - before.recovered_tokens,
        prompt_tokens_decoded=prompt_tokens_decoded,
        prompt_tokens_reused=max(0, prompt_tokens - prompt_tokens_decoded),
        queue_wait_ms=round(queue_wait_s * 1000),
    )
