"""Parameter ablation harness on top of the NIAH workload.

Sweeps one EVOKE knob at a time across the full 25-cell (5 needles x 5 depths)
NIAH grid at a single budget, reporting pass-rate per knob value. Targets the
"undefended defaults" the reviewer flagged: block_size, attention_capture_layer,
smart_recover_k, w_coherence, recent_tail_protect_frac.

Each ablation runs a base strategy (default evoke_attention, since it exercises
both kv_restore recovery and attention scoring) with one config field varied.
Other strategies are not swept because the knob isn't meaningful outside the
strategy that uses it.

Environment:
- EVOKE_ABL_DIM       block_size | attention_layer | topk | wcoherence | tailprotect
- EVOKE_ABL_VALUES    comma-separated values (defaults below)
- EVOKE_ABL_BASE      base strategy id (default evoke_attention)
- EVOKE_BUDGETS       single budget (default 1024)
- EVOKE_NIAH_NEEDLES  needle subset (default all 5)
- EVOKE_NIAH_DEPTHS   depth subset (default 5,25,50,75,95)
- EVOKE_NIAH_PARAGRAPHS  haystack size (default 40)
- EVOKE_ABL_JSON      output JSON path
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evoke.llama_engine import LlamaCppEngine

from niah_bench import (
    DEFAULT_DEPTHS,
    NEEDLES,
    STRATEGIES,
    run_cell,
)


@dataclass
class AblationCell:
    dim: str
    value: object
    needle_id: str
    depth: int
    strategy: str
    budget: int
    probe_ok: bool
    evictions: int
    recoveries: int
    active_tokens: int
    elapsed: float
    answer: str


_DIM_CONFIGS: dict[str, dict] = {
    "block_size": {
        "default_values": [32, 64, 128, 256, 512],
        "override_key": "block_size",
        "type": int,
    },
    "attention_layer": {
        "default_values": [4, 10, 14, 20, 24, 27],
        "override_key": "attention_capture_layer",
        "type": int,
    },
    "topk": {
        "default_values": [1, 2, 4, 8, 16],
        "override_key": "smart_recover_k",
        "type": int,
    },
    "wcoherence": {
        "default_values": [0.0, 0.2, 0.4, 0.6, 0.8],
        "override_key": "w_coherence",
        "type": float,
    },
    "tailprotect": {
        "default_values": [0.0, 0.05, 0.1, 0.2, 0.3],
        "override_key": "recent_tail_protect_frac",
        "type": float,
    },
}


def _wilson(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = passes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    model = os.environ.get("EVOKE_MODEL_PATH")
    if not model:
        print("set EVOKE_MODEL_PATH")
        return 1

    dim = os.environ.get("EVOKE_ABL_DIM", "block_size").strip().lower()
    if dim not in _DIM_CONFIGS:
        print(f"unknown EVOKE_ABL_DIM {dim!r}; choose from {list(_DIM_CONFIGS)}")
        return 1
    cfg = _DIM_CONFIGS[dim]
    cast = cfg["type"]
    override_key = cfg["override_key"]
    values_env = os.environ.get("EVOKE_ABL_VALUES")
    if values_env:
        values = [cast(v.strip()) for v in values_env.split(",")]
    else:
        values = cfg["default_values"]
    base_strategy = os.environ.get("EVOKE_ABL_BASE", "evoke_attention")
    if base_strategy not in STRATEGIES:
        print(f"unknown base strategy {base_strategy!r}")
        return 1
    base_overrides = dict(STRATEGIES[base_strategy])
    budget = int(os.environ.get("EVOKE_BUDGETS", "1024").split(",")[0])
    n_paragraphs = int(os.environ.get("EVOKE_NIAH_PARAGRAPHS", "40"))
    seed = int(os.environ.get("EVOKE_NIAH_SEED", "0"))

    needles = NEEDLES
    if os.environ.get("EVOKE_NIAH_NEEDLES"):
        wanted = {x.strip() for x in os.environ["EVOKE_NIAH_NEEDLES"].split(",")}
        needles = [n for n in NEEDLES if n["id"] in wanted]
    depths = DEFAULT_DEPTHS
    if os.environ.get("EVOKE_NIAH_DEPTHS"):
        depths = [int(d) for d in os.environ["EVOKE_NIAH_DEPTHS"].split(",")]
    out_json = os.environ.get("EVOKE_ABL_JSON")

    engine = LlamaCppEngine(model, n_ctx=16384, n_gpu_layers=-1, verbose=False)
    print(f"ablation | model={Path(model).stem} dim={dim} base={base_strategy}")
    print(
        f"values={values} budget={budget} needles={[n['id'] for n in needles]} "
        f"depths={depths}"
    )
    print(
        f"{'value':<12}{'needle':<10}{'depth':<7}{'probe':<7}{'evict':<7}{'recov':<7}"
    )
    print("-" * 60)

    cells: list[AblationCell] = []
    try:
        for value in values:
            overrides = dict(base_overrides)
            overrides[override_key] = value
            for needle in needles:
                for depth in depths:
                    try:
                        r = run_cell(
                            engine,
                            needle,
                            depth,
                            base_strategy,
                            overrides,
                            budget,
                            n_paragraphs,
                            seed,
                        )
                        mark = "PASS" if r.probe_ok else "fail"
                        print(
                            f"{str(value):<12}{needle['id']:<10}{depth:<7}"
                            f"{mark:<7}{r.evictions:<7}{r.recoveries:<7}"
                        )
                        cells.append(
                            AblationCell(
                                dim=dim,
                                value=value,
                                needle_id=needle["id"],
                                depth=depth,
                                strategy=base_strategy,
                                budget=budget,
                                probe_ok=r.probe_ok,
                                evictions=r.evictions,
                                recoveries=r.recoveries,
                                active_tokens=r.active_tokens,
                                elapsed=r.elapsed,
                                answer=r.answer,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"{str(value):<12}{needle['id']:<10}{depth:<7}ERROR: {exc}"
                        )
            print("-" * 60)
    finally:
        engine.close()

    print()
    print(f"Aggregate per {dim} value")
    print(f"{'value':<12}{'pass_rate':<11}{'95% CI':<22}{'n_cells':<10}")
    print("-" * 60)
    by_value: dict[object, tuple[int, int]] = {}
    for c in cells:
        passes, total = by_value.get(c.value, (0, 0))
        by_value[c.value] = (passes + int(c.probe_ok), total + 1)
    for value in values:
        passes, total = by_value.get(value, (0, 0))
        rate = passes / total if total > 0 else 0.0
        lo, hi = _wilson(passes, total)
        print(
            f"{str(value):<12}{rate:<11.2%}[{lo:.2%}, {hi:.2%}]      {passes}/{total}"
        )

    if out_json:
        Path(out_json).write_text(
            json.dumps([asdict(c) for c in cells], indent=2),
            encoding="utf-8",
        )
        print(f"\nresults JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
