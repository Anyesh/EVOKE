"""Fetch, filter, and save real agent trajectories for the EVOKE benchmark.

Pulls rows from the HuggingFace datasets-server rows API (plain HTTP, no
auth, no extra deps: the same pattern already used in
/mnt/data/cognitive-cache/benchmark/import_swebench.py), normalizes each row
with trajectory_normalizers.normalize, and keeps only rows that fit the eval host's
usable context window and have enough revisit steps to be worth scoring.

SWE-agent-trajectories is the primary source: at these filter thresholds it
gets roughly 10% survival on real data. SWE-rebench-openhands-trajectories
runs 31-64K tokens per trajectory, past the eval host's usable context window, so
almost nothing survives length filtering as-is; it's fetched and reported
separately as a stretch source that needs a windowing function (truncate to
a suffix of N tool calls) neither this script nor trajectory_bench.py
implements yet.

Usage:
  uv run python -m scripts.fetch_trajectories --dataset swe_agent --pages 30 --out data/trajectories/swe_agent
  uv run python -m scripts.fetch_trajectories --dataset openhands --pages 5 --out data/trajectories/openhands
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))

from trajectory_normalizers import classify, normalize  # noqa: E402
from trajectory_bench import find_revisit_steps  # noqa: E402

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASETS = {
    "swe_agent": "nebius/SWE-agent-trajectories",
    "openhands": "nebius/SWE-rebench-openhands-trajectories",
}
MIN_TOKENS = 8_000
MAX_TOKENS = 24_000
MIN_REVISITS = 3

# Rough chars-per-token estimate for filtering only; the actual bench run
# tokenizes with the real model tokenizer. Good enough to separate an 8K
# trajectory from a 40K one, not precise enough to trust near the boundary.
CHARS_PER_TOKEN = 4


def _fetch_rows(hf_name: str, offset: int, length: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "dataset": hf_name,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    req = urllib.request.Request(
        f"{ROWS_URL}?{params}", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _estimate_tokens(traj) -> int:
    chars = len(traj.system_prompt) + sum(len(s.text) for s in traj.steps)
    return chars // CHARS_PER_TOKEN


def _tool_call_count(traj) -> int:
    return sum(1 for s in traj.steps if s.tool_call is not None)


def run(dataset_key: str, pages: int, out_dir: str, page_size: int = 100) -> None:
    hf_name = DATASETS[dataset_key]
    os.makedirs(out_dir, exist_ok=True)

    total_seen = 0
    total_kept = 0
    offset = 0
    for page in range(pages):
        batch = _fetch_rows(hf_name, offset, page_size)
        rows = batch.get("rows", [])
        if not rows:
            break
        for entry in rows:
            raw = entry["row"]
            total_seen += 1
            instance_id = raw.get("instance_id", f"{dataset_key}-{offset}-{total_seen}")
            traj = normalize(raw, instance_id)
            tokens = _estimate_tokens(traj)
            if not (MIN_TOKENS <= tokens <= MAX_TOKENS):
                continue
            revisits = find_revisit_steps(traj)
            if len(revisits) < MIN_REVISITS:
                continue
            if _tool_call_count(traj) < 15:
                continue
            total_kept += 1
            # instance_id + model_name is not unique: this dataset stores
            # multiple rollout attempts per (instance, model) pair (seen up
            # to 6x for the same key on real data), so disambiguate on
            # content instead of trusting those fields to be a primary key.
            content_hash = hashlib.sha1(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest()[:8]
            model_tag = str(raw.get("model_name", "")).replace("/", "__")
            safe_id = str(instance_id).replace("/", "__")
            if model_tag:
                safe_id = f"{safe_id}__{model_tag}"
            safe_id = f"{safe_id}__{content_hash}"
            out_path = os.path.join(out_dir, f"{dataset_key}_{safe_id}.json")
            with open(out_path, "w") as f:
                json.dump(raw, f)
        offset += len(rows)
        print(
            f"page {page + 1}/{pages}: seen {total_seen}, kept {total_kept} "
            f"({100 * total_kept / max(total_seen, 1):.1f}%)",
            flush=True,
        )

    print(f"\n{dataset_key}: {total_kept}/{total_seen} trajectories saved to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--pages", type=int, default=20, help="pages of 100 rows each")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.dataset, args.pages, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
