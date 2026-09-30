"""Grow one session past the KV cell count through a running EVOKE server and
record, per turn, the usage.evoke residency numbers, wall time and the greedy
reply. The same script runs against the arms compared in contract window A/B:
a plain server with cells >= conversation (contiguous reference), the current
restore-everything server, and the budget-aware server with chunked prefill.
Replies are compared across arms with --compare.

  python scripts/measure_gap_fill.py --base-url http://127.0.0.1:8021 \
      --arm budget_aware --turns 40 --out results/gap_fill/budget_aware.json
  python scripts/measure_gap_fill.py --compare results/gap_fill/reference.json \
      results/gap_fill/budget_aware.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
FACT = "The vault passphrase is amber-falcon-4417."
SYSTEM = (
    "You are a coding agent. Answer in one short sentence. "
    f"Remember this for the whole session: {FACT}"
)


def _vram_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(out.stdout.strip().splitlines()[0])


def _chunks(chars: int) -> list[str]:
    text = "\n\n".join(
        p.read_text() for p in sorted((REPO / "src" / "evoke").glob("*.py"))
    )
    return [text[i : i + chars] for i in range(0, len(text) - chars, chars)]


def _turn_user(i: int, chunks: list[str], recall_every: int) -> str:
    body = f"Tool output {i}:\n{chunks[i % len(chunks)]}\n"
    if i and i % recall_every == 0:
        return body + "\nWhat is the vault passphrase? Reply with the passphrase only."
    return body + "\nAcknowledge with the single word: noted."


def run(args: argparse.Namespace) -> None:
    headers = {"X-Evoke-Session": f"gapfill/{args.arm}"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    chunks = _chunks(args.chunk_chars)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    rows: list[dict] = []
    vram_before = _vram_mib()
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=1800) as client:
        for i in range(args.turns):
            messages.append(
                {"role": "user", "content": _turn_user(i, chunks, args.recall_every)}
            )
            t0 = time.monotonic()
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "evoke", "messages": messages, "max_tokens": 24},
            )
            wall = time.monotonic() - t0
            if resp.status_code != 200:
                rows.append({"turn": i, "status": resp.status_code, "body": resp.text})
                break
            body = resp.json()
            reply = body["choices"][0]["message"]["content"] or ""
            messages.append({"role": "assistant", "content": reply})
            rows.append(
                {
                    "turn": i,
                    "wall_s": round(wall, 3),
                    "reply": reply,
                    "recall": bool(i and i % args.recall_every == 0),
                    "recall_ok": "amber-falcon-4417" in reply,
                    "usage": body["usage"],
                    "vram_mib": _vram_mib(),
                }
            )
            print(
                f"turn {i} wall={wall:.1f}s "
                f"peak={body['usage']['evoke']['peak_resident_tokens']} "
                f"end={body['usage']['evoke']['resident_tokens_end']} "
                f"logical={body['usage']['evoke']['logical_tokens']}",
                flush=True,
            )
    out = {
        "arm": args.arm,
        "vram_before_mib": vram_before,
        "turns": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


def compare(reference: str, other: str) -> None:
    ref = json.loads(Path(reference).read_text())["turns"]
    got = json.loads(Path(other).read_text())["turns"]
    same = diff = 0
    for r, g in zip(ref, got):
        if r.get("reply") == g.get("reply"):
            same += 1
        else:
            diff += 1
            print(f"turn {r['turn']}: {r.get('reply')!r} != {g.get('reply')!r}")
    recalls = [t for t in got if t.get("recall")]
    ok = sum(t["recall_ok"] for t in recalls)
    print(f"identical replies {same}, different {diff}; recall {ok}/{len(recalls)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--token", default="")
    parser.add_argument("--arm", default="run")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--chunk-chars", type=int, default=4000)
    parser.add_argument("--recall-every", type=int, default=8)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "OTHER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
