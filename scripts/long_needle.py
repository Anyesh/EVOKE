"""Retrieval at long absolute positions with no eviction and no cache shifts.

One request per case: a unique passphrase is planted at a given depth in filler
made of this repository's sources, and the question comes last. Cases past the
model's trained 40960 positions are where position scaling should matter, so the
same cases run against a server with and without YaRN.

  python scripts/long_needle.py --base-url http://127.0.0.1:8021 --arm yarn2 \
      --out results/gap_fill/needle_yarn2.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
CHARS_PER_TOKEN = 3.4


def _filler(chars: int, seed: int) -> str:
    files = sorted((REPO / "src" / "evoke").glob("*.py"))
    text = "\n".join(p.read_text() for p in files)
    lines = text.splitlines()
    rng = random.Random(seed)
    out: list[str] = []
    size = 0
    while size < chars:
        line = lines[rng.randrange(len(lines))]
        out.append(f"{len(out):06d} {line}")
        size += len(out[-1]) + 1
    return "\n".join(out)


def _case(total_tokens: int, depth: float, seed: int) -> tuple[str, str]:
    code = f"{random.Random(seed).randrange(10**6):06d}"
    needle = f"NOTE: the vault passphrase is amber-falcon-{code}."
    body = _filler(int(total_tokens * CHARS_PER_TOKEN), seed)
    cut = int(len(body) * depth)
    cut = body.rfind("\n", 0, cut) + 1
    text = body[:cut] + needle + "\n" + body[cut:]
    return text, f"amber-falcon-{code}"


def run(args: argparse.Namespace) -> None:
    rows = []
    headers = {"X-Evoke-Session": "needle/{}"}
    with httpx.Client(base_url=args.base_url, timeout=1800) as client:
        for total in args.lengths:
            for depth in args.depths:
                seed = total * 1000 + int(depth * 100)
                text, answer = _case(total, depth, seed)
                messages = [
                    {"role": "system", "content": "Answer with the passphrase only."},
                    {
                        "role": "user",
                        "content": text
                        + "\n\nWhat is the vault passphrase? Reply with it only.",
                    },
                ]
                t0 = time.monotonic()
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "evoke", "messages": messages, "max_tokens": 24},
                    headers={**headers, "X-Evoke-Session": f"needle/{total}/{depth}"},
                )
                wall = time.monotonic() - t0
                if resp.status_code != 200:
                    rows.append(
                        {"total": total, "depth": depth, "status": resp.status_code}
                    )
                    print(total, depth, "HTTP", resp.status_code, flush=True)
                    continue
                body = resp.json()
                reply = body["choices"][0]["message"]["content"] or ""
                prompt_tokens = body["usage"]["prompt_tokens"]
                rows.append(
                    {
                        "total": total,
                        "depth": depth,
                        "prompt_tokens": prompt_tokens,
                        "needle_position": int(prompt_tokens * depth),
                        "ok": answer in reply,
                        "reply": reply,
                        "wall_s": round(wall, 1),
                    }
                )
                print(
                    f"tokens={prompt_tokens} depth={depth} ok={answer in reply} "
                    f"wall={wall:.0f}s reply={reply[:30]!r}",
                    flush=True,
                )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"arm": args.arm, "cases": rows}, indent=2))
    scored = [r for r in rows if "ok" in r]
    print(f"{args.arm}: {sum(r['ok'] for r in scored)}/{len(scored)} retrieved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--arm", default="run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[30000, 60000, 78000])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.05, 0.5, 0.95])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
