"""Measure a Loom-configured EVOKE instance at Loom's scale: an ~18.8K-token
system prompt, one session grown past the residency budget with tool outputs,
and a second session to force engine state swaps. Records per-turn wall time
and usage.evoke, VRAM, and the server process's host memory.

Run on the GPU host next to the server:
  python scripts/measure_loom_instance.py --base-url http://127.0.0.1:8020 \
      --admin-key TOKEN --out results/loom_instance.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
MODEL = "evoke-qwen3-8b"
KEY_ID = "loom@measure@usr_0"


def _corpus() -> str:
    return "\n\n".join(
        p.read_text() for p in sorted((REPO / "src" / "evoke").glob("*.py"))
    )


def _vram_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def _server_rss_mib(port: int) -> int | None:
    if sys.platform != "win32":
        return None
    script = (
        f"(Get-Process -Id (Get-NetTCPConnection -State Listen -LocalPort {port})"
        ".OwningProcess).WorkingSet64"
    )
    out = subprocess.run(
        ["powershell", "-Command", script], capture_output=True, text=True
    )
    return int(out.stdout.strip()) // (1024 * 1024) if out.stdout.strip() else None


class Session:
    def __init__(
        self, client: httpx.Client, base: str, token: str, sid: str, system: str
    ):
        self.client, self.base, self.sid = client, base, sid
        self.headers = {"Authorization": f"Bearer {token}", "X-Evoke-Session": sid}
        self.messages = [{"role": "system", "content": system}]

    def turn(self, user: str, label: str) -> dict:
        self.messages.append({"role": "user", "content": user})
        start = time.monotonic()
        resp = self.client.post(
            f"{self.base}/v1/chat/completions",
            json={"model": MODEL, "messages": self.messages, "max_tokens": 64},
            headers=self.headers,
        )
        wall_ms = round((time.monotonic() - start) * 1000)
        resp.raise_for_status()
        body = resp.json()
        self.messages.append(
            {
                "role": "assistant",
                "content": body["choices"][0]["message"]["content"] or "",
            }
        )
        usage = body["usage"]
        row = {
            "label": label,
            "session": self.sid,
            "wall_ms": wall_ms,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            **{k: v for k, v in usage["evoke"].items() if k != "session"},
        }
        print(
            f"{label:<14} wall={wall_ms:>6}ms prompt={row['prompt_tokens']:>6} "
            f"decoded={row['prompt_tokens_decoded']:>6} peak={row['peak_resident_tokens']:>6} "
            f"end={row['resident_tokens_end']:>6} evicted={row['blocks_evicted']:>3} "
            f"recovered={row['blocks_recovered']:>3}",
            flush=True,
        )
        return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--admin-key", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--system-chars", type=int, default=66000)
    parser.add_argument("--tool-chars", type=int, default=14000)
    parser.add_argument("--grow-turns", type=int, default=4)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    port = httpx.URL(base).port or 80
    corpus = _corpus()
    system = (
        "You are Loom's coding agent. Reference material follows.\n\n"
        + corpus[: args.system_chars]
    )
    tool_texts = [
        corpus[
            args.system_chars + i * args.tool_chars : args.system_chars
            + (i + 1) * args.tool_chars
        ]
        for i in range(args.grow_turns + 1)
    ]

    admin = {"Authorization": f"Bearer {args.admin_key}"}
    rows: list[dict] = []
    with httpx.Client(timeout=1800) as client:
        vram_before = _vram_mib()
        token = client.post(
            f"{base}/admin/keys", json={"key_id": KEY_ID}, headers=admin
        ).json()["token"]
        try:
            a = Session(client, base, token, "measure/a", system)
            b = Session(client, base, token, "measure/b", system)
            rows.append(
                a.turn("Summarise the reference material in one line.", "A1 cold")
            )
            rows.append(a.turn("Say OK.", "A2 small"))
            for i in range(args.grow_turns):
                rows.append(
                    a.turn(f"Tool output:\n{tool_texts[i]}\nSay OK.", f"A{3 + i} +tool")
                )
            rows.append(a.turn("Say OK again.", f"A{3 + args.grow_turns} resend"))
            rows.append(
                b.turn("Summarise the reference material in one line.", "B1 cold")
            )
            rows.append(a.turn("Say OK once more.", f"A{4 + args.grow_turns} swap-in"))
            rows.append(b.turn("Say OK.", "B2 swap-in"))
            rss = _server_rss_mib(port)
            vram_after = _vram_mib()
        finally:
            client.delete(f"{base}/admin/keys/{KEY_ID}", headers=admin)

    result = {
        "vram_used_mib_before": vram_before,
        "vram_used_mib_after": vram_after,
        "server_rss_mib": rss,
        "turns": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"vram {vram_before} -> {vram_after} MiB, server rss {rss} MiB, wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
