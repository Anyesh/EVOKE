"""Head-to-head benchmark for EVOKE vs baseline policies.

Drives the same 14-turn planted-fact session against the server under three
policy configurations and prints a comparison table. Restarts the remote
server between runs via SSH so each policy starts fresh.

Required env: EVOKE_SSH_HOST (e.g. user@gpu-host), EVOKE_LIB_PATH (path
to llama.dll on the remote GPU host), EVOKE_MODEL_PATH (path to the GGUF
on the remote GPU host), EVOKE_SERVER (http://host:port the bench drives).
Optional: EVOKE_BUDGET (default 1024), EVOKE_N_CTX (default 16384),
EVOKE_REMOTE_DIR (default C:\\projects\\unlearn).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _missing(name: str) -> str:
    raise SystemExit(f"set {name} (see module docstring)")


BASE = os.environ.get("EVOKE_SERVER") or _missing("EVOKE_SERVER")
SSH_HOST = os.environ.get("EVOKE_SSH_HOST") or _missing("EVOKE_SSH_HOST")
LIB = os.environ.get("EVOKE_LIB_PATH") or _missing("EVOKE_LIB_PATH")
MODEL = os.environ.get("EVOKE_MODEL_PATH") or _missing("EVOKE_MODEL_PATH")
BUDGET = os.environ.get("EVOKE_BUDGET", "1024")
N_CTX = os.environ.get("EVOKE_N_CTX", "16384")
MODEL_NAME = os.environ.get("EVOKE_MODEL_NAME", "qwen25")
REMOTE_DIR = os.environ.get("EVOKE_REMOTE_DIR", "C:\\projects\\unlearn")
DRIFT_DEBUG = bool(os.environ.get("EVOKE_DEBUG_DRIFT"))
DRIFT_LOG = REMOTE_DIR + "\\evoke_drift.log"

PASSKEY = "4242"
FACT = (
    f"Important system note: my favorite number is {PASSKEY}. "
    "Please remember this for the rest of our conversation."
)
PROBE = (
    "Earlier in this conversation I told you my favorite number. "
    "What was it? Reply with just the number."
)
FILLERS = [
    "Explain the concept of an algorithm in one short paragraph.",
    "Explain the difference between recursion and iteration in one paragraph.",
    "Describe polymorphism in object-oriented programming in one paragraph.",
    "Explain Big-O notation briefly.",
    "Describe what a hash table is and one common use case.",
    "Explain garbage collection in one paragraph.",
    "Describe the publish-subscribe pattern.",
    "Explain what a closure is in programming.",
    "Describe the difference between TCP and UDP.",
    "Explain the model-view-controller pattern briefly.",
    "Describe what a binary search tree is.",
    "Explain how a relational database index works.",
]


@dataclass
class Result:
    policy: str
    probe_ok: bool
    answer: str
    evictions: int
    recoveries: int
    active_tokens: int
    elapsed: float
    error: str | None = None


def _ssh(command: str) -> None:
    # SSH to Windows OpenSSH + powershell occasionally exits 255 on otherwise-
    # successful commands when PowerShell emits to stderr (NativeCommandError),
    # so we tolerate non-zero exit codes here and let the downstream health
    # poll catch real failures.
    subprocess.run(
        ["ssh", SSH_HOST, "powershell", "-Command", command],
        check=False,
        capture_output=True,
    )


def _ssh_bg(command: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ["ssh", SSH_HOST, "powershell", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def restart_server(policy: str) -> subprocess.Popen[bytes]:
    # Kill whatever owns port 8000 — Get-Process by name misses processes uv
    # spawns under varying interpreter names. Get-NetTCPConnection finds the
    # listener regardless of what binary it is.
    _ssh(
        "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue "
        "| Where-Object {$_.State -eq 'Listen'} "
        "| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force "
        "-ErrorAction SilentlyContinue }; "
        "Get-Process python,python3,pythonw -ErrorAction SilentlyContinue "
        "| Stop-Process -Force"
    )
    # Wait for the OS to actually release port 8000 — otherwise the new
    # uvicorn fails to bind and our requests keep hitting the dying server.
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=1) as resp:
                resp.read()
            time.sleep(2)
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            break
    time.sleep(3)
    # no_eviction needs the budget to NOT pin to the tight value otherwise the
    # watermark still trips. Letting EVOKE_BUDGET fall back to n_ctx makes
    # eviction genuinely never fire for that baseline.
    budget_line = "" if policy == "no_eviction" else f"$env:EVOKE_BUDGET='{BUDGET}'; "
    # In drift-debug mode the server writes drift blocks directly to a known
    # file path (via EVOKE_DEBUG_DRIFT_FILE inside Python); we avoid PowerShell
    # `*>>` redirection because it doesn't reliably capture stderr through the
    # `ssh ... powershell -Command` boundary.
    drift_line = (
        f"$env:EVOKE_DEBUG_DRIFT='1'; $env:EVOKE_DEBUG_DRIFT_FILE='{DRIFT_LOG}'; "
        if DRIFT_DEBUG
        else ""
    )
    if DRIFT_DEBUG:
        _ssh(f"if (Test-Path '{DRIFT_LOG}') {{ Remove-Item -Force '{DRIFT_LOG}' }}")
    launch = (
        f"cd {REMOTE_DIR}; "
        f"$env:LLAMA_CPP_LIB='{LIB}'; "
        f"$env:EVOKE_MODEL_PATH='{MODEL}'; "
        f"$env:EVOKE_HOST='0.0.0.0'; $env:EVOKE_PORT='8000'; "
        f"$env:EVOKE_N_CTX='{N_CTX}'; {budget_line}{drift_line}"
        f"$env:EVOKE_POLICY='{policy}'; "
        f"$env:EVOKE_MODEL_NAME='{MODEL_NAME}'; "
        f"uv run python scripts/evoke_serve.py"
    )
    print(f"  launch: {launch}")
    expected_budget = int(N_CTX) if policy == "no_eviction" else int(BUDGET)
    proc = _ssh_bg(launch)
    deadline = time.time() + 240
    last_budget: int | None = None
    poll_count = 0
    while time.time() < deadline:
        poll_count += 1
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as resp:
                h = json.loads(resp.read())
            seen = h.get("budget")
            if seen != last_budget:
                print(f"  poll #{poll_count}: /health budget={seen}")
                last_budget = seen
            if seen == expected_budget:
                return proc
        except (urllib.error.URLError, ConnectionResetError) as err:
            if poll_count == 1 or poll_count % 10 == 0:
                print(
                    f"  poll #{poll_count}: /health not up yet ({type(err).__name__})"
                )
        time.sleep(2)
    raise RuntimeError(
        f"server did not come up with expected budget={expected_budget} "
        f"for policy={policy!r}"
    )


def post(messages: list[dict]) -> dict:
    body = json.dumps(
        {"model": MODEL_NAME, "messages": messages, "max_tokens": 256}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def health() -> dict:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as resp:
        return json.loads(resp.read())


def reset() -> None:
    req = urllib.request.Request(f"{BASE}/admin/reset", method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def fetch_drift_log(policy: str) -> None:
    if not DRIFT_DEBUG:
        return
    # Pull the remote-side server log over SSH (NOT _ssh_bg, since we need
    # the bytes here in the parent), then surface any drift diagnostic blocks.
    res = subprocess.run(
        [
            "ssh",
            SSH_HOST,
            "powershell",
            "-Command",
            f"if (Test-Path '{DRIFT_LOG}') {{ Get-Content '{DRIFT_LOG}' -Raw }}",
        ],
        capture_output=True,
        timeout=60,
    )
    log = res.stdout.decode("utf-8", errors="replace") if res.stdout else ""
    if "EVOKE drift diagnostic" not in log:
        print(f"  drift: no diagnostics captured for policy={policy}")
        return
    print(f"\n--- DRIFT DIAGNOSTICS captured for policy={policy} ---")
    # Slice each diagnostic block out so the print is targeted.
    marker = "=== EVOKE drift diagnostic ==="
    end_marker = "=== /drift diagnostic ==="
    cursor = 0
    block_no = 0
    while True:
        start = log.find(marker, cursor)
        if start == -1:
            break
        end = log.find(end_marker, start)
        if end == -1:
            end = len(log)
        else:
            end += len(end_marker)
        block_no += 1
        print(f"[drift block #{block_no}]")
        print(log[start:end])
        cursor = end


def run_policy(policy: str) -> Result:
    print(f"\n>>> running policy={policy}")
    restart_server(policy)
    reset()
    h = health()
    print(f"  budget={h['budget']} n_ctx={h['n_ctx']}")

    messages: list[dict] = [
        {"role": "system", "content": "You are a concise, helpful assistant."},
        {"role": "user", "content": FACT},
    ]
    t0 = time.perf_counter()
    try:
        # Append the SERVER's actual generated reply each turn, otherwise the
        # next request's templated prompt diverges from our cached state at
        # the assistant turn and the session resets, zeroing the eviction
        # counters we are trying to measure.
        r = post(messages)
        messages.append(
            {
                "role": "assistant",
                "content": r["choices"][0]["message"].get("content") or "",
            }
        )
        h0 = health()
        print(
            f"  fact: active={h0['active_tokens']} ev={h0['total_evictions']}"
            f" rec={h0['total_recoveries']}"
        )
        for i, q in enumerate(FILLERS):
            messages.append({"role": "user", "content": q})
            r = post(messages)
            messages.append(
                {
                    "role": "assistant",
                    "content": r["choices"][0]["message"].get("content") or "",
                }
            )
            h = health()
            print(
                f"  q{i}:   active={h['active_tokens']} ev={h['total_evictions']}"
                f" rec={h['total_recoveries']}"
            )
        messages.append({"role": "user", "content": PROBE})
        r = post(messages)
        answer = r["choices"][0]["message"].get("content") or ""
        h = health()
        elapsed = time.perf_counter() - t0
        fetch_drift_log(policy)
        return Result(
            policy=policy,
            probe_ok=PASSKEY in answer,
            answer=answer.strip().replace("\n", " ")[:80],
            evictions=h["total_evictions"],
            recoveries=h["total_recoveries"],
            active_tokens=h["active_tokens"],
            elapsed=elapsed,
        )
    except urllib.error.HTTPError as e:
        fetch_drift_log(policy)
        return Result(
            policy=policy,
            probe_ok=False,
            answer="",
            evictions=0,
            recoveries=0,
            active_tokens=0,
            elapsed=time.perf_counter() - t0,
            error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:120]}",
        )


def main() -> int:
    policies_env = os.environ.get("EVOKE_BENCH_POLICIES")
    policies = (
        [p.strip() for p in policies_env.split(",") if p.strip()]
        if policies_env
        else ["no_eviction", "truncate", "evoke"]
    )
    results: list[Result] = []
    for p in policies:
        try:
            results.append(run_policy(p))
        except Exception as exc:  # noqa: BLE001
            results.append(
                Result(
                    policy=p,
                    probe_ok=False,
                    answer="",
                    evictions=0,
                    recoveries=0,
                    active_tokens=0,
                    elapsed=0.0,
                    error=str(exc),
                )
            )

    print("\n## Baseline comparison\n")
    print(f"| policy        | probe | evictions | recoveries | active | elapsed |")
    print(f"|---------------|-------|----------:|-----------:|-------:|--------:|")
    for r in results:
        mark = "PASS" if r.probe_ok else ("FAIL" if not r.error else "ERR")
        suffix = "" if not r.error else f" ({r.error})"
        print(
            f"| {r.policy:<13} | {mark:<5} | {r.evictions:>9} | "
            f"{r.recoveries:>10} | {r.active_tokens:>6} | "
            f"{r.elapsed:>6.1f}s |{suffix}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
