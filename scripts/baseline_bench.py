"""Head-to-head benchmark for EVOKE vs baseline policies.

Drives an N-turn planted-fact session (default 14) against the server under
three policy configurations and prints a comparison table. Restarts the
remote server between runs via SSH so each policy starts fresh.

Required env: EVOKE_SSH_HOST (e.g. user@gpu-host), EVOKE_LIB_PATH (path
to llama.dll on the remote GPU host), EVOKE_MODEL_PATH (path to the GGUF
on the remote GPU host), EVOKE_SERVER (http://host:port the bench drives).
Optional: EVOKE_BUDGET (default 1024), EVOKE_N_CTX (default 16384),
EVOKE_REMOTE_DIR (default C:\\projects\\unlearn),
EVOKE_BENCH_TURNS (default 14: total user turns = 1 fact + (TURNS-2) fillers + 1 probe),
EVOKE_BENCH_OUT (optional JSON output path with per-policy timing + state).
"""

from __future__ import annotations

import base64
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
_FILLER_SEEDS = [
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
    "Describe what a futures or promises pattern accomplishes.",
    "Explain how a write-ahead log preserves crash consistency.",
    "Describe how a Bloom filter trades memory for false-positive rate.",
    "Explain what a SAT solver does at a high level.",
    "Describe the actor concurrency model in one paragraph.",
    "Explain how DHT-style consistent hashing distributes keys.",
    "Describe what a vector clock represents in distributed systems.",
    "Explain the difference between optimistic and pessimistic locking.",
    "Describe what an LSM tree is and where it is used.",
    "Explain how copy-on-write filesystems handle snapshots.",
    "Describe the difference between MVCC and 2PL.",
    "Explain what a circuit breaker pattern protects against.",
    "Describe how reservoir sampling picks a uniform sample.",
    "Explain how rate limiting via leaky bucket differs from token bucket.",
]


def _fillers_for(turns: int) -> list[str]:
    n_fillers = max(0, turns - 2)
    base = _FILLER_SEEDS
    if n_fillers <= len(base):
        return base[:n_fillers]
    out = list(base)
    while len(out) < n_fillers:
        idx = len(out)
        cycle = idx // len(base)
        out.append(f"Variant {cycle + 1}. {base[idx % len(base)]}")
    return out


TURNS = int(os.environ.get("EVOKE_BENCH_TURNS", "14"))
OUT_JSON = os.environ.get("EVOKE_BENCH_OUT")
FILLERS = _fillers_for(TURNS)


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


def _ssh(command: str) -> str:
    # Windows OpenSSH defaults to cmd.exe as the login shell; that means
    # `powershell -Command "<...|...>"` gets re-parsed by cmd, which
    # interprets PowerShell pipes (|) as cmd pipes and breaks the command
    # mid-stream. Using -EncodedCommand with a base64 UTF-16LE payload
    # ships the whole script as one opaque arg that cmd can't reinterpret.
    # 255 exit codes still happen from PowerShell's NativeCommandError
    # behaviour on stderr writes — tolerate them; downstream health poll
    # catches real failures.
    encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    res = subprocess.run(
        ["ssh", SSH_HOST, "powershell", "-NoProfile", "-EncodedCommand", encoded],
        check=False,
        capture_output=True,
    )
    return res.stdout.decode("utf-8", errors="replace")


def _ssh_bg(command: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ["ssh", SSH_HOST, "powershell", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _port_is_free() -> bool:
    # Probe the remote port directly via netcat-like behaviour. Returns
    # True if no process is listening or accepting. Used after kill to
    # confirm the OS has released the listener before we launch the
    # replacement. /health-fail polling alone is unreliable because the
    # old uvicorn can be unresponsive but still hold the port.
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=1) as resp:
            resp.read()
        return False  # something answered = port still owned
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        return True


def restart_server(policy: str) -> subprocess.Popen[bytes]:
    # Two-stage kill: first taskkill /F /IM python.exe (most reliable on
    # Windows — it walks the process tree and kills uv-spawned children
    # the PowerShell Stop-Process path sometimes misses), then a second
    # pass via Get-NetTCPConnection to catch anything still bound to the
    # port (a pythonw or other variant the IM match doesn't hit).
    # Wait a couple of seconds for the OS to actually release the listener
    # — Windows TIME_WAIT on the listening socket itself is fast but the
    # process-tree teardown can take a moment.
    kill_out = _ssh(
        # Stop-Process by name covers python.exe / pythonw.exe / uv.exe.
        # Get-NetTCPConnection catches any other process still bound to
        # port 8000 (rare, but cheap to do). The before/after count is
        # printed so flaky runs leave an audit trail of how many processes
        # the kill actually terminated.
        "$before = @(Get-Process python -ErrorAction SilentlyContinue).Count;"
        "Get-Process -Name python,pythonw,uv -ErrorAction SilentlyContinue "
        "| Stop-Process -Force -ErrorAction SilentlyContinue;"
        "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue "
        "| Where-Object {$_.State -eq 'Listen'} "
        "| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force "
        "-ErrorAction SilentlyContinue };"
        "Start-Sleep -Seconds 3;"
        "$after = @(Get-Process python -ErrorAction SilentlyContinue).Count;"
        'Write-Output ("kill: python before=" + $before + " after=" + $after)'
    )
    if kill_out.strip():
        for line in kill_out.strip().splitlines():
            print(f"  {line}")
    # Wait for the port to be free. Up to 30s — taskkill is usually
    # immediate, but if a CUDA driver shutdown holds the process briefly
    # we wait it out.
    deadline = time.time() + 30
    while time.time() < deadline:
        if _port_is_free():
            break
        time.sleep(1)
    # Belt + braces: even after the port reports free, give the OS one
    # more second to fully release. Without this we sometimes see EADDRINUSE
    # on the new uvicorn bind.
    time.sleep(1)
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
    # Multi-signal scorer pass-through. If the local bench has EVOKE_W_ATTENTION
    # set (and the policy is evoke), forward it so the remote server constructs
    # the AttentionScorer. Default 0 preserves pre-multi-signal behavior.
    extra_env = ""
    w_attn = os.environ.get("EVOKE_W_ATTENTION")
    if w_attn and policy == "evoke":
        extra_env += f"$env:EVOKE_W_ATTENTION='{w_attn}'; "
        attn_layer = os.environ.get("EVOKE_ATTN_LAYER")
        if attn_layer:
            extra_env += f"$env:EVOKE_ATTN_LAYER='{attn_layer}'; "
    ram_budget = os.environ.get("EVOKE_KV_RESTORE_RAM_BUDGET_BYTES")
    if ram_budget and policy == "evoke":
        extra_env += f"$env:EVOKE_KV_RESTORE_RAM_BUDGET_BYTES='{ram_budget}'; "
    spill_path = os.environ.get("EVOKE_KV_RESTORE_SPILL_PATH")
    if spill_path and policy == "evoke":
        extra_env += f"$env:EVOKE_KV_RESTORE_SPILL_PATH='{spill_path}'; "
    n_rs_seq = os.environ.get("EVOKE_N_RS_SEQ")
    if n_rs_seq:
        extra_env += f"$env:EVOKE_N_RS_SEQ='{n_rs_seq}'; "
    suppress_thinking_strip = os.environ.get("EVOKE_SUPPRESS_THINKING_STRIP")
    if suppress_thinking_strip:
        extra_env += f"$env:EVOKE_SUPPRESS_THINKING_STRIP='{suppress_thinking_strip}'; "
    # Smart-recovery knobs (forwarded only for the evoke policy; truncate and
    # no_eviction have no recovery step). Setting min_similarity > 0 plus
    # use_retrieval_embeddings=1 closes the recover-then-re-evict thrash that
    # the session-length sweep exposed at T=28: bge-small widens the cosine
    # band so the floor blocks weak recoveries that would just get evicted
    # again on the next turn.
    sr_min_sim = os.environ.get("EVOKE_SMART_RECOVER_MIN_SIMILARITY")
    if sr_min_sim and policy == "evoke":
        extra_env += f"$env:EVOKE_SMART_RECOVER_MIN_SIMILARITY='{sr_min_sim}'; "
    sr_k = os.environ.get("EVOKE_SMART_RECOVER_K")
    if sr_k and policy == "evoke":
        extra_env += f"$env:EVOKE_SMART_RECOVER_K='{sr_k}'; "
    use_retrieval = os.environ.get("EVOKE_USE_RETRIEVAL_EMBEDDINGS")
    if use_retrieval and policy == "evoke":
        extra_env += f"$env:EVOKE_USE_RETRIEVAL_EMBEDDINGS='{use_retrieval}'; "
    # Recovery-aware eviction knobs (decision-recovery-aware-eviction).
    w_recovery = os.environ.get("EVOKE_W_RECOVERY")
    if w_recovery and policy == "evoke":
        extra_env += f"$env:EVOKE_W_RECOVERY='{w_recovery}'; "
    recovery_decay = os.environ.get("EVOKE_RECOVERY_DECAY")
    if recovery_decay and policy == "evoke":
        extra_env += f"$env:EVOKE_RECOVERY_DECAY='{recovery_decay}'; "
    recovery_strength_init = os.environ.get("EVOKE_RECOVERY_STRENGTH_INIT")
    if recovery_strength_init and policy == "evoke":
        extra_env += f"$env:EVOKE_RECOVERY_STRENGTH_INIT='{recovery_strength_init}'; "
    launch = (
        f"cd {REMOTE_DIR}; "
        f"$env:LLAMA_CPP_LIB='{LIB}'; "
        f"$env:EVOKE_MODEL_PATH='{MODEL}'; "
        f"$env:EVOKE_HOST='0.0.0.0'; $env:EVOKE_PORT='8000'; "
        f"$env:EVOKE_N_CTX='{N_CTX}'; {budget_line}{drift_line}{extra_env}"
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

    print(f"\n## Baseline comparison (turns={TURNS})\n")
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

    if OUT_JSON:
        payload = {
            "turns": TURNS,
            "budget": int(BUDGET),
            "n_ctx": int(N_CTX),
            "model": MODEL_NAME,
            "results": [
                {
                    "policy": r.policy,
                    "probe_ok": r.probe_ok,
                    "evictions": r.evictions,
                    "recoveries": r.recoveries,
                    "active_tokens": r.active_tokens,
                    "elapsed_s": r.elapsed,
                    "answer": r.answer,
                    "error": r.error,
                }
                for r in results
            ],
        }
        Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT_JSON).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nresults JSON: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
