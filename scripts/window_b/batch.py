import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from schedule import new_state, next_job, record, recover, start

RUN_TIMEOUT = 2 * 3600 + 900
HEALTH_TIMEOUT = 600

stopping = False


def on_signal(signum, frame):
    global stopping
    stopping = True


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else new_state()


def save(path, state):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)


def parse_stop(text):
    h, m = map(int, text.split(":"))
    stop = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if stop <= datetime.now():
        stop += timedelta(days=1)
    return stop.timestamp()


def sh(template, **kw):
    return subprocess.run(template.format(**kw), shell=True, check=True, timeout=300)


def wait_healthy(url):
    deadline = time.time() + HEALTH_TIMEOUT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(5)
    return False


def run_arm(args, task, arm):
    result_path = Path(args.runs) / f"{task}__{arm}.json"
    result_path.unlink(missing_ok=True)
    started = time.time()
    sh(args.stop_cmd)
    sh(args.start_cmd, arm=arm)
    if not wait_healthy(args.health_url):
        return {"infra_failure": True, "resolved": None, "seconds": 0, "why": "server not healthy"}
    try:
        proc = subprocess.run(
            args.run_cmd.format(
                task=task, arm=arm, base_url=args.base_url, result=result_path
            ),
            shell=True,
            timeout=RUN_TIMEOUT,
        )
        ok = proc.returncode == 0 and result_path.exists()
    except subprocess.TimeoutExpired:
        ok = False
    if not ok:
        return {
            "infra_failure": True,
            "resolved": None,
            "seconds": time.time() - started,
            "why": "runner failed or timed out",
        }
    result = json.loads(result_path.read_text())
    result.setdefault("infra_failure", False)
    result["seconds"] = time.time() - started
    result["night"] = datetime.fromtimestamp(started).strftime("%Y-%m-%d")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perm", default="results/window_b/permutation.json")
    parser.add_argument("--state", default="results/window_b/state.json")
    parser.add_argument("--runs", default="results/window_b/runs")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--stop-at", required=True, help="HH:MM local time")
    parser.add_argument("--start-cmd", required=True, help="uses {arm}")
    parser.add_argument("--stop-cmd", required=True)
    parser.add_argument("--run-cmd", required=True, help="uses {task} {arm} {base_url} {result}")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--health-url", required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    Path(args.runs).mkdir(parents=True, exist_ok=True)
    perm = json.loads(Path(args.perm).read_text())["order"]
    stop_at = parse_stop(args.stop_at)
    state = load(args.state)
    recover(state)
    save(args.state, state)

    while not stopping:
        job = next_job(state, perm, time.time(), stop_at, limit=args.limit)
        if job is None:
            break
        task, arm = job
        start(state, task, arm)
        save(args.state, state)
        result = run_arm(args, task, arm)
        if stopping and result.get("infra_failure"):
            recover(state)
        else:
            record(state, task, arm, result)
        save(args.state, state)
        print(f"{task} {arm} {result}", flush=True)

    sh(args.stop_cmd)
    print("batch stopped", flush=True)


if __name__ == "__main__":
    sys.exit(main())
