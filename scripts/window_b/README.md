# Control-batch scheduler

`scripts/window_b/` runs a paired comparison of two serving arms over a fixed task list, in batches that can be interrupted and resumed. It does not run tasks itself: it starts a server for the arm, waits for it to answer, calls a runner command you supply, and records the result.

Arms are labelled `E` (EVOKE server) and `C` (a plain llama.cpp server with the client's own compaction). Each task runs under both arms back to back, and the arm order alternates with the task's position so neither arm always goes first.

## Arm conditions (clarifications)

- 2026-10-02, before any control run: thinking is off in both arms and is enforced on the server as well as by the client. Arm E ignores a request-level `chat_template_kwargs` and renders with the server's thinking-off setting. Arm C's server runs with a reasoning budget of 0; a request asking for thinking then gets no reasoning text, and requests that leave the setting unset or ask for thinking off get byte-identical content and the same prompt token counts as without the flag (checked on three prompts at temperature 0). The client also sends thinking off on every request, as a second layer. The arms, task set, grader and endpoints are unchanged.
- 2026-10-02, before any control run: a task refused before any work was done (low disk, runtime or environment unavailable) never counts as an attempt and does not exclude the task; the batch stops and resumes from the same state.

## Task order

```bash
python scripts/window_b/permutation.py --ids results/window_b/instance_ids.txt --out results/window_b/permutation.json
```

The ids are sorted lexicographically and shuffled with `random.Random(20260927)`; the output records the seed, the method and a SHA-256 of the id list, so the order can be regenerated and checked. `instance_ids.txt` holds the 500 SWE-bench Verified instance ids.

## Running a batch

```bash
python scripts/window_b/batch.py \
  --stop-at 08:00 --limit 60 \
  --start-cmd '<command that starts arm {arm}>' \
  --stop-cmd '<command that stops whichever server is running>' \
  --run-cmd '<runner> --instance {task} --arm {arm} --base-url {base_url} --out {result}' \
  --base-url http://HOST:PORT \
  --health-url E=http://HOST:PORT/healthz --health-url C=http://HOST:PORT/health
```

- `--stop-at` is a local wall-clock time (`HH:MM`, the clock and time zone of the machine running the scheduler; a time already past means tomorrow). A task pair is started only if both arms are expected to finish before it, using the median of observed arm durations plus 120 s for a server swap (600 s per arm before any run has finished).
- `--stop-at none` removes the deadline: pairs keep starting until the task list (up to `--limit`) is done or the process is stopped. Use it for a continuous run; resume covers any stop.
- `--limit` caps how many tasks of the permutation are considered.
- The start command is run per arm with `{arm}` filled in; the stop command runs before each start and at the end. Only one server runs at a time.
- The health check polls for up to 600 s; a server that never answers counts as an infrastructure failure.
- The runner must write a JSON file to `{result}` and exit 0 when it did. The fields the scheduler reads are `resolved` (bool) and `infra_failure` (bool, default false); other fields are kept as given. A non-zero exit, a missing file, or a run longer than 2 h 15 min is an infrastructure failure.

## Resume and failure rules

State is written after every run to `results/window_b/state.json` (override with `--state`), and per-run result files go to `results/window_b/runs/` (`--runs`). Starting the same command again resumes where it stopped.

- SIGINT or SIGTERM finish bookkeeping and stop the server; an arm that was running is put back to pending and does not use a retry.
- If a runner call fails and the arm's server no longer answers its health check (the host went off or away), the arm is treated as interrupted instead of an infrastructure failure: it goes back to pending without using its retry and the scheduler exits with status 3. Start it again once the host is back.
- The runner exits 75 (or writes `tempfail: true` in its result file) when it refuses a task before doing any work, for example for low disk, a missing container runtime or a missing environment. The scheduler then puts the arm back to pending without using an attempt, stops the server, logs `tempfail: <reason>` and exits with status 75; fix the cause and rerun the same command.
- A run found as `running` at startup (power loss) is treated the same way and its `interrupted` counter is incremented.
- An infrastructure failure is retried once; a second failure marks the task failed and excludes it from both arms.
- If one arm of a pair finished and the other did not, the other arm runs next and needs time for one arm only.

## Logs

The scheduler prints one line per finished arm (task, arm, result) to standard output; redirect it to a file if you want a record. Server logs are whatever your start command captures (see [OPERATING.md](../../OPERATING.md#logs)).

## Tests

`uv run pytest tests/test_window_b_schedule.py -q` covers the permutation, arm order, the stop-time rule, resume, retries and exclusion.
