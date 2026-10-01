# Operating the EVOKE server

`scripts/evoke_serve.py` runs EVOKE as an OpenAI-compatible chat completions server. This guide covers installing and building it, every setting it reads, running and stopping it, checking on it, and the problems people usually hit. For what EVOKE does, see the [README](README.md).

## Install and build

You need Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), a GGUF model, and the EVOKE-forked llama.cpp built as a shared library with CUDA.

```bash
git clone https://github.com/Anyesh/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
```

Older llama.cpp trees spell the CUDA flag `-DLLAMA_CUDA=ON`. The result is `build/bin/libllama.so` (Linux) or `build/bin/llama.dll` (Windows), with the ggml libraries next to it. Build with the same options you would use for stock llama.cpp (CUDA, shared libraries, native architecture).

```bash
git clone https://github.com/Anyesh/EVOKE.git && cd EVOKE
uv sync --extra server --extra dev
```

`--extra dev` adds pytest; leave it out on a machine that only serves. Point the server at the fork with `LLAMA_CPP_LIB=/path/to/libllama.so`. Without it the server runs on the stock llama-cpp-python wheel, where the KV block primitives are missing: `/healthz` reports `kv_block_primitives: false` and `kv_restore` recovery fails at the first request that needs it.

The engine measures the library's `llama_context_params` layout when it loads and refuses a library whose layout it does not recognise, so a build that does not match fails at startup instead of silently ignoring settings.

### YaRN

Static YaRN scaling is opt-in through `EVOKE_YARN_FACTOR` and `EVOKE_YARN_ORIG_CTX` (for example factor 2 over an original context of 40960). Use it only with a fork build that includes the K-shift fix (fork commit `1af01e534` or later); earlier builds rescale cached keys under YaRN and produce garbled output once positions are shifted. If you have no reason to scale, leave YaRN unset.

## Start, stop, status

Only `EVOKE_MODEL_PATH` is required:

```bash
LLAMA_CPP_LIB=/path/to/libllama.so \
EVOKE_MODEL_PATH=/path/to/model.gguf \
EVOKE_BUDGET=8192 \
uv run python scripts/evoke_serve.py
```

`uv run python scripts/evoke_serve.py --help` prints the short variable list. The server binds `127.0.0.1:8000` by default; set `EVOKE_HOST` and `EVOKE_PORT` to change that. **`EVOKE_BUDGET` must be set for the eviction policy to take effect** (see below).

To stop it, send SIGINT (Ctrl-C) or SIGTERM to the process. If you run it detached, stop it by process id or by whatever supervises it (systemd, a scheduled task); then confirm the GPU is free (`nvidia-smi`), because an orphaned process keeps its VRAM.

To check on it:

```bash
curl -s http://127.0.0.1:8000/healthz
```

A healthy answer has `"status": "ok"` and `"model_loaded": true`. Note that `/healthz` answers before any chat request has been made and, after an idle unload, reports `model_loaded: false` until the next chat request.

## Configuration

Settings are environment variables. There is no config file.

### Which variables take effect

- With the default policy (`EVOKE_POLICY=evoke`) and `EVOKE_BUDGET` **unset**, every policy variable below (recovery, position, window, scorer weights and so on) is ignored, and sessions use built-in defaults: a budget of 75% of `EVOKE_N_CTX` (at least 2048), 128-token blocks, watermarks 0.92 and 0.70, `kv_restore` recovery, compact positions. Set `EVOKE_BUDGET` to use any of them.
- `EVOKE_POLICY=truncate` (recency only) and `no_eviction` build their own fixed configuration and honour only `EVOKE_BUDGET` and `EVOKE_SUPPRESS_THINKING_STRIP`.
- A few variables are enabled by any non-empty value, so `0` or `false` still turns them on: `EVOKE_SUPPRESS_THINKING_STRIP`, `EVOKE_USE_RETRIEVAL_EMBEDDINGS`, `EVOKE_LLAMA_LOG`, and the `EVOKE_DEBUG_*` variables. Unset them to disable. The ones that say "`== 1`" below need exactly `1`.

### Server and model

| Variable | Default | Meaning |
|---|---|---|
| `EVOKE_MODEL_PATH` | none, required | GGUF file to load. |
| `EVOKE_MODEL_NAME` | file name without extension | Model id returned by `/healthz`, `/v1/models` and `/models`. |
| `EVOKE_MODEL_DIR` | directory of `EVOKE_MODEL_PATH` | Directory scanned for `*.gguf` (names containing `mmproj` are skipped) by `/models` and `/models/load`. |
| `EVOKE_HOST` | `127.0.0.1` | Bind address. |
| `EVOKE_PORT` | `8000` | Port. |
| `EVOKE_N_CTX` | `32768` | KV cells in the engine. Also the logical window unless `EVOKE_LOGICAL_WINDOW` is set. |
| `EVOKE_KV_QUANT` | empty | KV cache type for K and V: `f32`, `f16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`. Empty, `f16` or `none` keep the engine default. |
| `EVOKE_FLASH_ATTN` | unset | `-1` auto, `0` off, `1` on. Leave unset unless you are testing; forcing it on breaks Qwen3 models with QK-norm. |
| `EVOKE_NO_EMBEDDINGS` | unset | `1` creates the context without embeddings. |
| `EVOKE_N_RS_SEQ` | `0` | Recurrent-state snapshots per token for hybrid (attention plus state-space) models; above 0 allows mid-sequence eviction there. |
| `EVOKE_YARN_FACTOR` | `0` | YaRN rope scale; at least 1 when enabled. |
| `EVOKE_YARN_ORIG_CTX` | `0` | The model's original context for YaRN; must be above 0 when YaRN is on. |
| `EVOKE_YARN_EXT_FACTOR` | `1` | YaRN extrapolation mix, read only when YaRN is on. |
| `EVOKE_ENABLE_THINKING` | unset | `1`/`true` or `0`/`false` forces the chat template's `enable_thinking` flag; anything else leaves the template default. |
| `EVOKE_SUPPRESS_THINKING_STRIP` | unset | Keep `<think>` text in returned content instead of stripping it. |
| `EVOKE_IDLE_TIMEOUT` | unset | Seconds of inactivity before the model is unloaded from VRAM; it reloads on the next chat request. Unset or `0` keeps it resident. |
| `LLAMA_CPP_LIB` | unset | Path to the fork's shared library (see above). |

### Eviction policy

| Variable | Default | Meaning |
|---|---|---|
| `EVOKE_POLICY` | `evoke` | `evoke`, `truncate` or `no_eviction`. Anything else is an error. |
| `EVOKE_BUDGET` | unset | Maximum resident (active) KV tokens. Required for the `evoke` policy to apply. `truncate` defaults to 1024 and `no_eviction` to `EVOKE_N_CTX`. `POST /admin/set_budget` changes it at runtime. |
| `EVOKE_RECOVERY_MODE` | `kv_restore` | `discard`, `breadcrumb` or `kv_restore`. An invalid value fails at the first session, not at startup. |
| `EVOKE_RECOVERY_MATCH` | `identity` | `identity` restores a block by exact token identity at its original position; `similarity` is the older cosine-similarity path. |
| `EVOKE_POSITION_MODE` | `sparse` when match is `identity`, else `compact` | `sparse` leaves holes at the original positions; `compact` re-indexes the survivors. |
| `EVOKE_GAP_FILL_BUDGET_AWARE` | `0` | `1` restores saved blocks only while resident tokens, the new tail and the generation reserve fit the budget. |
| `EVOKE_PREFILL_CHUNK_TOKENS` | `0` | Decode the new tail in chunks of this many tokens, evicting between chunks. `0` decodes in one call. |
| `EVOKE_LOGICAL_WINDOW` | `0` | Maximum logical context (prompt plus completion). `0` means `EVOKE_N_CTX`. Above `EVOKE_N_CTX` it needs `EVOKE_PREFILL_CHUNK_TOKENS` above 0 and sparse positions, or the server refuses to start. |
| `EVOKE_RECOVERY_PROTECT_THRESHOLD` | `0.0` | A block whose recovery strength is at or above this value (and above 0) is not an eviction candidate. It decays each turn. Without it, a block restored in a turn can be evicted again in the same turn. |
| `EVOKE_W_RECOVERY` | `0.0` | Scorer weight on a block's recovery strength. |
| `EVOKE_RECOVERY_STRENGTH_INIT` | `1.0` | Recovery strength given to a restored block. |
| `EVOKE_RECOVERY_DECAY` | `0.7` | Per-turn multiplier on recovery strength. |
| `EVOKE_SMART_RECOVER_K` | `4` | Top-K for similarity recovery; `0` disables it. |
| `EVOKE_SMART_RECOVER_MIN_SIMILARITY` | `0.0` | Cosine floor for similarity recovery. |
| `EVOKE_USE_RETRIEVAL_EMBEDDINGS` | unset | Use a retrieval embedding model (needs the `fastembed` package) for block embeddings. |
| `EVOKE_KV_RESTORE_RAM_BUDGET_BYTES` | unset (unbounded) | Host RAM cap for saved KV blocks; past it the oldest blocks drop their K/V bytes, or spill if a spill path is set. |
| `EVOKE_KV_RESTORE_SPILL_PATH` | unset | Directory for blocks that exceed the RAM budget. |
| `EVOKE_W_ATTENTION` | `0.0` | Weight of the attention signal in scoring; above 0 builds an attention scorer, which needs the fork build. |
| `EVOKE_ATTN_LAYER` | `20` | Layer tapped for attention capture. |
| `EVOKE_W_JLENS`, `EVOKE_JLENS_PROBE`, `EVOKE_JLENS_LAYERS`, `EVOKE_JLENS_STAT`, `EVOKE_JLENS_BLOCK_AGG` | `0.0`, empty, all layers, `kurtosis`, `mean` | Research scorer. It needs a probe file in `EVOKE_JLENS_PROBE`; `EVOKE_JLENS_BLOCK_AGG` is `mean` or `max`. |

### Sessions, auth and queueing

| Variable | Default | Meaning |
|---|---|---|
| `EVOKE_API_KEYS_FILE` | unset | JSON keys file. Unset means open mode with no authentication. |
| `EVOKE_ISSUED_KEYS_FILE` | unset | Store for keys issued at runtime through `/admin/keys`. Needs `EVOKE_API_KEYS_FILE`. |
| `EVOKE_MAX_SESSIONS` | `8` | Session pool size; the least recently used session is dropped past it. |
| `EVOKE_MAX_SESSIONS_PER_KEY` | `2` | Per-key session cap, applied only when keys are configured. |
| `EVOKE_MAX_WAITING_PER_SESSION` | `1` | Turns allowed to queue behind the running one in a session; past it the request gets 409 `evoke_session_busy`. |
| `EVOKE_QUEUE_TIMEOUT` | unset (wait indefinitely) | Seconds a turn may wait for the engine before 503 `evoke_queue_timeout`. |
| `EVOKE_PIN_SYSTEM_PROMPT` | `0` | `1` pins the leading system messages (including the tool block a chat template puts there) so they are never evicted. It applies from a session's first turn, not retroactively. |
| `EVOKE_PIN_CAP` | `0.6` | Largest share of the budget left after the pinned system prompt that client `evoke_keep: "pin"` marks may cover; past it the request gets 400 `evoke_pin_budget_exceeded`. |

### Diagnostics

| Variable | Meaning |
|---|---|
| `EVOKE_LLAMA_LOG` | Send llama.cpp's own log to stderr. Without it the log is discarded, so a `llama_decode failed with code N` carries no reason. |
| `EVOKE_DEBUG_EVICT` | Print an `[evict_ranges]` line to stdout for every eviction. |
| `EVOKE_DEBUG_IDENTITY` | Diagnostics for identity gap-fill and identity resets, on stderr. |
| `EVOKE_DEBUG_DRIFT`, `EVOKE_DEBUG_DRIFT_FILE` | When the resent prompt diverges from the cached prefix, write a diagnostic block to the file, or to stderr if no file is given. |

## Authentication

With `EVOKE_API_KEYS_FILE` unset the server is open: no `Authorization` header is checked, the `X-Evoke-Session` header is optional, and requests are routed to the session with the longest shared token prefix. The `/admin/keys` routes answer 501.

With a keys file, every request except `/healthz` must send `Authorization: Bearer <token>`, and every chat request must send an `X-Evoke-Session` header (`[A-Za-z0-9._:@/+=-]`, up to 256 characters). Sessions are namespaced by key, so one key cannot see another's.

The keys file is a JSON object mapping a key id to its entry:

```json
{
  "ops": {"token": "at-least-12-characters", "admin": true},
  "svc": {"sha256": "<hex sha256 of the token>"}
}
```

- Key ids match `[A-Za-z0-9._@:+-]{1,128}`. Tokens must be at least 12 characters (a `sha256` digest is not length checked), and no two keys may share a token.
- Only static keys can be admin. An admin sees all sessions and may omit the session header on `/health` and `/admin/reset`.
- Static keys change only by editing the file and restarting.
- With `EVOKE_ISSUED_KEYS_FILE` set, an admin can issue and revoke non-admin keys at runtime (`POST /admin/keys` with `{"key_id": "..."}` returns the token once). Only the SHA-256 of each token is stored, in a file written with mode 0600. Revoking a key drops its sessions and refuses its queued turns. An issued key id may not equal a static key id.

Keep key files and tokens out of the repository and out of shell history.

## HTTP interface

| Route | Auth | Purpose |
|---|---|---|
| `GET /healthz` | none | Liveness and configuration, without session ids. |
| `GET /v1/models` | key | Model list with `context_length` and an `evoke` block (`kv_cells`, `logical_window`, `budget`). |
| `GET /models` | key | List of `.gguf` models in the model directory plus the active one. |
| `POST /models/load` | admin | Body `{"model": "<name without .gguf>"}`. Loads that model and drops all sessions; rolls back on failure. |
| `POST /models/unload` | admin | Frees VRAM now. 409 while requests are in flight. |
| `GET /health` | key | Per-session statistics for the session named in `X-Evoke-Session`. |
| `GET /v1/sessions` | key | Sessions visible to the caller. |
| `DELETE /v1/sessions/{id}` | key | Drop a session. |
| `POST /admin/reset` | key | Reset the session in `X-Evoke-Session`. |
| `GET/POST /admin/keys`, `DELETE /admin/keys/{id}` | admin | List, issue and revoke keys. |
| `POST /admin/set_budget` | admin | Body `{"tokens": N}` (at least 32). Sets the budget for new sessions and resets every existing session. |
| `POST /v1/chat/completions` | key | OpenAI chat completions, with streaming. |

The `model` field of a chat request is required but does not select a model; use `/models/load` for that. `max_tokens` defaults to 2048. A prompt at or above the logical window is rejected with 400 `context_length_exceeded`.

### EVOKE request fields

- `evoke_keep` on a message: `"pin"` (never evict) or `"prefer"` (score multiplier of 2.0 by default). Marks are restated on every request. They apply to content already resident only when positions are sparse and recovery is by identity, which `/healthz` reports as `keep_marks_on_resident`; otherwise a mark covers only content decoded in that request. Any other value is 400 `evoke_keep_invalid`.
- Top-level `evoke_priority` (float, default 1.0), `evoke_pinned` (bool) and `evoke_task_boundary` (bool, resets the task-focus embedding).

### Per-turn metrics

Every non-streaming response carries `X-EVOKE-*` headers, and `usage.evoke` in the body (in the final chunk when streaming with `stream_options.include_usage`): `logical_tokens`, `logical_window`, `kv_cells`, `peak_resident_tokens`, `resident_tokens_end`, `pinned_tokens`, `budget`, `blocks_evicted`, `tokens_evicted`, `blocks_recovered`, `tokens_recovered`, `prompt_tokens_decoded`, `prompt_tokens_reused`, `queue_wait_ms`. Report `peak_resident_tokens` and `resident_tokens_end` together: the peak is what the turn needed in the cache, the end value is what remains after eviction.

### `/healthz` fields

| Field | Meaning |
|---|---|
| `status` | Always `ok`. |
| `model`, `model_loaded` | Active model name; false after an idle or explicit unload. |
| `kv_cells` | Engine context size. |
| `logical_window` | `EVOKE_LOGICAL_WINDOW` if set, else `kv_cells`. |
| `budget` | Active-token budget of the pool configuration. |
| `pin_system_prompt`, `pin_cap_fraction` | Effective `EVOKE_PIN_SYSTEM_PROMPT` and `EVOKE_PIN_CAP`. |
| `keep_marks_on_resident` | True when positions are sparse and recovery is by identity. |
| `auth_required`, `session_required` | A keys file is configured. |
| `kv_block_primitives` | The fork's KV block primitives are loaded. |
| `queue.running`, `queue.waiting` | A turn holds the engine; number of turns waiting. |
| `version` | Application version. |

`GET /health` adds per-session counters: `active_tokens`, `active_blocks`, `budget_utilization`, `total_evictions`, `total_recoveries`, `peak_active`, `total_prompt_tokens`, `total_new_decoded`, `identity_recovered`, `identity_mismatch`.

## Logs

The server has no log file setting. It prints to standard output and uvicorn writes its own startup and access lines to standard error, so redirect both where you want them (`> server.log 2>&1`). A redirect with `>` truncates on every start; use `>>` or a supervisor that rotates logs if you want history.

On startup you will see the policy line, `loading model`, `n_ctx=... kv_quant=...`, `ready (...)` and `serving on http://HOST:PORT`. Per request, a line like

```
[req chatcmpl-... ] msgs=3 tools=14 stream=True max_new=2048 prompt_chars=... prompt_tokens=20662 pinned=20572 keep=[(20572, 20656, 'pin')] session=<key>|<session> stops=[...]
```

shows the prompt size, the pinned tokens, any keep marks received, and the session. Other tagged lines: `[stream ...]` (end of a stream, engine errors, tool-call parse failures), `[models]` (unload, reload, switch), `[tmpl]` (chat template fallbacks), `[idle]`, `[500]`, and `[422]`.

The `[422]` line logs up to 2000 characters of the rejected request body, which can include prompt text; keep that in mind before sharing logs. The Authorization token is not logged, but the key id is part of the session id in `[req ...]` lines.

For more detail turn on `EVOKE_LLAMA_LOG` (llama.cpp messages, on stderr) and `EVOKE_DEBUG_EVICT`.

## Tests

```bash
uv run --extra server --extra dev pytest tests/ -x -q
```

The suite runs on CPU against a mock engine in well under a minute; tests that need a GPU or a model are skipped. Passing it does not show that the server works on your GPU. For that, start the server and run a short conversation against it (`scripts/eviction_demo.py` is one example), then check `/health` for evictions and recoveries.

## Troubleshooting

**The server exits with `FAIL: set EVOKE_MODEL_PATH ...`.** The variable is unset or empty. It does not check that the file exists; a wrong path shows up as `Failed to load model`.

**`LLAMA_CPP_LIB is set but its llama_context_params layout is not a known fork layout`.** The library is not an EVOKE fork build, or it is a build newer than this checkout knows. Rebuild the fork, or update the layout table in `src/evoke/_engine_lib.py`.

**`KV block primitives unavailable; set LLAMA_CPP_LIB`.** You are running on the stock wheel and recovery mode is `kv_restore`. Point `LLAMA_CPP_LIB` at the fork build.

**`Failed to create context`.** The context does not fit in VRAM. Lower `EVOKE_N_CTX`, use `EVOKE_KV_QUANT=q8_0`, or close other GPU processes.

**`EVOKE_LOGICAL_WINDOW above EVOKE_N_CTX needs ...`.** Set `EVOKE_PREFILL_CHUNK_TOKENS` above 0 and make sure positions are sparse (`EVOKE_RECOVERY_MATCH=identity`, or `EVOKE_POSITION_MODE=sparse`).

**`EVOKE_ISSUED_KEYS_FILE needs EVOKE_API_KEYS_FILE`.** Set both. This and the keys file errors are raised after the model has loaded, so they take a while to appear.

**Eviction and recovery settings seem to do nothing.** `EVOKE_BUDGET` is unset; see the configuration notes above. Check the startup `policy=` line: it is missing when the budget is unset.

**Generation is extremely slow, or output is wrong on a Qwen3 model.** Flash attention should be left on automatic. With it forced off the KV block splice runs hundreds of times slower; with it forced on, Qwen3 models with QK-norm can produce bad output. Leave `EVOKE_FLASH_ATTN` unset.

**Garbled replies after a few thousand tokens with YaRN on.** Your fork build lacks the K-shift fix (see YaRN above), or the factor and original context do not match the model.

**A recovered block is evicted again in the same turn.** Set `EVOKE_RECOVERY_PROTECT_THRESHOLD` above 0 (0.5 is a reasonable start).

**`401 invalid_api_key` or `400 evoke_session_required`.** A keys file is configured: send the Bearer token and an `X-Evoke-Session` header with every chat request. A key issued at runtime stops working the moment it is revoked, including for turns already queued.

**`409 evoke_session_busy`, or `503 evoke_queue_timeout`.** The session already has a turn running and `EVOKE_MAX_WAITING_PER_SESSION` queued, or the engine was busy longer than `EVOKE_QUEUE_TIMEOUT`. The 503 carries `Retry-After: 5`.

**`400 context_length_exceeded`.** The prompt reaches the logical window. Compact the client's history, or raise `EVOKE_LOGICAL_WINDOW` (with chunked prefill) or `EVOKE_N_CTX` if the model and VRAM allow.

**Empty replies or `llama_decode failed with code N`.** Turn on `EVOKE_LLAMA_LOG` to see llama.cpp's reason. A common cause is a prompt plus generation that no longer fits the free cells.

**VRAM stays in use after a crash or stop.** Find the leftover process (`nvidia-smi`, then your process list) and end it before starting again.

**Stale code after copying files to another machine.** Delete `__pycache__` directories under `src/` before starting, or the old bytecode can be used.

**A restart loses all sessions.** Sessions live in memory and are not restored after a restart. Keys issued at runtime are stored on disk and survive it.
