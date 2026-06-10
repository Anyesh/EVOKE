# Live agent demo: opencode builds a webapp through EVOKE (Qwen3-8B)

2026-06-11. A real coding agent (opencode 1.16.0, 9 tools) builds a notes
webapp through the EVOKE OpenAI-compatible server on chihiro, Qwen3-8B-Q4_K_M
(a thinking model, dense attention), n_ctx=16384. Same task, model, and server
across three arms; only the eviction/recovery policy changes. Raw counters in
`agent_opencode_qwen3_8b.json`.

## Arms and signatures

| arm | budget | prompt tokens seen | decoded | evictions | identity splices | peak resident |
|---|---|---|---|---|---|---|
| EVOKE (kv_restore) | 2048 | 17,397 | 9,719 (56%) | 124 | 59/59, 0 mismatch | trimmed to budget at turn ends |
| discard | 2048 | per-turn = decoded | 100% every turn (~17.2K) | 63/turn-2 | 0 | trimmed to budget at turn ends |
| no eviction | 14000 | 44,886 | 9,910 (22%) | 0 | 0 | 10,952 and growing |

Turn counts differ across arms (2, 2, 5) because generation varies run to run,
so per-arm signatures are the comparison, not absolute totals.

## Reading

- The no-eviction arm shows what an intact cache buys: natural prefix caching
  makes decode cheap, but the resident set grows with session length without
  bound (10,952 tokens at the end of a short build; a long session exhausts
  the context).
- The discard arm respects the budget but pays for it in decode: with the
  saved KV gone, the identity path resets each turn and the full prompt
  re-decodes every time.
- The EVOKE arm holds both properties at once: the budget is enforced at turn
  ends (124 evictions), and the next request's identity gap-fill splices every
  evicted block back recompute-free (59/59, zero mismatches), so only the
  re-templated echo of the generated turn plus genuinely new content decodes.
  Turn 2 recovered the entire 7,678-token turn-1 prefix without a forward
  pass.

## Correctness

opencode re-sends full history every turn, so no arm loses content by design.
The EVOKE and discard arms produced working apps (discard's verified serving
GET / and the notes API; EVOKE's had one model-emitted syntax flaw, a
double-escaped newline). The no-eviction arm's build failed because the model
emitted its final tool call inside the thinking trace and stopped, a model
behavior unrelated to the arm.

## Mechanism notes

- Title-generation side requests routed to their own session by
  prefix-affinity routing in all arms; without that isolation they reset the
  agent session and destroy the recovery archive (measured earlier the same
  day).
- The echo of a tool-calling assistant turn re-renders through the chat
  template with different JSON spacing than the raw emit, so it never
  byte-matches the cache. The post-gap-fill tail-evict drops just that region
  (1-2K tokens) instead of resetting the session.
- Generation clamps to physical context capacity and finishes with "length"
  at the wall; before that clamp, thinking spirals that reached n_ctx crashed
  llama_decode mid-stream.
