# EVOKE Competitive Benchmark Results

## Run: Qwen 2.5 7B (2026-05-22, baseline_bench n=5 + reviewer revisions complete)

**Model**: Qwen2.5-7B-Instruct-Q4_K_M
**Hardware**: RTX 4070 Ti SUPER, 16GB VRAM

Reviewer asked for n>=5 reruns of the §7.3 head-to-head to back the elapsed-time claim with a CI rather than a single-run point estimate. Ran `scripts/baseline_bench.py` 5 times against the EVOKE server on gpuhost. Per-run elapsed times (seconds):

| policy        | run 1 | run 2 | run 3 | run 4 | run 5 |
|---------------|------:|------:|------:|------:|------:|
| no_eviction   | 19.4  | 19.2  | 19.5  | 19.4  | 20.6  |
| truncate      | 21.6  | 21.3  | 21.8  | 21.8  | 26.7  |
| evoke         | 21.7  | 21.6  | 22.1  | 21.7  | 21.9  |

Aggregate (mean, 95% CI via t-distribution, df=4):

| policy        | active end | evict | recov | elapsed (mean) | 95% CI            |
|---------------|-----------:|------:|------:|---------------:|-------------------|
| no_eviction   |       2476 |     0 |     0 |     **19.62s** | [18.93s, 20.31s]  |
| truncate      |        685 |    66 |     0 |     **22.64s** | [19.81s, 25.47s]  |
| evoke         |        684 |    90 |    24 |     **21.80s** | [21.55s, 22.05s]  |

**Finding**: the v1 single-run "31% faster than no_eviction" claim does not reproduce against the current code. Smart-recovery's bge-small per-block embeddings + per-turn similarity scoring add ~1.5-2s of overhead beyond truncate. EVOKE now sits inside truncate's CI ([19.81, 25.47]) and ~11% above no_eviction. The cache-footprint story (684 vs 2476 active tokens, 4× smaller) holds; the wall-clock story is now "comparable to truncate, slower than no_eviction" rather than "31% faster". Updated §7.3 paper text and abstract accordingly.

Raw output: `results/baseline_bench_n5_qwen25_7b.txt`.

---

## Run: Qwen 2.5 7B (2026-05-22, SnapKV + InfLLM head-to-head added per reviewer)

**Model**: Qwen2.5-7B-Instruct-Q4_K_M
**Hardware**: RTX 4070 Ti SUPER, 16GB VRAM
**n_ctx**: 16384

Two new same-substrate baselines added per reviewer feedback:
- **SnapKV** (Liu et al., NeurIPS 2024): observation-window snapshot of attention from the last 32 prompt tokens, frozen at end of each user message; eviction picks top-scoring blocks by frozen scores. `recovery_mode=discard`.
- **InfLLM** (Xiao et al., NeurIPS 2024): aggressive eviction to sinks + 25% local window resident; per-turn top-K=8 smart recovery via `kv_block_load` (our same-substrate adaptation of InfLLM's external-memory + attention-mask design).

### NIAH (5 needles × 5 depths × 3 budgets, 25 cells per (budget, strategy))

| policy           | budget 512 | budget 1024 | budget 2048 |
|------------------|-----------:|------------:|------------:|
| recency          |       20%  |        20%  |        40%  |
| streaming_llm    |        0%  |        20%  |        20%  |
| evoke_discard    |        0%  |        20%  |        20%  |
| evoke_breadcrumb |        0%  |        20%  |        20%  |
| h2o              |       20%  |        20%  |        40%  |
| **snapkv**       |       20%  |        20%  |        40%  |
| **infllm**       |     **96%**|      **96%**|        80%  |
| evoke_kv_restore |       96%  |       100%  |       100%  |
| evoke_attention  |       96%  |       100%  |       100%  |

Story: recovery is the differentiator. Smarter selection alone (H2O, SnapKV) flattens at the same 20-40% as no scoring at all. EVOKE matches the closest external-memory baseline (InfLLM) at the tight budget and beats it as budget headroom grows — InfLLM's fixed aggressive footprint evicts naturally-resident needles when there is room to keep them. Raw: `results/niah_qwen25_7b_with_snapkv_infllm.json`.

### Multifact (5 seeds × 5 facts, budget 1024, gemma judge)

| strategy          | pass rate | 95% Wilson CI         |
|-------------------|----------:|-----------------------|
| recency           |      4.0% | [0.71%, 19.54%]       |
| streaming_llm     |      0.0% | [0.00%, 13.32%]       |
| evoke_discard     |      0.0% | [0.00%, 13.32%]       |
| evoke_breadcrumb  |      0.0% | [0.00%, 13.32%]       |
| h2o               |      0.0% | [0.00%, 13.32%]       |
| **snapkv**        |      4.0% | [0.71%, 19.54%]       |
| evoke_attention   |     48.0% | [30.03%, 66.50%]      |
| evoke_kv_restore  |     60.0% | [40.74%, 76.60%]      |
| **infllm**        |   **64.0%**| [44.52%, 79.75%]    |

Story: InfLLM (K=8) edges EVOKE (K=4) on multifact's multi-topic-shift workload — larger retrieval window catches more relevant blocks across the five fact-shifts. CIs overlap heavily so the difference is within statistical noise. Recovery-less policies cluster at 0-4%. evoke_attention drops below evoke_kv_restore (48% vs 60%) on this multi-probe workload — the multi-signal scorer's attention contribution adds noise across topic shifts, the inverse of NIAH's single-probe regime. Raw: `results/mfb_qwen25_7b_with_snapkv_infllm.json`.

### Agentic eval (14-turn, oracle FACT_KEY recovery for kv_restore policies)

| budget | strategy          | probe | evict | recov | rec_ms |
|-------:|-------------------|-------|------:|------:|-------:|
|    512 | recency           | fail  | 35    | 0     | 0.00   |
|    512 | streaming_llm     | fail  | 35    | 0     | 0.00   |
|    512 | evoke_discard     | fail  | 35    | 0     | 0.00   |
|    512 | h2o               | fail  | 34    | 0     | 0.00   |
|    512 | snapkv            | fail  | 34    | 0     | 0.00   |
|    512 | evoke_breadcrumb  | PASS  | 35    | 0     | 17.25  |
|    512 | evoke_kv_restore  | PASS  | 35    | 1     | **3.13** |
|    512 | evoke_attention   | PASS  | 31    | 0     | **0.01** |
|    512 | infllm            | PASS  | 39    | 1     | 3.12   |
|   1024 | recency           | fail  | 29    | 0     | 0.00   |
|   1024 | streaming_llm     | fail  | 28    | 0     | 0.00   |
|   1024 | evoke_discard     | fail  | 28    | 0     | 0.00   |
|   1024 | h2o               | fail  | 24    | 0     | 0.00   |
|   1024 | snapkv            | fail  | 24    | 0     | 0.00   |
|   1024 | evoke_breadcrumb  | PASS  | 28    | 0     | 15.40  |
|   1024 | evoke_kv_restore  | PASS  | 28    | 1     | **3.86** |
|   1024 | evoke_attention   | PASS  | 26    | 1     | 3.75   |
|   1024 | infllm            | PASS  | 35    | 1     | 3.76   |
|   2048 | recency           | fail  | 9     | 0     | 0.00   |
|   2048 | streaming_llm     | fail  | 10    | 0     | 0.00   |
|   2048 | evoke_discard     | fail  | 10    | 0     | 0.00   |
|   2048 | h2o               | fail  | 4     | 0     | 0.00   |
|   2048 | snapkv            | fail  | 4     | 0     | 0.00   |
|   2048 | evoke_breadcrumb  | PASS  | 10    | 0     | 15.70  |
|   2048 | evoke_kv_restore  | PASS  | 10    | 1     | **3.89** |
|   2048 | evoke_attention   | PASS  | 8     | 0     | **0.00** |
|   2048 | infllm            | PASS  | 31    | 1     | 3.07   |

Story: every recovery-less policy fails at every budget; SnapKV and H2O join recency/streaming_llm/discard in the failure column. InfLLM passes everywhere with kv_restore-equivalent recovery latency (~3 ms). evoke_attention sometimes avoids recovery entirely (0 recov at budgets 512 and 2048) because attention-driven scoring keeps the fact block resident. Raw: `results/agent_bench_qwen25_7b_with_snapkv_infllm.txt`.

---

## Run: Qwen 2.5 7B (2026-05-20, head-to-head + agentic eval — drift fix landed)

**Model**: Qwen2.5-7B-Instruct-Q4_K_M
**Hardware**: RTX 4070 Ti SUPER, 16GB VRAM
**n_ctx**: 16384, budget 1024 (where applicable)

### Head-to-head baselines (`scripts/baseline_bench.py`, 14-turn planted-fact session)

| policy        | budget | active end | evictions | recoveries | probe | elapsed |
|---------------|------:|-----------:|----------:|-----------:|-------|--------:|
| no_eviction   | 16384 |       2476 |         0 |          0 | PASS  |   18.7s |
| truncate      |  1024 |        683 |        40 |          0 | PASS  |   19.8s |
| evoke         |  1024 |        747 |        40 |         24 | PASS  | **12.9s** |

Both truncate and evoke contain the active footprint inside the budget. evoke runs the same conversation 35% faster than truncate with the same eviction count — recoveries via `kv_block_load` (~3 ms) replace truncate's per-turn re-decode of missing history (~150-250 ms).

### Agentic eval (`scripts/agent_bench.py`, planted-config-file probe after 10 unrelated file reads)

| budget | strategy          | probe | evict | recov | rec_ms |
|-------:|-------------------|-------|------:|------:|-------:|
|    512 | recency           | fail  | 35    | 0     | 0.00   |
|    512 | streaming_llm     | fail  | 35    | 0     | 0.00   |
|    512 | evoke_discard     | fail  | 35    | 0     | 0.00   |
|    512 | evoke_breadcrumb  | PASS  | 35    | 0     | 17.26  |
|    512 | evoke_kv_restore  | PASS  | 35    | 1     | **3.13** |
|   1024 | recency           | fail  | 29    | 0     | 0.00   |
|   1024 | streaming_llm     | fail  | 28    | 0     | 0.00   |
|   1024 | evoke_discard     | fail  | 28    | 0     | 0.00   |
|   1024 | evoke_breadcrumb  | PASS  | 28    | 0     | 15.22  |
|   1024 | evoke_kv_restore  | PASS  | 28    | 1     | **2.96** |
|   2048 | recency           | fail  | 9     | 0     | 0.00   |
|   2048 | streaming_llm     | fail  | 10    | 0     | 0.00   |
|   2048 | evoke_discard     | fail  | 10    | 0     | 0.00   |
|   2048 | evoke_breadcrumb  | PASS  | 10    | 0     | 15.80  |
|   2048 | evoke_kv_restore  | PASS  | 10    | 1     | **3.66** |

Recovery is the differentiator: every strategy without recovery fails at every budget. `kv_restore` is ~5x faster than `breadcrumb` while answering equivalently.

Raw output: `results/agent_bench_qwen25_7b.txt`.

### Drift fix details

Two underlying bugs were fixed before these numbers became trustworthy.

1. **Non-canonical BPE in the assistant emit**: the model can generate `**` + `:\n` (tokens 334, 510) for text whose canonical retokenization is `**:` + `\n` (tokens 95518, 198). Subsequent requests' Jinja-templated prompt retokenizes to the canonical form, so prefix-match diverged mid-history and forced session resets that zeroed eviction stats. Fix: `Session._canonicalize_assistant` re-tokenizes the visible content after each generation, evicts the model's emit, and re-decodes the canonical tokens at the same position. The K/V cache stays consistent with the model's actual computation history; only the token-id labels in `_cached_tokens` change.

2. **Stale Session token view after manager eviction**: when `EvokeManager._enforce_budget` evicts blocks from the engine, the engine cache shifts (positions of remaining blocks compact down). `Session._cached_tokens` did not reflect this and continued to claim it had content the engine had dropped. Fix: `Session.sync_prefix` re-derives the cached view from `EvokeManager.get_token_view()` (concatenation of active block token-ids in physical position order). When divergence is still detected (typical for `truncate` policy: prior eviction dropped middle history that the next request resupplies), the session tail-evicts the diverged portion instead of fully resetting, preserving the matching prefix and the eviction counter.

After the fix, the eviction counter monotonically accumulates across turns for `truncate` (40 evictions over 14 turns at budget 1024, instead of resetting to 0 mid-session), which is what makes the head-to-head numbers comparable across policies.

---

## Run: Qwen 2.5 7B (2026-05-19, v1)

**Model**: Qwen2.5-7B-Instruct-Q4_K_M
**Hardware**: RTX 4070 Ti SUPER, 16GB VRAM
**Context**: n_ctx=131072, n_embd=3584
**Template**: ChatMLTemplate (non-thinking)

### Strategies

| Strategy | Scoring | Sinks | Retrieval | Description |
|---|---|---|---|---|
| full | N/A | N/A | N/A | No eviction. Quality ceiling. |
| truncate | recency only | no | no | Pure sliding window. Naive baseline. |
| streaming_llm | recency only | yes | no | StreamingLLM: sinks + sliding window. |
| evoke_no_ret | recency + coherence | yes | no | EVOKE scoring without retrieval (ablation). |
| evoke | recency + coherence | yes | yes | Full EVOKE system. |

All strategies use block_size=32, watermark policy (high=0.95, low=0.75).

### Test Cases

- **Needle-in-haystack**: Filler document (~3001 tokens) with "CRYSTALLINE-HORIZON-42" planted at 5 positions (10%, 25%, 50%, 75%, 90%). Single QA turn after document load.
- **Multi-turn recall**: Filler document + fact injected in a user message, followed by 4-5 filler conversation turns, then an eval question. Three cases: recall-code (AURORA-SEVEN), recall-date (March 15th meeting), recall-name (Dr. Elena Vasquez).

### Needle-in-Haystack Results (Contains %)

| Strategy | Budget 512 | Budget 1024 | Budget 2048 |
|---|---|---|---|
| full | 100% | 100% | 100% |
| **evoke** | **80%** | **100%** | **100%** |
| evoke_no_ret | 20% | 20% | 40% |
| streaming_llm | 20% | 20% | 40% |
| truncate | 20% | 20% | 40% |

**Takeaway**: Retrieval is the differentiator. EVOKE with retrieval finds the needle 80-100% of the time. Without retrieval, EVOKE scores identically to baselines. The scoring alone (coherence + recency) does not help retain the needle; retrieval from archive does.

### Multi-Turn Recall Results (Contains %)

| Strategy | Budget 512 | Budget 1024 | Budget 2048 |
|---|---|---|---|
| full | 100% | 100% | 100% |
| truncate | 100% | 100% | 100% |
| streaming_llm | 100% | 100% | 100% |
| evoke_no_ret | 100% | 100% | 100% |
| **evoke** | **0%** | **33%** | **100%** |

**Takeaway**: EVOKE's retrieval mechanism HURTS multi-turn recall at tight budgets. All strategies without retrieval achieve 100% because the conversation is compact (~400 tokens) and fits entirely in the recency window, even at 512 budget.

### Root Cause Analysis: Multi-Turn Recall Regression

EVOKE shows 40 promotions and 122 demotions on recall-name at 512 budget, vs 0 promotions and 83 demotions for baselines.

**What's happening**: The retrieval gate fires on every user message, including filler turns. When the user asks "What is the weather forecast for tomorrow?", lexical matching finds the word "weather" in archived document blocks. Those blocks get promoted (with neighbor expansion), triggering a rebuild and budget enforcement. Budget enforcement evicts the conversation block containing the planted fact (low coherence with "weather"), because coherence has weight 0.6 in the composite score. Over 5 filler turns, this thrashing pattern pushes the planted fact out of reach.

**The bug**: `archive.retrieve_by_similarity` includes lexical hits with ANY overlap (`lex > 0`). A single shared word between a filler question and an archived document block triggers promotion. There is no minimum quality threshold on lexical matches.

**Why baselines survive**: Truncation and StreamingLLM use pure recency scoring (w_coherence=0.0). They never evict the conversation block because it has high recency relative to the document blocks. The entire conversation fits within 512 tokens, so no conversation content is lost.

**Specific failure example** (recall-name, budget=512):
- Inject: "Dr. Elena Vasquez is the lead researcher on the deep-sea bioluminescence project."
- Filler turns match filler doc blocks on shared words ("weather", "deep-sea")
- Each filler turn promotes 6-7 document blocks (lexical hits + neighbors)
- Budget enforcement evicts the inject turn (low coherence with filler topic)
- Eval query promotes correct block but model context is corrupted from thrashing
- Model output: "The passage does not provide information about the lead researcher."

### Fix Applied: min_lexical_recall + raw_query (v2, v3)

Two fixes were applied to address the retrieval precision problem:

1. **min_lexical_recall threshold** (v2, v3): `archive.retrieve_by_similarity` now requires `lex >= min_lexical_recall` (0.4) instead of `lex > 0`. Filters single-word coincidental matches while preserving genuine recall queries.
2. **raw_query parameter** (v3): `process_user_message` accepts `raw_query` for clean text matching, avoiding template token contamination (e.g., `<|im_start|>`, `user`, `assistant` appearing as lexical matches).

### v1 Raw Data (before fix)

```
Strategy         Budget Type         F1 Contains Promo  Demo   Time
-------------------------------------------------------------------
evoke               512 needle    0.178      80%   2.6    84   0.21
evoke               512 recall    0.000       0%  26.3   108   0.18
evoke              1024 needle    0.222     100%   2.6    71   0.16
evoke              1024 recall    0.175      33%  23.7    95   0.23
evoke              2048 needle    0.222     100%   1.8    47   0.17
evoke              2048 recall    0.449     100%  21.0    66   0.20
```

JSON data: `results/bench_qwen25_7b_competitive_v1.json`

---

## Run: Qwen 2.5 7B (2026-05-19, v3 — lexical threshold 0.4 + raw_query)

Same hardware, model, and test cases as v1. Fixes: `min_lexical_recall=0.4`, `raw_query` parameter.

### Needle-in-Haystack Results (Contains %)

| Strategy | Budget 512 | Budget 1024 | Budget 2048 |
|---|---|---|---|
| full | 100% | 100% | 100% |
| **evoke** | **80%** | **100%** | **100%** |
| evoke_no_ret | 20% | 20% | 40% |
| streaming_llm | 20% | 20% | 40% |
| truncate | 20% | 20% | 40% |

Needle results unchanged from v1. The fixes targeted retrieval precision, not needle recall.

### Multi-Turn Recall Results (Contains %)

| Strategy | Budget 512 | Budget 1024 | Budget 2048 |
|---|---|---|---|
| full | 100% | 100% | 100% |
| truncate | 100% | 100% | 100% |
| streaming_llm | 100% | 100% | 100% |
| evoke_no_ret | 100% | 100% | 100% |
| **evoke** | **67%** | **100%** | **100%** |

Major improvement from v1 (0%/33%/100% → 67%/100%/100%). Average EVOKE promotions at 512 dropped from 26.3 to 5.7 per recall case.

### Remaining Failures at Budget 512

**1. recall-date (EVOKE, budget=512)**: 11 promotions, no date fact retrieved.

The filler turns ("stock market trends", "bioluminescent deep-sea creatures") semantically match archived document blocks. `retrieve_by_similarity` uses both lexical and semantic paths: the lexical threshold blocks weak word matches, but semantic hits (cosine similarity ≥ 0.85) still trigger promotions. Each promotion also triggers neighbor expansion (up to 3× the hit count), so 4 semantic hits can become 12 promoted blocks = 384 tokens into a 512-token budget. Budget enforcement then evicts conversation blocks (including the date fact) to make room.

**2. needle@10% (EVOKE, budget=512)**: 3 promotions, wrong blocks retrieved.

Generated answer starts with `<|im_start|>user\n...`, indicating template token leakage from promoted document blocks. The promoted blocks contain template wrapper tokens from the original `wrap_document_prefix()` call; after KV rebuild, these tokens appear mid-sequence and confuse the model into generating as if in a user role.

### Root Cause: Promotion Thrash at Tight Budgets

At 512 budget (16 blocks of 32), any promotion of 4+ blocks replaces 25%+ of the active window. Neighbor expansion (which triples hit count) makes this much worse. The baselines avoid this entirely because they never promote.

The fundamental tradeoff: EVOKE's retrieval gives it 80% needle recall vs 20% for baselines, but the same retrieval mechanism causes churn that drops conversation recall from 100% to 67% at 512 budget. At 1024+, there is enough headroom for both.

### v3 Raw Data

```
Strategy         Budget Type         F1 Contains Promo  Demo   Time
-------------------------------------------------------------------
evoke               512 needle    0.178      80%   2.4    84   0.21
evoke               512 recall    0.274      67%   7.0    90   0.21
evoke              1024 needle    0.222     100%   2.4    71   0.16
evoke              1024 recall    0.449     100%   7.0    78   0.19
evoke              2048 needle    0.222     100%   1.6    47   0.16
evoke              2048 recall    0.449     100%   6.3    47   0.21
evoke_no_ret        512 needle    0.044      20%   0.0    83   0.29
evoke_no_ret        512 recall    0.449     100%   0.0    83   0.19
evoke_no_ret       1024 needle    0.044      20%   0.0    71   0.21
evoke_no_ret       1024 recall    0.449     100%   0.0    71   0.20
evoke_no_ret       2048 needle    0.089      40%   0.0    47   0.24
evoke_no_ret       2048 recall    0.449     100%   0.0    47   0.20
full                512 needle    0.222     100%   0.0     0   0.17
full                512 recall    0.466     100%   0.0     0   0.20
full               1024 needle    0.222     100%   0.0     0   0.16
full               1024 recall    0.466     100%   0.0     0   0.20
full               2048 needle    0.222     100%   0.0     0   0.16
full               2048 recall    0.466     100%   0.0     0   0.21
streaming_llm       512 needle    0.044      20%   0.0    83   0.26
streaming_llm       512 recall    0.449     100%   0.0    83   0.20
streaming_llm      1024 needle    0.044      20%   0.0    71   0.21
streaming_llm      1024 recall    0.449     100%   0.0    71   0.19
streaming_llm      2048 needle    0.089      40%   0.0    47   0.23
streaming_llm      2048 recall    0.449     100%   0.0    47   0.20
truncate            512 needle    0.044      20%   0.0    83   0.29
truncate            512 recall    0.449     100%   0.0    83   0.19
truncate           1024 needle    0.044      20%   0.0    71   0.20
truncate           1024 recall    0.466     100%   0.0    71   0.19
truncate           2048 needle    0.089      40%   0.0    47   0.23
truncate           2048 recall    0.449     100%   0.0    47   0.20
```

JSON data: `results/bench_qwen25_7b_v3.json`

---

## Run: Qwen 2.5 7B (2026-05-19, v2 — full rearchitecture)

Same hardware and model. Changes from v3:
1. **Block source classification**: blocks tagged as SYSTEM/DOCUMENT/USER/ASSISTANT
2. **Source-aware eviction**: USER floor 0.6, ASSISTANT floor 0.5, DOCUMENT no floor
3. **Budget-aware promotion**: 25% cap per retrieval cycle, skip if above high watermark
4. **No neighbor expansion**: disabled by default (was tripling promotion volume)
5. **BPE token + IDF retrieval**: subword-level matching with BM25-style weighting
6. **Rolling context embedding**: max similarity over last 5 context embeddings
7. **Generated token tracking**: assistant output tracked as ASSISTANT blocks
8. **Promotion grace period**: recently promoted blocks protected for 64 steps

### Multi-Turn Recall (AURORA-SEVEN test, v2 multi_turn_bench.py)

| Budget | AURORA Check | Promotions | Answer Quality |
|--------|-------------|------------|----------------|
| **512** | **PASS** | 3 | "AURORA-SEVEN has a budget of exactly $4.2 million." |
| **1024** | **PASS** | 3 | All 3 facts recalled: codename, budget, Friday deadline |

### Comparison: v3 vs v2 at Budget 512

| Metric | v3 | v2 |
|--------|----|----|
| Recall check | **FAIL** (67%) | **PASS** (100%) |
| Total promotions | 7-11 | 3 |
| Thrashing | Yes (neighbor expansion) | No |
| Template leak | Yes | Still present (separate issue) |

The critical regression at 512 budget is fixed. Source-aware scoring protects conversation blocks from eviction while document blocks are demoted first. Budget-capped promotion without neighbor expansion prevents the thrashing cycle.

Raw data: `results/bench_qwen25_7b_v2.txt`
