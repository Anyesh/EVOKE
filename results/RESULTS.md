# EVOKE Competitive Benchmark Results

## Run: Qwen 2.5 7B (2026-05-19, v1)

**Model**: Qwen2.5-7B-Instruct-Q4_K_M
**Hardware**: RTX 4070 Ti SUPER, 16GB VRAM (gpu-host)
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
