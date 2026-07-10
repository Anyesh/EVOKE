---
title: EVOKE KV Cache Demo
emoji: 🧠
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: true
---

# EVOKE: Live KV Cache Recovery Demo

This Space runs **EVOKE** (EVict and recOver KV cache Entries), a selective KV cache
eviction and recovery system for long-context LLM inference.

The demo loads Qwen3-4B under a configurable KV budget (default 384 tokens) and lets you
compare three strategies in a live multi-turn chat:

- **EVOKE (kv_restore)**: evicted blocks are saved to RAM and spliced back in on demand by
  content identity, with zero forward-pass recompute.
- **EVOKE + workspace eviction**: blocks are additionally scored by a probe distilled from
  the model's Jacobian-lens workspace, so content the model will read from later tends not
  to be evicted in the first place.
- **Evict, no recovery**: evicted tokens are discarded; the model works from whatever
  remains in the active cache.

Tell the assistant a fact about yourself, ask a few EVOKE questions to push the budget, then
ask it to recall your fact. At a tight budget the EVOKE arms splice the fact back and answer
correctly while the discard arm has forgotten it.

The Space runs on CPU hardware, so generation is slower than the GPU prototype; thinking
mode is disabled to keep turns interactive.

Source: [github.com/Anyesh/EVOKE](https://github.com/Anyesh/EVOKE)
