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

The demo loads Qwen2.5-3B-Instruct under a configurable KV budget (default 384 tokens) and
lets you compare two strategies in a live multi-turn chat:

- **EVOKE (kv_restore)**: evicted blocks are saved to RAM and spliced back in on demand by
  content identity, with zero forward-pass recompute.
- **Evict, no recovery**: evicted tokens are discarded; the model works from whatever
  remains in the active cache.

Tell the assistant a fact about yourself, ask a few EVOKE questions to push the budget, then
ask it to recall your fact. At a tight budget the EVOKE arm splices the fact back and answers
correctly while the discard arm has forgotten it. Raise the budget to 512 and the chat fits
without eviction, so both arms remember.

Source: [github.com/Anyesh/EVOKE](https://github.com/Anyesh/EVOKE)
