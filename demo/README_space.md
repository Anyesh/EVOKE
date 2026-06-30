---
title: EVOKE KV Cache Demo
emoji: 🧠
colorFrom: orange
colorTo: blue
sdk: docker
pinned: true
---

# EVOKE: Live KV Cache Recovery Demo

This Space runs **EVOKE** (EVict and recOver KV cache Entries), a selective KV cache
eviction and recovery system for long-context LLM inference.

The demo loads Qwen2.5-7B-Instruct under a KV budget of 2048 tokens and lets you compare
two eviction strategies in a live multi-turn chat:

- **EVOKE (kv_restore)**: evicted blocks are saved to RAM and spliced back in on demand,
  zero forward-pass recompute.
- **Evict, no recovery**: evicted tokens are discarded; the model works from whatever
  remains in the active cache.

Source: [github.com/Anyesh/unlearn](https://github.com/Anyesh/unlearn)
Paper: coming soon
