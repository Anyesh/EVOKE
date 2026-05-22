# Multifact per-fact pass-rate cross-tab (qwen25_7b_snapkv_infllm)

Pass rate of each strategy on each of the five planted facts, aggregated
over the n=5 seeds. The reviewer's Q4 asks whether multifact's 60%
headline is selection failure (wrong block recovered) or substitution
failure (right block, wrong K/V). A strategy that fails the same fact
across most seeds is consistent with that fact being structurally hard
for the strategy's selection rule. A strategy whose failure mass is
spread evenly across facts is more consistent with substitution noise.

| strategy        | amount   | capital  | code     | date     | password |  overall |
|-----------------|----------|----------|----------|----------|----------|---------:|
| recency         | 0/5      | 0/5      | 0/5      | 1/5      | 0/5      |    4.0% |
| streaming_llm   | 0/5      | 0/5      | 0/5      | 0/5      | 0/5      |    0.0% |
| evoke_discard   | 0/5      | 0/5      | 0/5      | 0/5      | 0/5      |    0.0% |
| evoke_breadcrumb | 0/5      | 0/5      | 0/5      | 0/5      | 0/5      |    0.0% |
| h2o             | 0/5      | 0/5      | 0/5      | 0/5      | 0/5      |    0.0% |
| snapkv          | 0/5      | 0/5      | 0/5      | 1/5      | 0/5      |    4.0% |
| infllm          | 4/5      | 5/5      | 2/5      | 1/5      | 4/5      |   64.0% |
| evoke_kv_restore | 3/5      | 5/5      | 3/5      | 0/5      | 4/5      |   60.0% |
| evoke_attention | 3/5      | 5/5      | 0/5      | 0/5      | 4/5      |   48.0% |
