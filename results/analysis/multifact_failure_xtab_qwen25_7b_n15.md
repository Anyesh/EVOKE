# Multifact per-fact pass-rate cross-tab (qwen25_7b_n15)

Pass rate of each strategy on each of the five planted facts, aggregated
over 15 seeds x 3 budgets (45 cells per (strategy, fact)).
The reviewer's Q4 asks whether multifact's pass-rate headline is selection
failure (wrong block recovered) or substitution failure (right block,
wrong K/V). A strategy that fails the same fact across most seeds is
consistent with that fact being structurally hard for the strategy's
selection rule. A strategy whose failure mass is spread evenly across
facts is more consistent with substitution noise.

| strategy        | amount   | capital  | code     | date     | password |  overall |
|-----------------|----------|----------|----------|----------|----------|---------:|
| recency         | 2/45     | 0/45     | 0/45     | 19/45    | 0/45     |    9.3% |
| streaming_llm   | 0/45     | 0/45     | 0/45     | 10/45    | 0/45     |    4.4% |
| evoke_discard   | 0/45     | 0/45     | 0/45     | 15/45    | 0/45     |    6.7% |
| evoke_breadcrumb | 0/45     | 0/45     | 0/45     | 15/45    | 0/45     |    6.7% |
| h2o             | 0/45     | 0/45     | 4/45     | 16/45    | 0/45     |    8.9% |
| snapkv          | 0/45     | 0/45     | 5/45     | 16/45    | 0/45     |    9.3% |
| infllm          | 38/45    | 41/45    | 17/45    | 15/45    | 36/45    |   65.3% |
| evoke_kv_restore | 33/45    | 42/45    | 13/45    | 4/45     | 36/45    |   56.9% |
| evoke_attention | 33/45    | 41/45    | 7/45     | 7/45     | 36/45    |   55.1% |
