# NIAH per-cell tail latency (qwen25_7b_snapkv_infllm)

Each cell aggregates over the (needle, depth) combinations: 25 cells per
(budget, strategy) for the standard 5-needle x 5-depth grid. p99 over
25 samples is the 25th-from-best ranked value; p95 is the second-worst
rounded; report p50 / p95 / p99 alongside the mean reviewers asked for.

| budget | strategy        |  n |  mean | p50 | p95 | p99 | max |
|-------:|-----------------|---:|------:|----:|----:|----:|----:|
|    512 | evoke_attention | 25 |  2.36 | 2.36 | 2.37 | 2.38 | 2.39 |
|    512 | evoke_breadcrumb | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
|    512 | evoke_discard   | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
|    512 | evoke_kv_restore | 25 |  2.36 | 2.36 | 2.37 | 2.37 | 2.37 |
|    512 | h2o             | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
|    512 | infllm          | 25 |  2.19 | 2.19 | 2.20 | 2.26 | 2.28 |
|    512 | recency         | 25 |  2.31 | 2.30 | 2.32 | 2.36 | 2.37 |
|    512 | snapkv          | 25 |  2.31 | 2.32 | 2.33 | 2.35 | 2.36 |
|    512 | streaming_llm   | 25 |  2.31 | 2.31 | 2.32 | 2.34 | 2.34 |
|   1024 | evoke_attention | 25 |  2.36 | 2.35 | 2.37 | 2.37 | 2.37 |
|   1024 | evoke_breadcrumb | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
|   1024 | evoke_discard   | 25 |  2.31 | 2.30 | 2.32 | 2.32 | 2.32 |
|   1024 | evoke_kv_restore | 25 |  2.36 | 2.36 | 2.37 | 2.37 | 2.37 |
|   1024 | h2o             | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
|   1024 | infllm          | 25 |  2.20 | 2.20 | 2.21 | 2.22 | 2.22 |
|   1024 | recency         | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
|   1024 | snapkv          | 25 |  2.31 | 2.32 | 2.33 | 2.33 | 2.33 |
|   1024 | streaming_llm   | 25 |  2.31 | 2.30 | 2.32 | 2.32 | 2.32 |
|   2048 | evoke_attention | 25 |  2.36 | 2.36 | 2.37 | 2.37 | 2.37 |
|   2048 | evoke_breadcrumb | 25 |  2.31 | 2.30 | 2.32 | 2.32 | 2.32 |
|   2048 | evoke_discard   | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
|   2048 | evoke_kv_restore | 25 |  2.35 | 2.35 | 2.37 | 2.37 | 2.37 |
|   2048 | h2o             | 25 |  2.31 | 2.31 | 2.32 | 2.33 | 2.33 |
|   2048 | infllm          | 25 |  2.35 | 2.36 | 2.36 | 2.37 | 2.37 |
|   2048 | recency         | 25 |  2.30 | 2.30 | 2.32 | 2.32 | 2.32 |
|   2048 | snapkv          | 25 |  2.32 | 2.32 | 2.33 | 2.33 | 2.33 |
|   2048 | streaming_llm   | 25 |  2.31 | 2.31 | 2.32 | 2.32 | 2.32 |
