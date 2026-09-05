Trajectory-based discard-regime eviction benchmark, run via `scripts/trajectory_bench.py`
against 138 real SWE-agent trajectories (`data/trajectories/swe_agent/`). Full narrative
and reasoning: wiki decision pages `benchmark-regime-split-and-repo-correction` and
`evoke-attention-killed-jlens-untested` (product-ideas project).

## Files, in the order they were produced

- `sweep_3b_original_buggy_scorer.txt` - Qwen2.5-3B-Instruct. First full sweep
  (no_eviction, recency, h2o, snapkv, evoke_attention), discard + recompute regimes.
  Uses the original `score_prediction`, which substring-matches the model's whole
  response instead of just its issued command (see the fix below).
- `jlens_7b_buggy_scorer.txt` - Qwen2.5-7B-Instruct. `jlens` wired into `POLICIES` and
  run alone (discard regime) plus a no_eviction x2 "ceiling" check, still under the
  buggy scorer. jlens appeared to beat both baselines by 34-44%.
- `spotcheck_10traj_scorer_ab_test.txt` / `spotcheck_10traj_predictions.jsonl` - Qwen2.5-7B,
  10-trajectory no_eviction-vs-jlens run with per-hibernation prediction dumping
  (`EVOKE_DUMP_PREDICTIONS`), used to hand-read the actual generated text and check
  whether jlens's lead was real or a scoring artifact. Verdict: partly artifact.
  SWE-agent's environment echoes `(Open file: X)` boilerplate after most commands,
  which the old whole-response substring match picked up regardless of what the
  model actually did.
- `sweep_7b_corrected_scorer.txt` / `sweep_7b_corrected_scorer_predictions.jsonl` -
  Qwen2.5-7B, all five policies (recency, no_eviction, h2o, snapkv, jlens) plus the
  ceiling, full 138 trajectories, using the fixed scorer (`extract_action()` in
  `trajectory_bench.py`, isolates the model's first fenced code block instead of
  substring-matching the whole response).

## The scorer fix

`score_prediction` originally checked whether the ground-truth tool name/target/args
strings appeared anywhere in the model's raw generated text. `extract_action()` instead
pulls the fenced code block's language tag plus its first content line, i.e. the command
the model actually issued, and scores against that alone. This removes the boilerplate-
echo false positives found in the spot-check.

## Final numbers (target-tier next-action agreement, discard regime, 138 trajectories)

All five rows below are one run, same model (Qwen2.5-7B-Instruct), same scorer,
directly comparable to each other:

| Policy | Score |
|---|---|
| recency | 0.023 |
| no_eviction | 0.035 |
| h2o | 0.065 |
| snapkv | 0.071 |
| jlens | 0.068 |
| self-agreement ceiling (no_eviction, 2 seeds) | 0.035 / 0.035 |

jlens ties/marginally trails snapkv, the best baseline.

## A caveat found while archiving this

`sweep_3b_original_buggy_scorer.txt`'s numbers (recency 0.005, no_eviction 0.009,
h2o 0.068, snapkv 0.073, evoke_attention 0.068) were run on the 3B model. The wiki's
original `evoke_attention` kill decision, and this session's early "old vs corrected
scorer" comparison, cited those numbers as if they were on the same model as the 7B
jlens/h2o/snapkv numbers. They are not: model size changed between those two sweeps in
addition to the scorer fix. The one clean, single-variable comparison across the whole
history here is jlens on 7B, same model both times: 0.098 (buggy scorer) versus 0.068
(fixed scorer). The five-policy table above is the only fully controlled comparison
(one model, one scorer, one run) and is what the kill/validate conclusion rests on.
