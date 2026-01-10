# Reproducing v2_holdout results (Phase 1 artifact)

From repo root:

## Full run (paper numbers)
RUN_FULL=1 FULL_LIMIT=180 bash last_mile.sh

## Expected outputs
- logs/eval_base_like_v2_holdout.csv
- logs/eval_gated_mixed_v2_holdout.csv
- logs/eval_guard_always_v2_holdout.csv
- results/v2_holdout_table.md

## Table-only (if logs already exist)
python scripts/summarize_v2_holdout_results.py \
  --base_like logs/eval_base_like_v2_holdout.csv \
  --cmwe logs/eval_gated_mixed_v2_holdout.csv \
  --always_guard logs/eval_guard_always_v2_holdout.csv \
  --out results/v2_holdout_table.md
