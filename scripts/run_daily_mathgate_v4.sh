#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="data/daily_bench_v4/${STAMP}"
LOG="logs/daily_mathgate_v4_${STAMP}.log"

mkdir -p "$OUTDIR" logs

python -m scripts.build_daily_mathgate_bench_v4 \
  --out_dir "$OUTDIR" \
  --n_answer 50000 \
  --n_refuse 50000 \
  --n_nonmath 50000 \
  --n_unsupported 20000

python -m scripts.eval_daily_mathgate_bench_v4 \
  --data_dir "$OUTDIR" \
  --out_csv "$OUTDIR/eval.csv" |& tee "$LOG"

echo "OK: $OUTDIR"
echo "LOG: $LOG"
