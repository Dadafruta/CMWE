#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BENCH_DIR="data/daily_bench_v4"
RUN_DIR="$(ls -td "${BENCH_DIR}"/* | head -n 1)"

# If caller didn't pass --out_csv, default to <run_dir>/eval.csv
if [[ " $* " != *" --out_csv "* ]]; then
  set -- --out_csv "${RUN_DIR}/eval.csv" "$@"
fi

exec python -m scripts.eval_daily_mathgate_bench_v4 --data_dir "${RUN_DIR}" "$@"
