#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${ROOT}/data/daily_bench_v4"
RUN_DIR="$(ls -td "${BENCH_DIR}"/* | head -n 1)"
exec python -m scripts.eval_daily_mathgate_bench_v4 --data_dir "${RUN_DIR}" "$@"
