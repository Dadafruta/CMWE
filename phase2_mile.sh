#!/usr/bin/env bash
set -euo pipefail

# disable bash history expansion so things like "!r" never explode
set +H 2>/dev/null || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p logs results

# Defaults (override via env)
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
FULL_LIMIT="${FULL_LIMIT:-200}"
RUN_FULL="${RUN_FULL:-0}"
LIMIT="${LIMIT:-$SMOKE_LIMIT}"
if [[ "$RUN_FULL" == "1" ]]; then
  LIMIT="$FULL_LIMIT"
fi

DATA="${DATA:-data/math_eval_v1.jsonl}"
OUT_TABLE="${OUT_TABLE:-results/math_eval_v1_table.md}"

PATCH_TAG="PATCH_CSV_WRITE_V2_OUTCSV"
if ! grep -q "$PATCH_TAG" scripts/eval_v2_holdout.py; then
  echo "ERROR: expected CSV write patch tag missing in scripts/eval_v2_holdout.py"
  echo "Run the patch block first (adds: $PATCH_TAG)."
  exit 1
fi

export PYTHONPATH="$ROOT"

run_mode () {
  local mode="$1"
  local out_csv="$2"

  echo
  echo "=== RUN ${mode} (limit=${LIMIT}) ==="
  rm -f "$out_csv" 2>/dev/null || true

  python -m scripts.eval_v2_holdout \
    --mode "$mode" \
    --limit "$LIMIT" \
    --data "$DATA" \
    --out_csv "$out_csv"

  if [[ ! -s "$out_csv" ]]; then
    echo "ERROR: expected CSV missing or empty: $out_csv" >&2
    echo "Try: ls -lah logs | tail -n 50" >&2
    exit 1
  fi
}

run_mode base_like     "logs/eval_base_like_math_eval_v1.csv"
run_mode cmwe          "logs/eval_gated_math_eval_v1.csv"
run_mode always_guard  "logs/eval_guard_always_math_eval_v1.csv"

echo
echo "=== SUMMARIZE (paper table) ==="
python scripts/summarize_v2_holdout_results.py \
  --base_like "logs/eval_base_like_math_eval_v1.csv" \
  --cmwe "logs/eval_gated_math_eval_v1.csv" \
  --always_guard "logs/eval_guard_always_math_eval_v1.csv" \
  --out "$OUT_TABLE"

echo
echo "=== VERIFY ==="
ls -lh logs/*math_eval_v1*.csv
ls -lh "$OUT_TABLE"
echo
echo "DONE."
