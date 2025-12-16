#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Require v1 pools (these fix the 5-question duplication in nonmath/unsupported)
test -s data/daily_v1_nonmath.jsonl
test -s data/daily_v1_unsupported.jsonl

# 1) Generate a fresh run
bash scripts/run_daily_mathgate_v4.sh

# 2) Patch newest run's nonmath/unsupported to be unique (backs up originals)
python scripts/fix_latest_daily_bench_v4_nonmath_unsupported.py

# 3) Evaluate newest run
bash scripts/eval_latest_daily_mathgate_v4.sh

# 4) Print newest log summary lines
LOG="$(ls -t logs/daily_mathgate_v4_*.log | head -n 1)"
echo "LOG=${LOG}"
grep -nE '^(=== SUMMARY|=== CONFUSION|route_counts=|tool_coverage=|accuracy(_given_tool)?=|refused_rate=|false_(tool|pass|trigger)_rate=|top_reasons=)' "${LOG}" || true
