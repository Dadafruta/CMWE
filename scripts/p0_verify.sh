#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Basic sanity: required files exist
for f in scripts/preflight.py scripts/run_gated_infer.py scripts/smoke.sh requirements.txt; do
  if [[ ! -f "$f" ]]; then
    echo "[p0] ERROR: missing $f" >&2
    exit 1
  fi
done

# Ensure venv exists
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# Install deps
source .venv/bin/activate
python -m pip install -U pip wheel setuptools >/dev/null
python -m pip install -r requirements.txt

# Compile + shell syntax check (silent on success)
python -m py_compile scripts/preflight.py scripts/run_gated_infer.py
bash -n scripts/smoke.sh

# Run checks
python scripts/preflight.py
bash scripts/smoke.sh

echo "[p0] OK"
