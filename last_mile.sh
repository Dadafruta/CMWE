#!/usr/bin/env bash
set -euo pipefail
# disable bash history expansion so things like "!r" never explode
set +H 2>/dev/null || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# defaults
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
FULL_LIMIT="${FULL_LIMIT:-180}"
RUN_FULL="${RUN_FULL:-0}"
LIMIT="${LIMIT:-$SMOKE_LIMIT}"
if [[ "$RUN_FULL" == "1" ]]; then
  LIMIT="$FULL_LIMIT"
fi

# sanity: ensure the decode patch is actually present
if ! grep -q "PATCH_GEN_ROBUST_DECODE_ONLY_NEW_TOKENS" scripts/eval_v2_holdout.py; then
  echo "ERROR: expected patch tag missing in scripts/eval_v2_holdout.py"
  echo "Run your patcher or rebase to the commit that added it."
  exit 1
fi

run_mode() {
  local mode="$1"
  echo
  echo "=== RUN ${mode} (limit=${LIMIT}) ==="
  python -m scripts.eval_v2_holdout --mode "$mode" --limit "$LIMIT"
}

run_mode base_like
run_mode cmwe
run_mode always_guard

# If logs landed in /artifacts/logs, move them into ./logs
if [[ -d /artifacts/logs ]]; then
  mkdir -p logs
  shopt -s nullglob
  for f in /artifacts/logs/*v2_holdout*.csv; do
    mv -v "$f" "logs/$(basename "$f")"
  done
  shopt -u nullglob
fi

echo
echo "=== VALIDATE ==="
python -m scripts.validate_v2_holdout_logs

echo
echo "=== SUMMARIZE (paper table) ==="
if [[ -f scripts/summarize_v2_holdout_results.py ]]; then
  mkdir -p results
  python scripts/summarize_v2_holdout_results.py \
    --base_like logs/eval_base_like_v2_holdout.csv \
    --cmwe logs/eval_gated_mixed_v2_holdout.csv \
    --always_guard logs/eval_guard_always_v2_holdout.csv \
    --out results/v2_holdout_table.md
  echo "Wrote: results/v2_holdout_table.md"
else
  echo "NOTE: scripts/summarize_v2_holdout_results.py not found; skipping."
fi

echo
echo "=== QUICK LOG CHECKS (should never crash) ==="
python - <<'PY'
import pandas as pd
from pathlib import Path

paths = [
    ("base_like", "logs/eval_base_like_v2_holdout.csv"),
    ("cmwe", "logs/eval_gated_mixed_v2_holdout.csv"),
    ("always_guard", "logs/eval_guard_always_v2_holdout.csv"),
]

def safe_mean(x):
    try:
        return float(x.mean())
    except Exception:
        return float("nan")

for name, p in paths:
    p = Path(p)
    if not p.exists():
        print(f"{name}: MISSING {p}")
        continue

    df = pd.read_csv(p, keep_default_na=False)

    if "out" not in df.columns:
        print(f"{name}: missing 'out' column; cols={list(df.columns)}")
        continue

    # Fix: some logs may not have 'i' — create it so downstream printing never crashes.
    if "i" not in df.columns:
        df.insert(0, "i", range(len(df)))

    out = df["out"].astype(str)
    lens = out.str.len()
    mx = int(lens.max()) if len(lens) else 0

    empty_rate = safe_mean(lens == 0)
    unk_any_rate = safe_mean(out.str.contains("<unk>", regex=False))
    unique_out_rate = float(out.nunique()) / max(1, len(out))
    hit_max_len_rate = safe_mean(lens == mx) if mx else float("nan")

    print(f"\n== {name} ({p}) ==")
    print("rows:", len(df))
    print("empty_rate:", round(empty_rate, 4))
    print("unk_any_rate:", round(unk_any_rate, 4))
    print("unique_out_rate:", round(unique_out_rate, 4))
    print("len min/median/max:", int(lens.min()), float(lens.median()), int(lens.max()))
    print("hit_max_len_rate:", round(hit_max_len_rate, 4), f"(max_len={mx})")

    # show a few <unk> examples if any
    bad = df[out.str.contains("<unk>", regex=False)]
    if len(bad):
        cols = [c for c in ["i", "route", "q", "out"] if c in bad.columns]
        bad = bad[cols].head(3)
        print("first <unk> examples:")
        for _, r in bad.iterrows():
            q = str(r.get("q", ""))[:90]
            o = str(r.get("out", ""))[:90]
            print(f"  i={r.get('i')} route={r.get('route','')}  {q!r} -> {o!r}")
PY

echo
echo "DONE."
