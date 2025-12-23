#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v rg >/dev/null 2>&1; then
  echo "[patch] rg (ripgrep) not found; skipping auto-patch" >&2
  exit 0
fi

# Only touch shell scripts under scripts/
mapfile -t files < <(rg -l --glob 'scripts/*.sh' 'python -m scripts\.gated_infer' scripts || true)

if (( ${#files[@]} == 0 )); then
  echo "[patch] OK: no scripts/*.sh calling scripts.gated_infer directly"
  exit 0
fi

for f in "${files[@]}"; do
  sed -i 's/python -m scripts\.gated_infer/python -m scripts.run_gated_infer/g' "$f"
  echo "[patch] updated: $f"
done

echo "[patch] DONE"
