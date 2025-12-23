#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
RUN_DIR="${RUN_DIR:-/tmp/cmwe_smoke_$(date +%Y%m%d_%H%M%S)}"
TIMEOUT="${TIMEOUT:-180}"

echo "[smoke] RUN_DIR=${RUN_DIR}"
echo "[smoke] TIMEOUT=${TIMEOUT}"
mkdir -p "${RUN_DIR}/raw"

run_case () {
  local expected_route="$1"
  local expected_refused="$2"
  local th_math="$3"
  local th_cite="$4"
  local q="$5"

  # Wrapper prints EXACTLY one JSON object to stdout
  local out
  out="$("${PYTHON}" scripts/run_gated_infer.py \
    --run-dir "${RUN_DIR}" \
    --timeout "${TIMEOUT}" \
    --th_math "${th_math}" \
    --th_cite "${th_cite}" \
    --q "${q}"
  )"

  python - <<'PY' "${expected_route}" "${expected_refused}" "${out}"
import json, sys

expected_route = sys.argv[1]
expected_refused = sys.argv[2].lower() == "true"
s = sys.argv[3]

obj = json.loads(s)

missing = [k for k in ["route","risk","scale","refused","out","ok","returncode","raw_log"] if k not in obj]
if missing:
    print(f"[smoke] FAIL: missing keys: {missing}", file=sys.stderr)
    print(obj, file=sys.stderr)
    sys.exit(1)

if not obj["ok"]:
    print(f"[smoke] FAIL: ok != True (got {obj['ok']}); route={obj.get('route')}", file=sys.stderr)
    print(obj, file=sys.stderr)
    sys.exit(1)

if obj["route"] != expected_route:
    print(f"[smoke] FAIL: route mismatch got={obj['route']} expected={expected_route}", file=sys.stderr)
    print(obj, file=sys.stderr)
    sys.exit(1)

if bool(obj["refused"]) != expected_refused:
    print(f"[smoke] FAIL: refused mismatch got={obj['refused']} expected={expected_refused}", file=sys.stderr)
    print(obj, file=sys.stderr)
    sys.exit(1)

print(f"[smoke] OK: route={obj['route']} risk={obj['risk']} refused={obj['refused']}")
PY
}

# (1) Force base: thresholds high => risk below thresholds => base
run_case "base" "true"  "0.99" "0.99" "Evaluate the integral of x^2 from 0 to 1."

# (2) Force math_guard: th_math low => risk above threshold => math_guard
run_case "math_guard" "false" "0.10" "0.99" "Evaluate the integral of x^2 from 0 to 1."

# (3) Force citation_guard: th_cite low => risk above threshold => citation_guard
run_case "citation_guard" "true" "0.99" "0.10" "Find a DOI for the 2019 BERT paper and give a citation."

echo "[smoke] DONE"
echo "[smoke] raw logs in: ${RUN_DIR}/raw/"
