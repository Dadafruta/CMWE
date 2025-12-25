#!/usr/bin/env python3
"""
Fix mislabeled math-refusal targets in a JSONL by inferring the "undefined reason"
from the prompt and assigning a canonical target template.

Example:
  python3 scripts/fix_math_refusal_targets.py --in data/refusal_train_v3.jsonl --inplace
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Canonical targets (ASCII source; JSON output will escape unicode by default)
T_UNDEF_REALS = "The requested value is undefined in the reals."
T_DIV0 = "Division by zero is undefined."
T_LOG = "Logarithm is undefined at this input in \u211d."
T_STD_ARITH = "That calculation isn\u2019t defined in standard arithmetic."
T_SINGULAR = "Matrix is singular; inverse is undefined in \u211d."


EXPECTED: dict[str, str] = {
    "matrix": T_SINGULAR,
    "div0": T_DIV0,
    "log": T_LOG,
    "sqrt_neg": T_UNDEF_REALS,
    "trig": T_UNDEF_REALS,
    "fact_neg": T_STD_ARITH,
    "zero_zero": T_STD_ARITH,
    "limit": T_UNDEF_REALS,
}


# Prompt classifiers (heuristic, but designed to be stable + cheap)
RE_MATRIX = re.compile(r"\binverse\b.*\[\[", re.IGNORECASE)

RE_DIV0 = re.compile(
    r"(?:"  # any of:
    r"(?:/|÷)\s*0\b"  # a/0, a ÷ 0
    r"|\\over\s*0\b"  # TeX \over 0
    r"|\bover\s+0\b"  # "over 0"
    r"|\bdivided\s+by\s+0\b"
    r")",
    re.IGNORECASE,
)

RE_LOG = re.compile(r"(?:\\ln\b|\bln\b|\blog\b|log_)", re.IGNORECASE)
RE_LOG_DOMAIN = re.compile(
    r"(?:"  # any of:
    r"(?:\\ln|\bln|\blog|log_)[^()\n]{0,30}\(\s*0\s*\)"  # ln(0), log(0)
    r"|(?:\\ln|\bln|\blog|log_)[^()\n]{0,30}\(\s*-\s*\d"  # ln(-12), log(-12)
    r"|\bnonpositive\b"
    r"|\bnegative\b"
    r")",
    re.IGNORECASE,
)

# Matches:
# - sqrt(-123), sqrt( -123 ), \sqrt{-123}, √(-123)
# - "square root of -123"
# - template: "real square root of {n}" (your current unclassified cases)
RE_SQRTNEG = re.compile(
    r"(?:"  # any of:
    r"(?:\bsqrt\b|\\sqrt|√)\s*(?:\(|\{)?\s*-\s*[\d{]"
    r"|\bsquare\s+root\s+of\s*-\s*[\d{]"
    r"|\breal\s+square\s+root\s+of\s*\{n\}"
    r")",
    re.IGNORECASE,
)

RE_FACTNEG = re.compile(r"(?:\(-\s*\d+\)|-\s*\d+)\s*!\b", re.IGNORECASE)
RE_ZZ = re.compile(r"\b0\s*\^\s*0\b", re.IGNORECASE)
RE_LIMIT = re.compile(r"\blim\b|\blimit\b", re.IGNORECASE)
RE_TRIG = re.compile(r"\b(?:acos|arccos|asin|arcsin|atanh)\b", re.IGNORECASE)


def infer_category(prompt: str) -> str | None:
    s = prompt
    if RE_MATRIX.search(s):
        return "matrix"
    if RE_DIV0.search(s):
        return "div0"
    if RE_LOG.search(s) and RE_LOG_DOMAIN.search(s):
        return "log"
    if RE_SQRTNEG.search(s):
        return "sqrt_neg"
    if RE_FACTNEG.search(s):
        return "fact_neg"
    if RE_ZZ.search(s):
        return "zero_zero"
    if RE_LIMIT.search(s):
        return "limit"
    if RE_TRIG.search(s):
        return "trig"
    return None


def _get_prompt_and_target_keys(obj: dict[str, Any]) -> tuple[str, str]:
    # Be tolerant to minor schema variants.
    prompt_key = "prompt" if "prompt" in obj else ("q" if "q" in obj else "prompt")
    target_key = "target" if "target" in obj else ("out" if "out" in obj else "target")
    return prompt_key, target_key


def fix_file(in_path: Path, out_path: Path) -> int:
    changed = 0
    by_category = Counter()
    total = 0

    with (
        in_path.open("r", encoding="utf-8") as fin,
        out_path.open("w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            total += 1
            obj = json.loads(line)
            prompt_key, target_key = _get_prompt_and_target_keys(obj)

            prompt = str(obj.get(prompt_key, ""))
            cat = infer_category(prompt)
            if cat is None:
                # Leave untouched
                fout.write(json.dumps(obj) + "\n")
                continue

            by_category[cat] += 1
            expected = EXPECTED[cat]
            if obj.get(target_key) != expected:
                obj[target_key] = expected
                changed += 1

            fout.write(json.dumps(obj) + "\n")

    print(f"wrote: {out_path}")
    print(f"total: {total}")
    print(f"changed: {changed}")
    print("by_category:", dict(by_category))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    ap.add_argument(
        "--out", dest="out_path", default=None, help="Output JSONL path (optional)"
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Modify the input file in-place (writes via a temp file).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)

    if args.inplace:
        tmp = in_path.with_suffix(in_path.suffix + ".tmp")
        rc = fix_file(in_path, tmp)
        tmp.replace(in_path)
        return rc

    out_path = Path(args.out_path) if args.out_path else Path("-")
    if str(out_path) == "-":
        # stdout mode
        # (not needed for your use-case; keep simple)
        print("ERROR: use --inplace or --out <path>", file=sys.stderr)
        return 2

    return fix_file(in_path, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
