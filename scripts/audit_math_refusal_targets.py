#!/usr/bin/env python3
"""
Audit a JSONL for math-refusal target correctness.

Fails (exit 1) if:
- A prompt matches a known "undefined" category but the target is not the expected template, OR
- The target is one of the known templates but the prompt is unclassified (meaning our regex needs updating).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


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

KNOWN_TARGETS = {T_UNDEF_REALS, T_DIV0, T_LOG, T_STD_ARITH, T_SINGULAR}


RE_MATRIX = re.compile(r"\binverse\b.*\[\[", re.IGNORECASE)

RE_DIV0 = re.compile(
    r"(?:(?:/|÷)\s*0\b|\\over\s*0\b|\bover\s+0\b|\bdivided\s+by\s+0\b)",
    re.IGNORECASE,
)

RE_LOG = re.compile(r"\b(?:ln|log(?:_[a-z0-9]+)?|logarithm)\b", re.IGNORECASE)
RE_NEGNUM = re.compile(r"(?<!\w)-\d+\b")
RE_ZERO = re.compile(r"\b(?:0|zero)\b", re.IGNORECASE)
RE_LOG_DOMAIN = re.compile(
    r"(?:(?:\\ln|\bln|\blog|log_)[^()\n]{0,30}\(\s*0\s*\)"
    r"|(?:\\ln|\bln|\blog|log_)[^()\n]{0,30}\(\s*-\s*\d"
    r"|\bnonpositive\b|\bnegative\b)",
    re.IGNORECASE,
)

RE_SQRTNEG = re.compile(
    r"(?:(?:\bsqrt\b|\\sqrt|√)\s*(?:\(|\{)?\s*-\s*[\d{]"
    r"|\bsquare\s+root\s+of\s*-\s*[\d{]"
    r"|\breal\s+square\s+root\s+of\s*\{n\})",
    re.IGNORECASE,
)

RE_FACTNEG = re.compile(r"(?:\(\s*-\s*\d+\s*\)|-\s*\d+)\s*!", re.IGNORECASE)
RE_ZZ = re.compile(r"\b0\s*\^\s*0\b", re.IGNORECASE)
RE_LIMIT = re.compile(r"\blim\b|\blimit\b", re.IGNORECASE)
RE_TRIG = re.compile(r"\b(?:acos|arccos|asin|arcsin|atanh)\b", re.IGNORECASE)


def infer_category(prompt: str) -> str | None:
    s = prompt
    if RE_MATRIX.search(s):
        return "matrix"
    if RE_DIV0.search(s):
        return "div0"
    if RE_LOG.search(s) and (
        RE_NEGNUM.search(s)
        or RE_ZERO.search(s)
        or "nonpositive" in s.lower()
        or "negative" in s.lower()
    ):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="JSONL to audit")
    ap.add_argument("--max", type=int, default=50, help="Max mismatches to print")
    args = ap.parse_args()

    path = Path(args.path)
    mismatches = 0
    by_cat = Counter()
    by_kind = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or obj.get("q") or "")
            target = obj.get("target") if "target" in obj else obj.get("out")

            cat = infer_category(prompt)

            # If the record is one of the known target templates, we enforce classification.
            if target in KNOWN_TARGETS and cat is None:
                mismatches += 1
                by_kind["unclassified_prompt"] += 1
                if mismatches <= args.max:
                    print(
                        f"{path}:{line_no} [unclassified_prompt] "
                        f"target={target!r} prompt={prompt!r}"
                    )
                continue

            if cat is None:
                continue

            by_cat[cat] += 1
            expected = EXPECTED[cat]
            if target != expected:
                mismatches += 1
                by_kind["wrong_target"] += 1
                if mismatches <= args.max:
                    print(
                        f"{path}:{line_no} [wrong_target] "
                        f"cat={cat} expected={expected!r} got={target!r} prompt={prompt!r}"
                    )

    if mismatches:
        print(f"FAIL: {mismatches} mismatches")
        print("by_kind:", dict(by_kind))
        print("by_category:", dict(by_cat))
        return 1

    print("PASS")
    print("by_category:", dict(by_cat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
