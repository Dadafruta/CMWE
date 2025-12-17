#!/usr/bin/env python3
"""Filter eval no math v2.

Run:
  python -m scripts.filter_eval_no_math_v2 --help
"""

from __future__ import annotations
import json
from pathlib import Path

SRC = Path("data/mixed_eval_v2_holdout.jsonl")
OUT = Path("data/mixed_eval_v2_holdout_nomath.jsonl")

rows = [
    json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()
]


def keep_bucket(b: str) -> bool:
    if b.startswith("A_normal_math"):
        return False
    if b.startswith("B_hallucination_bait_math"):
        return False
    return True


filtered = [r for r in rows if keep_bucket(r.get("bucket", ""))]

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in filtered:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"kept {len(filtered)} / {len(rows)} rows")
