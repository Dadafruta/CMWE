#!/usr/bin/env python3
"""Generate holdout v2.

Run:
  python -m scripts.make_holdout_v2 --help
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Mapping, List


def load_jsonl(path: str | Path) -> List[dict]:
    p = Path(path)
    rows: List[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_balanced(rows: List[dict], n: int, rng: random.Random) -> List[dict]:
    ans = [r for r in rows if not r.get("unanswerable", False)]
    unans = [r for r in rows if r.get("unanswerable", False)]
    if not ans or not unans:
        return rng.sample(rows, k=min(n, len(rows)))
    half = n // 2
    take_ans = min(len(ans), half)
    take_unans = min(len(unans), n - take_ans)
    rng.shuffle(ans)
    rng.shuffle(unans)
    picked = ans[:take_ans] + unans[:take_unans]
    rng.shuffle(picked)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/mixed_eval_v2_full.jsonl")
    ap.add_argument("--out", default="data/mixed_eval_v2_holdout.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--balance", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.source)
    n = args.n
    if n <= 0 or n > len(rows):
        n = len(rows)

    if args.balance:
        selected = sample_balanced(rows, n, rng)
    else:
        selected = rng.sample(rows, k=n)

    write_jsonl(args.out, selected)
    n_ans = sum(not r.get("unanswerable", False) for r in selected)
    n_unans = sum(r.get("unanswerable", False) for r in selected)
    print(f"Wrote {len(selected)} rows to {args.out}")
    print(f"Answerable: {n_ans} | Unanswerable: {n_unans}")


if __name__ == "__main__":
    main()
