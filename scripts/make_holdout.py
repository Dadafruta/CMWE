"""Create a deterministic holdout split from an evaluation JSONL file."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmwe.metrics_utils import load_jsonl, normalize_jsonl_rows


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_balanced(rows: List[Mapping], n: int, rng: random.Random) -> List[Mapping]:
    answerable = [r for r in rows if not r["unanswerable"]]
    unanswerable = [r for r in rows if r["unanswerable"]]
    if not answerable or not unanswerable:
        return rng.sample(rows, k=min(n, len(rows)))

    half = n // 2
    take_ans = min(len(answerable), half)
    take_unans = min(len(unanswerable), n - take_ans)
    rng.shuffle(answerable); rng.shuffle(unanswerable)
    picked = answerable[:take_ans] + unanswerable[:take_unans]
    rng.shuffle(picked)
    return picked


def build_holdout(source: Path, out: Path, n: int, seed: int, balance: bool) -> None:
    raw_rows = load_jsonl(source)
    rows = normalize_jsonl_rows(raw_rows)

    rng = random.Random(seed)
    if n <= 0 or n > len(rows):
        n = len(rows)

    if balance:
        selected = _sample_balanced(rows, n, rng)
    else:
        selected = rng.sample(rows, k=n)

    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, selected)

    print(
        f"Wrote {len(selected)} rows to {out} (balance={balance}, seed={seed})\n"
        f"Answerable: {sum(not r['unanswerable'] for r in selected)} | "
        f"Unanswerable: {sum(r['unanswerable'] for r in selected)}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/mixed_eval_v1_50.jsonl", help="Input JSONL file")
    ap.add_argument("--out", default="data/mixed_eval_v1_holdout.jsonl", help="Destination JSONL path")
    ap.add_argument("--n", type=int, default=500, help="Number of rows to sample (<= dataset size)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--balance", action="store_true", help="Approximate 50/50 answerable split")
    return ap.parse_args()


def main():
    args = parse_args()
    build_holdout(Path(args.source), Path(args.out), args.n, args.seed, args.balance)


if __name__ == "__main__":
    main()
