#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _arith(rng: random.Random) -> tuple[str, str, str]:
    a = rng.randint(-200, 200)
    b = rng.randint(-200, 200)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    else:
        ans = a * b
    q = f"Solve this math problem. Compute: {a} {op} {b}. Answer with only the integer."
    return "A_math_arith", q, str(ans)


def _ops(rng: random.Random) -> tuple[str, str, str]:
    a = rng.randint(-50, 50)
    b = rng.randint(-50, 50)
    c = rng.randint(-50, 50)
    form = rng.choice(
        [
            ("({a} + {b}) * {c}", lambda a, b, c: (a + b) * c),
            ("{a} + ({b} * {c})", lambda a, b, c: a + (b * c)),
            ("({a} - {b}) * {c}", lambda a, b, c: (a - b) * c),
            ("{a} - ({b} * {c})", lambda a, b, c: a - (b * c)),
        ]
    )
    expr = form[0].format(a=a, b=b, c=c)
    ans = form[1](a, b, c)
    q = f"Solve this math problem. Evaluate: {expr}. Answer with only the integer."
    return "A_math_ops", q, str(ans)


def _linear(rng: random.Random) -> tuple[str, str, str]:
    # Make integer-solution linear equations: kx + b = y
    k = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
    x = rng.randint(-50, 50)
    b = rng.randint(-100, 100)
    y = k * x + b
    q = f"Solve this math problem. Solve for x: {k}x + {b} = {y}. Answer with only x as an integer."
    return "A_math_linear", q, str(x)


def _sqrt(rng: random.Random) -> tuple[str, str, str]:
    n = rng.randint(0, 30)
    sq = n * n
    q = f"Solve this math problem. What is sqrt({sq})? Answer with only the integer."
    return "A_math_sqrt", q, str(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/math_eval_v1.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    makers = [_arith, _ops, _linear, _sqrt]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.n):
        bucket, q, gold = rng.choice(makers)(rng)
        rows.append({"i": i, "bucket": bucket, "q": q, "gold": gold})

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
