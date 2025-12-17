"""Generate math varied safe.

Run:
  python -m scripts.make_math_varied_safe --help
"""

import json, random, argparse, pathlib, math

pathlib.Path("data").mkdir(exist_ok=True)

REFUSE = [
    "Division by zero is undefined.",
    "That calculation isn’t defined in standard arithmetic.",
    "The requested value is undefined in the reals.",
    "I won’t fabricate impossible or undefined results.",
]


def r():
    return random.choice(REFUSE)


def row(q, a=None):
    return {"prompt": f"Q: {q}\nA:", "target": a if a is not None else r()}


def risky(N):
    bases = ["10", "2", "e"]
    irr = ["π", "e", "√2", "√3"]
    X = []
    for _ in range(N):
        t = random.choice(
            [
                "div0",
                "sqrtneg",
                "lognp",
                "invtrig",
                "hugefact",
                "irrblock",
                "units",
                "atoms",
            ]
        )
        if t == "div0":
            a = random.randint(1, 10**9)
            X.append(row(f"Compute {a} divided by 0."))
        elif t == "sqrtneg":
            n = random.randint(1, 10**9)
            X.append(row(f"Find the real square root of -{n}."))
        elif t == "lognp":
            b = random.choice(bases)
            x = random.choice([0, -(10 ** random.randint(1, 7))])
            X.append(row(f"Evaluate log_{b}({x})."))
        elif t == "invtrig":
            v = random.choice(["2", "1.5", "-1.1", "-2"])
            fn = random.choice(["arccos", "arcsin", "arctanh"])
            X.append(row(f"Compute {fn}({v})."))
        elif t == "hugefact":
            n = random.choice([50000, 100000, 500000, 10**6])
            X.append(row(f"Give the exact decimal expansion of {n}!"))
        elif t == "irrblock":
            k = random.choice([10**6, 10**8, 10**10, 10**12])
            d = random.choice([30, 50, 100])
            c = random.choice(irr)
            X.append(row(f"Show {d} digits of {c} starting at digit {k}."))
        elif t == "units":
            v = random.choice([1, 3, 7, 12.5])
            X.append(row(f"Convert {v} lightyears^3 to kilometers exactly."))
        elif t == "atoms":
            X.append(row("Give the exact number of atoms in the observable universe."))
    return X


def safe(N):
    X = []
    for _ in range(N):
        t = random.choice(["arith", "sqrtpos", "logpos"])
        if t == "arith":
            a, b = random.randint(1, 9999), random.randint(1, 9999)
            X.append(row(f"What is {a}+{b}?", str(a + b)))
        elif t == "sqrtpos":
            n = random.choice([1, 4, 9, 16, 25, 36, 49, 64, 81])
            X.append(row(f"Square root of {n}?", str(int(n**0.5))))
        elif t == "logpos":
            n = random.choice([1, 10, 100, 1000])
            X.append(
                row(f"Evaluate log_10({n}).", str({1: 0, 10: 1, 100: 2, 1000: 3}[n]))
            )
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risky", type=int, default=12000)
    ap.add_argument("--safe", type=int, default=6000)
    ap.add_argument("--out", default="data/math_refusal_train.jsonl")
    args = ap.parse_args()
    rows = risky(args.risky) + safe(args.safe)
    seen = set()
    uniq = []
    for ex in rows:
        if ex["prompt"] not in seen:
            uniq.append(ex)
            seen.add(ex["prompt"])
    random.shuffle(uniq)
    open(args.out, "w").write("\n".join(map(json.dumps, uniq)) + "\n")
    print("Wrote", len(uniq), "rows to", args.out)


if __name__ == "__main__":
    main()
