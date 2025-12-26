"""Generate math data varied v2.

Run:
  python -m scripts.make_math_data_varied_v2 --help
"""

import json
import random
import argparse
import pathlib

pathlib.Path("data").mkdir(exist_ok=True)

REFUSALS = [
    "Division by zero is undefined.",
    "That calculation isn’t defined in standard arithmetic.",
    "I won’t fabricate impossible or undefined results.",
    "That quantity cannot be computed exactly.",
    "This request asks for a non-real or undefined value.",
    "That exceeds mathematical limits; I can’t produce a real answer.",
    "Log of a nonpositive number is undefined in the reals.",
    "The real square root of a negative number is undefined.",
    "There is no finite exact answer to that.",
]


def R():
    return random.choice(REFUSALS)


def row(q, r=None):
    return {"prompt": f"Q: {q}\nA:", "target": r or R()}


DIV0_T = [
    "What is {a}/0?",
    "Compute {a} divided by 0.",
    "Evaluate {a} ÷ 0.",
    "Give the exact value of {a} over zero.",
    "Find {a} divided by zero in the real numbers.",
]
SQRTNEG_T = [
    "Compute the square root of -{n} in real numbers.",
    "Evaluate √(-{n}) over the reals.",
    "Find the real square root of negative {n}.",
    "Give sqrt(-{n}) assuming real domain.",
]
LOGNP_T = [
    "Evaluate log_{b}({x}).",
    "Compute logarithm base {b} of {x}.",
    "Find log base {b} of {x} in the reals.",
]
TRIGBAD_T = [
    "Compute arccos({v}).",
    "Evaluate arcsin({v}).",
    "Find arctanh({v}).",
    "Give arccot({v}) in real numbers.",
]
HUGEFACT_T = [
    "Provide the exact decimal expansion of {n}!",
    "Write out {n}! in full digits.",
    "Give {n}! exactly (no approximation).",
]
IRRPREC_T = [
    "Show {d} digits of {c} starting at digit {k}.",
    "Provide {d} consecutive digits of {c} beginning at index {k}.",
    "Give a block of {d} digits from {c} starting at position {k}.",
]
COSMIC_T = [
    "Give the exact number of atoms in the observable universe.",
    "Provide an exact integer count of all atoms in the universe.",
    "State the precise number of atoms in existence.",
]
UNITBAD_T = [
    "Convert {val} lightyears^3 to kilometers exactly.",
    "Convert {val} joules to hours exactly.",
    "Convert one Planck length to miles as an exact rational.",
]
OTHERBAD_T = [
    "Solve 0·x = {c} for a real x.",
    "Compute ln(0).",
    "Find log base 10 of -{m}.",
    "Evaluate 0/0.",
    "Give a real value for ∞ - ∞.",
    "Compute a finite exact value for lim_{x→0} 1/x.",
]

IRR = ["π", "pi", "e", "√2", "sqrt(2)", "√3", "sqrt(3)"]
BASES = ["10", "2", "e"]
TRIG_V = ["2", "1.5", "-2", "-1.1", "1.0001", "-1.0001"]


def gen_div0(N):
    for _ in range(N):
        a = random.randint(1, 10**6)
        t = random.choice(DIV0_T)
        yield row(t.format(a=a))


def gen_sqrtneg(N):
    for _ in range(N):
        n = random.randint(1, 10**6)
        t = random.choice(SQRTNEG_T)
        yield row(t.format(n=n))


def gen_lognp(N):
    xs = [0] + [-(10**k) for k in range(1, 7)] + [-random.randint(1, 10**6)]
    for _ in range(N):
        x = random.choice(xs)
        b = random.choice(BASES)
        t = random.choice(LOGNP_T)
        yield row(t.format(b=b, x=x))


def gen_trigbad(N):
    for _ in range(N):
        v = random.choice(TRIG_V)
        t = random.choice(TRIGBAD_T)
        yield row(t.format(v=v))


def gen_hugefact(N):
    for _ in range(N):
        n = random.choice([5000, 10000, 50000, 100000, 500000, 10**6, 10**7])
        t = random.choice(HUGEFACT_T)
        yield row(t.format(n=n))


def gen_irrprec(N):
    for _ in range(N):
        c = random.choice(IRR)
        d = random.choice([20, 30, 40, 50, 100])
        k = random.choice([10**6, 10**8, 10**9, 10**10, 10**12])
        t = random.choice(IRRPREC_T)
        yield row(t.format(c=c, d=d, k=k))


def gen_cosmic(N):
    for _ in range(N):
        t = random.choice(COSMIC_T)
        yield row(t)


def gen_unitbad(N):
    for _ in range(N):
        val = random.choice([1, 3, 7, 12.5])
        t = random.choice(UNITBAD_T)
        yield row(t.format(val=val))


def gen_otherbad(N):
    for _ in range(N):
        c = random.randint(1, 999)
        m = random.randint(1, 10**6)
        t = random.choice(OTHERBAD_T)
        # Only format if placeholders exist; otherwise leave the string as-is
        if "{c}" in t or "{m}" in t:
            t = t.format(c=c, m=m)
        yield row(t)


def main():
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=5000)
    args = ap.parse_args()
    buckets = [
        (gen_div0, 0.18),
        (gen_sqrtneg, 0.15),
        (gen_lognp, 0.12),
        (gen_trigbad, 0.10),
        (gen_hugefact, 0.10),
        (gen_irrprec, 0.12),
        (gen_cosmic, 0.07),
        (gen_unitbad, 0.06),
        (gen_otherbad, 0.10),
    ]
    rows = []
    for fn, w in buckets:
        k = max(1, int(args.size * w))
        rows.extend(list(fn(k)))
    seen = set()
    uniq = []
    for ex in rows:
        if ex["prompt"] not in seen:
            uniq.append(ex)
            seen.add(ex["prompt"])
    random.shuffle(uniq)
    uniq = uniq[: args.size]
    with open("data/math_refusal_train.jsonl", "w") as f:
        for ex in uniq:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(uniq)} rows to data/math_refusal_train.jsonl")


if __name__ == "__main__":
    main()
