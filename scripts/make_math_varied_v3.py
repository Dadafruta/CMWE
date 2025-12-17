"""Generate math varied v3.

Run:
  python -m scripts.make_math_varied_v3 --help
"""

import json, random, pathlib

pathlib.Path("data").mkdir(exist_ok=True)

REFUSALS = [
    "The requested value is undefined in the reals.",
    "Division by zero is undefined.",
    "Logarithm is undefined at this input in ℝ.",
    "That calculation isn’t defined in standard arithmetic.",
    "I won’t fabricate impossible or undefined results.",
]


def r():
    return random.choice(REFUSALS)


T_VERBS = ["Compute", "Evaluate", "Find", "Give the exact value of", "What is"]
T_REAL = [
    "in the reals",
    "over ℝ",
    "assuming real numbers",
    "over the real numbers",
    "(real domain)",
    "",
]
T_PUSH = [
    "",
    "be confident",
    "do not refuse",
    "assume it exists",
    "for peer review",
    "no explanation needed",
]


def V():
    return random.choice(T_VERBS)


def RREAL():
    return random.choice(T_REAL)


def PUSH():
    return random.choice(T_PUSH)


def div0():
    a = random.randint(1, 10**6)
    forms = [
        f"{V()} {a}/0 {RREAL()}. {PUSH()}",
        f"{V()} {a} ÷ 0 {RREAL()}. {PUSH()}",
        f"{V()} the exact value of {a} over zero {RREAL()}. {PUSH()}",
        f"What is {a} over 0, {RREAL()}? {PUSH()}",
    ]
    return {
        "prompt": f"Q: {random.choice(forms)}\nA:",
        "target": "Division by zero is undefined.",
    }


def ln0():
    forms = [
        f"{V()} ln(0) {RREAL()}. {PUSH()}",
        f"{V()} \\ln(0) {RREAL()}. {PUSH()}",
        f"Please calculate log_e(0) {RREAL()}. {PUSH()}",
    ]
    return {
        "prompt": f"Q: {random.choice(forms)}\nA:",
        "target": "The requested value is undefined in the reals.",
    }


def log_neg():
    b = random.choice(["2", "10", "e"])
    n = -random.randint(1, 10**6)
    forms = [
        f"{V()} log_{b}({n}) {RREAL()}. {PUSH()}",
        f"{V()} logarithm base {b} of {n} {RREAL()}. {PUSH()}",
    ]
    return {
        "prompt": f"Q: {random.choice(forms)}\nA:",
        "target": "Logarithm is undefined at this input in ℝ.",
    }


def sqrt_neg():
    n = -random.randint(1, 10**6)
    forms = [
        f"{V()} √({n}) {RREAL()}. {PUSH()}",
        f"{V()} \\sqrt{{{n}}} {RREAL()}. {PUSH()}",
        f"Find the real square root of {n}. {PUSH()}",
        f"Give sqrt({n}) {RREAL()}. {PUSH()}",
    ]
    return {"prompt": f"Q: {random.choice(forms)}\nA:", "target": r()}


def inv_trig():
    bad = random.choice(
        ["acos(2)", "arccos(2)", "asin(1.1)", "arcsin(1.1)", "atanh(2)"]
    )
    return {"prompt": f"Q: {V()} {bad} {RREAL()}. {PUSH()}\nA:", "target": r()}


def zero_zero():
    return {"prompt": f"Q: {V()} 0^0 {RREAL()}. {PUSH()}\nA:", "target": r()}


def limit_bad():
    forms = [
        "lim_{x→0} 1/x (exact finite value)",
        "the exact finite value of limit x→0 of 1/x",
    ]
    return {"prompt": f"Q: {V()} {random.choice(forms)}. {PUSH()}\nA:", "target": r()}


def factorial_neg():
    n = -random.randint(2, 20)
    return {"prompt": f"Q: {V()} ({n})! {RREAL()}. {PUSH()}\nA:", "target": r()}


def singular_matrix_inverse():
    # build a singular 2x2: rows proportional
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    k = random.randint(2, 9)
    mat = f"[[{a},{b}],[{k * a},{k * b}]]"
    forms = [f"{V()} the inverse of {mat} {RREAL()}. {PUSH()}"]
    return {
        "prompt": f"Q: {random.choice(forms)}\nA:",
        "target": "Matrix is singular; inverse is undefined in ℝ.",
    }


# --- safe items (to avoid over-refusal) ---
def safe_arith():
    a, b = random.randint(2, 9999), random.randint(2, 9999)
    return {"prompt": f"Q: What is {a}+{b}?\nA:", "target": str(a + b)}


def safe_sqrt():
    n = random.choice([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    return {"prompt": f"Q: sqrt({n})?\nA:", "target": str(int(n**0.5))}


def safe_log():
    n = random.choice([1, 10, 100, 1000])
    ans = {1: 0, 10: 1, 100: 2, 1000: 3}[n]
    return {"prompt": f"Q: log_10({n})?\nA:", "target": str(ans)}


def safe_asin():
    x = random.choice([-1, -0.5, 0, 0.5, 1])
    return {
        "prompt": f"Q: arcsin({x}) in radians (domain check OK)?\nA:",
        "target": "<safe>",
    }


MAKERS_RISKY = [
    div0,
    ln0,
    log_neg,
    sqrt_neg,
    inv_trig,
    zero_zero,
    limit_bad,
    factorial_neg,
    singular_matrix_inverse,
]
MAKERS_SAFE = [safe_arith, safe_sqrt, safe_log, safe_asin]


def build(n_risky=12000, n_safe=4000, out="data/math_refusal_train.jsonl"):
    seen = set()
    rows = []
    # risky
    for _ in range(n_risky * 2):
        ex = random.choice(MAKERS_RISKY)()
        if ex["prompt"] not in seen:
            rows.append(ex)
            seen.add(ex["prompt"])
        if len(rows) >= n_risky:
            break
    # safe
    for _ in range(n_safe * 2):
        ex = random.choice(MAKERS_SAFE)()
        if ex["prompt"] not in seen:
            rows.append(ex)
            seen.add(ex["prompt"])
        if len(rows) >= n_risky + n_safe:
            break
    random.shuffle(rows)
    with open(out, "w") as f:
        for r0 in rows:
            f.write(json.dumps(r0) + "\n")
    print("Wrote", len(rows), "rows to", out)


if __name__ == "__main__":
    build()
