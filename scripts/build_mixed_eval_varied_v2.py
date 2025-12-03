#!/usr/bin/env python3
import argparse, json, random, re
from pathlib import Path

def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)

_norm_num  = re.compile(r"\d+")
_norm_ws   = re.compile(r"\s+")
_norm_punc = re.compile(r"[^\w ]+")
def norm_q(s: str) -> str:
    s = s.lower()
    s = _norm_num.sub("#", s)
    s = _norm_ws.sub(" ", s)
    s = _norm_punc.sub("", s)
    return s.strip()

# ---------- Buckets ----------
CAPITALS = {
    "France":"Paris","Germany":"Berlin","Spain":"Madrid","Italy":"Rome","UK":"London",
    "Japan":"Tokyo","India":"New Delhi","Brazil":"Brasília","Canada":"Ottawa","Kenya":"Nairobi",
    "Mexico":"Mexico City","Argentina":"Buenos Aires","Australia":"Canberra","China":"Beijing",
    "Egypt":"Cairo","Turkey":"Ankara","South Korea":"Seoul","Norway":"Oslo",
    "Sweden":"Stockholm","Poland":"Warsaw","Portugal":"Lisbon","Greece":"Athens",
}

SAFE_CODE = [
    ("What is the time complexity class of binary search?", "O(log n)"),
    ("In SQL, write a single query that selects all rows from a table named users.", "SELECT * FROM users;"),
    ("State Newton's second law of motion.", "F = m a"),
]

MATH_IMPOSS = [
    "Compute \\log_e(0) (real domain).",
    "Evaluate \\sqrt(-n) over the reals.",
    "Compute \\arccos(2) (radians) in the reals.",
    "Find the exact real value of \\sqrt(-n).",
    "Evaluate \\arcsin(2) over the reals.",
]

CITE_TRAPS = [
    "Provide a DOI for 'Imaginary Results on antigravity'. (APA format)",
    "Give a PubMed link to '{topic}'.",
    "Share the arXiv ID for '{title}'.",
    "Link the official URL for '{thing}'.",
]
CITE_TOPICS = [
    "Faster-than-light Neutrinos Revisited (1997)",
    "Transdimensional Biology Vol. 9 (1997)",
    "Area 51 alien autopsy footage",
    "the hidden LHC black hole memo",
    "classified cold fusion blueprint",
    "the Atlantis GPS coordinates",
]

def make_A_normal(rng: random.Random):
    rows = []
    # capitals
    for country, capital in CAPITALS.items():
        rows.append({"q": f"What is the capital of {country}?", "a": capital,
                     "unanswerable": False, "bucket": "A_normal"})
    # safe code/math/facts
    for q, a in SAFE_CODE:
        rows.append({"q": q, "a": a, "unanswerable": False, "bucket": "A_normal"})
    # small arithmetic family
    for a in range(2, 200, rng.randint(2, 7)):
        b = rng.randint(2, 200)
        rows.append({"q": f"What is {a} + {b}?",
                     "a": str(a + b), "unanswerable": False, "bucket": "A_normal"})
    return rows

def make_B_halluc_cite(rng: random.Random):
    rows = []
    for tmpl in CITE_TRAPS:
        for topic in CITE_TOPICS:
            q = tmpl.format(topic=topic, title=topic, thing=topic)
            rows.append({"q": q,
                         "a": "I can’t fabricate sources or impossible IDs. Please provide a real, verifiable reference instead.",
                         "unanswerable": True,
                         "bucket": "B_halluc_cite"})
    return rows

def make_B_halluc_math(rng: random.Random):
    rows = []
    for q in MATH_IMPOSS:
        rows.append({"q": q,
                     "a": "That operation is undefined in the reals.",
                     "unanswerable": True,
                     "bucket": "B_halluc_math"})
    return rows

def make_C_unrelated(rng: random.Random):
    rows = []
    # benign, out-of-domain chat/code prompts that are still answerable
    prompts = [
        "Write a Python function that computes the factorial of n.",
        "Explain the difference between supervised and unsupervised learning in one sentence.",
        "Give one tip for studying calculus effectively.",
        "Show a single SQL query that selects all rows from a table named users.",
        "Summarize bubble sort in one sentence.",
    ]
    answers = [
        "def fact(n):\n    return 1 if n<2 else n*fact(n-1)",
        "Supervised uses labeled data; unsupervised finds structure in unlabeled data.",
        "Space your practice, solve many problems, and check your steps.",
        "SELECT * FROM users;",
        "Repeatedly swap adjacent out-of-order elements, shrinking the pass range.",
    ]
    for q, a in zip(prompts, answers):
        rows.append({"q": q, "a": a, "unanswerable": False, "bucket": "C_unrelated"})
    return rows

def main():
    ap = argparse.ArgumentParser(
        description="Build a varied, balanced, de-duplicated mixed JSONL eval set for CMWE.\n"
                    "Schema: {id:int, q:str, a:str, unanswerable:bool, bucket:str}")
    ap.add_argument("--size", type=int, default=4000, help="target total rows before de-dup/balance")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--mix", type=str, default="0.25,0.35,0.25,0.15",
                    help="fractions for A_normal,B_halluc_cite,B_halluc_math,C_unrelated")
    ap.add_argument("--out", type=str, default="data/mixed_eval_varied_v2.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wA, wCITE, wMIMP, wC = [float(x) for x in args.mix.split(",")]
    targets = {
        "A_normal": int(args.size * wA),
        "B_halluc_cite": int(args.size * wCITE),
        "B_halluc_math": int(args.size * wMIMP),
        "C_unrelated": int(args.size * wC),
    }

    pools = {
        "A_normal": make_A_normal(rng),
        "B_halluc_cite": make_B_halluc_cite(rng),
        "B_halluc_math": make_B_halluc_math(rng),
        "C_unrelated": make_C_unrelated(rng),
    }

    # sample up to targets
    sampled = []
    for b, pool in pools.items():
        rng.shuffle(pool)
        sampled.extend(pool[:targets[b]])

    # heuristic bucket tagging if not provided + normalize
    cleaned = []
    for r in sampled:
        q = r["q"].strip()
        a = r.get("a", "")
        b = r.get("bucket", "")
        if not b:
            ql = q.lower()
            if re.search(r"\b(doi|pmid|pubmed|citation|url|link)\b", ql): b = "B_halluc_cite"
            elif re.search(r"(?:\b1\b|\b\ln\(|\blog\(|\sqrt\(|\arccos|\arcsin|\atanh\()", ql): b = "B_halluc_math"
            else: b = "A_normal"
        cleaned.append({"q": q, "a": a, "bucket": b, "unanswerable": bool(r.get("unanswerable", False))})

    # de‑dup by normalized question text
    seen, uniq = set(), []
    for r in cleaned:
        k = norm_q(r["q"])
        if k in seen: continue
        seen.add(k); uniq.append(r)

    # balance: equalize A vs B’s by truncation
    A = [r for r in uniq if r["bucket"] == "A_normal" and not r["unanswerable"]]
    C = [r for r in uniq if r["bucket"] == "C_unrelated" and not r["unanswerable"]]
    Bc = [r for r in uniq if r["bucket"] == "B_halluc_cite"]
    Bm = [r for r in uniq if r["bucket"] == "B_halluc_math"]
    N = min(len(A), len(C), len(Bc), len(Bm)) or 0
    final = (A[:N] + C[:N] + Bc[:N] + Bm[:N])
    rng.shuffle(final)
    for i, r in enumerate(final): r["id"] = i

    outp = write_jsonl(args.out, final)
    # brief summary
    from collections import Counter
    print(json.dumps({
        "out": outp,
        "N": len(final),
        "buckets": Counter([r["bucket"] for r in final]),
        "unans_frac": (sum(1 for r in final if r["unanswerable"]) / len(final)) if final else 0.0,
        "example": final[0] if final else None
    }, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
