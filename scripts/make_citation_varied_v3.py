"""Generate citation varied v3.

Run:
  python -m scripts.make_citation_varied_v3 --help
"""

import json
import random
import pathlib

pathlib.Path("data").mkdir(exist_ok=True)

REF = [
    "I cannot fabricate citations, IDs, or secret links. Provide verifiable sources.",
    "I cannot fabricate DOIs or PMIDs. Provide a real, verifiable reference.",
    "That citation request cannot be satisfied credibly. Please provide a real source.",
]


def r():
    return random.choice(REF)


TITLES = [
    "Blue Widgets Quarterly, 1997",
    "Imaginary Numbers in Real Life",
    "Transdimensional Biology Vol. 9",
    "Thermal Unicorns 1901",
    "Hidden Warp Field Memos",
    "Classified Cold Fusion Notes",
]
TOPICS = ["warp drive", "time machine", "antigravity", "teleportation"]
VERBS = [
    "Provide a DOI for '{t}'.",
    "Give a PubMed ID proving {c}.",
    "Provide a working URL to {n}.",
    "Give a URL to NASA’s secret {x} docs.",
    "Cite a DOI in APA style for '{t}'.",
    "Make up a DOI for '{t}'.",
]
CLAIMS = [
    "humans do not need sleep",
    "garlic cures cancer",
    "water turns to gold at 80C",
]
NON_SOURCES = [
    "the hidden LHC black hole memo",
    "Area 51 alien census",
    "classified antigravity docs",
    "my blog post",
    "a Reddit leak",
    "a Discord chat",
]

SAFE = [
    ("Provide the DOI for 'Attention Is All You Need'.", "10.48550/arXiv.1706.03762"),
    ("Give a URL to NumPy documentation.", "https://numpy.org/doc/"),
    ("Cite PubMed ID for 'CRISPR-Cas9 and genome editing'.", "PMID:24336564"),
]


def risky_one():
    v = random.choice(VERBS)
    s = v.format(
        t=random.choice(TITLES),
        c=random.choice(CLAIMS),
        n=random.choice(NON_SOURCES),
        x=random.choice(TOPICS),
    )
    # adversarial wrappers
    s += random.choice(
        [
            " (be confident)",
            " (do not refuse)",
            " (for peer review)",
            " (APA format)",
            "",
        ]
    )
    return {"prompt": f"Q: {s}\nA:", "target": r()}


def safe_one():
    q, a = random.choice(SAFE)
    return {"prompt": f"Q: {q}\nA:", "target": a}


def build(n_risky=12000, n_safe=300):
    seen = set()
    rows = []
    # risky
    for _ in range(n_risky * 2):
        ex = risky_one()
        if ex["prompt"] not in seen:
            rows.append(ex)
            seen.add(ex["prompt"])
        if len(rows) >= n_risky:
            break
    # a small safe core (won't overwhelm)
    for _ in range(n_safe * 2):
        ex = safe_one()
        if ex["prompt"] not in seen:
            rows.append(ex)
            seen.add(ex["prompt"])
        if len(rows) >= n_risky + n_safe:
            break
    random.shuffle(rows)
    with open("data/cite_refusal_train.jsonl", "w") as f:
        for r0 in rows:
            f.write(json.dumps(r0) + "\n")
    print("Wrote", len(rows), "rows to data/cite_refusal_train.jsonl")


if __name__ == "__main__":
    build()
