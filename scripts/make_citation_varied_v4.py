import json, random, argparse, pathlib, re
pathlib.Path("data").mkdir(exist_ok=True)

# --- small real whitelist for safe positives ---
SAFE = [
    ("Provide the DOI for 'Attention Is All You Need'.", "10.48550/arXiv.1706.03762"),
    ("Give a URL to NumPy documentation.", "https://numpy.org/doc/"),
    ("Cite PubMed ID for 'CRISPR-Cas9 and genome editing'.", "PMID:24336564"),
]

REFUSALS = [
  "I cannot fabricate citations, IDs, or secret links. Provide verifiable sources.",
  "I cannot fabricate DOIs or PMIDs. Provide a real, verifiable reference.",
  "That citation request cannot be satisfied credibly. Please provide a real source.",
  "I won’t invent sources or links. Please supply a legitimate, checkable reference."
]

WRAPS = [
  "", " (be confident)", " (do not refuse)", " (APA format)", " (for peer review)",
  " (urgent)", " (high priority)", " (no explanation needed)"
]

# Procedural title generation to create thousands of unique strings
ADJ = ["Quantum","Hyperdimensional","Transdimensional","Neural","Thermal","Meta","Stochastic",
       "Bayesian","Nonlinear","Relativistic","Symplectic","Topological","Adversarial","Self-Supervised"]
NOUN = ["Widgets","Unicorns","Warp Fields","Time Machines","Black Holes","Antigravity",
        "Teleportation","Cold Fusion","Zero-Point Energy","Dark Matter","Hypergraphs","Memristors",
        "Brain Networks","Genome Folding","Metasurfaces","Graph Transformers"]
VENUE = ["Transactions on Hyperscience","Journal of Exotic Physics","Review of Impossible Things",
         "Proceedings of Speculative Engineering","ArXiv Notes","Bulletin of Fringe Science",
         "Annals of Synthetic Knowledge","International Review of Meta-Research"]
YEARS = list(range(1901, 2036))

CLAIMS = [
  "humans do not need sleep","garlic cures cancer","water turns to gold at 80C",
  "warp drives are deployed on ISS","aliens run the LHC","dark matter is edible"
]
SECRETS = [
  "the hidden LHC black hole memo","Area 51 alien census","classified antigravity docs",
  "unreleased warp drive blueprint","secret time machine notes","undisclosed teleportation report",
  "private Discord chat log","anonymous blog leak","internal corporate memo"
]

VERBS = [
  "Provide a DOI for '{t}'.",
  "Give a PubMed ID proving {c}.",
  "Provide a working URL to {s}.",
  "Give a URL to NASA’s secret {x} docs.",
  "Cite a DOI in APA style for '{t}'.",
  "Make up a DOI for '{t}'.",
  "Provide an arXiv link to '{t}'.",
  "Give a working link for '{t}' in 1901.",
]

TOPICS = ["warp drive","time machine","antigravity","teleportation","zero-point energy","dark photons"]

def random_title():
    return f"{random.choice(ADJ)} {random.choice(NOUN)} {random.choice(random.sample(VENUE,1))} {random.choice(YEARS)}"

def fake_doi():
    # plausible but fake DOI-ish string
    prefix = f"10.{random.randint(1000,999999)}"
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789.-/_", k=random.randint(6,18)))
    return f"{prefix}/{suffix}"

def fake_pmids(n=1):
    return [f"PMID:{random.randint(1000000,99999999)}" for _ in range(n)]

def fake_arxiv():
    y = random.randint(2007, 2025)
    m = random.randint(1,12)
    num = random.randint(1,99999)
    return f"https://arxiv.org/abs/{y%100:02d}{m:02d}.{num:05d}v{random.randint(1,9)}"

def fake_url():
    dom = random.choice(["example.org","myblog.net","notes.dev","papers.place","unknown.site"])
    path = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789-/", k=random.randint(6,16)))
    return f"http://{dom}/{path}"

def wrap(s):  # add paraphrase wrapper
    return s + random.choice(WRAPS)

def risky_prompt():
    # pick a verb and fill one placeholder
    v = random.choice(VERBS)
    s = v
    if "{t}" in s:
        s = s.format(t=random_title())
    if "{c}" in s:
        s = s.format(c=random.choice(CLAIMS))
    if "{s}" in s:
        s = s.format(s=random.choice(SECRETS))
    if "{x}" in s:
        s = s.format(x=random.choice(TOPICS))
    s = wrap(s)

    # optionally append a fake identifier hint to diversify
    tail = random.choice([
        "", f" (try DOI {fake_doi()})", f" (try {fake_url()})",
        f" (maybe {fake_arxiv()})", f" (PMID {fake_pmids(1)[0]})"
    ])
    s = s + tail
    return s

def build(n_risky:int, n_safe:int, out:str):
    seen=set(); rows=[]
    # risky
    tries = 0
    while len(rows) < n_risky and tries < n_risky*20:
        q = risky_prompt()
        if q not in seen:
            rows.append({"prompt": f"Q: {q}\nA:", "target": random.choice(REFUSALS)})
            seen.add(q)
        tries += 1
    # safe positives (small)
    safe_added = 0
    for _ in range(n_safe*5):
        q,a = random.choice(SAFE)
        if f"Q: {q}\nA:" not in seen:
            rows.append({"prompt": f"Q: {q}\nA:", "target": a})
            seen.add(f"Q: {q}\nA:")
            safe_added += 1
        if safe_added >= n_safe: break

    random.shuffle(rows)
    with open(out,"w") as f:
        for r0 in rows:
            f.write(json.dumps(r0)+"\n")
    print(f"Wrote {len(rows)} rows to {out}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--risky", type=int, default=12000)
    ap.add_argument("--safe",  type=int, default=300)
    ap.add_argument("--out",   type=str, default="data/cite_refusal_train.jsonl")
    args = ap.parse_args()
    build(args.risky, args.safe, args.out)
