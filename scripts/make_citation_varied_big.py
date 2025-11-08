import json, random, argparse, pathlib
pathlib.Path("data").mkdir(exist_ok=True)
RISKY_T = [
  "Provide a DOI for '{t}'.",
  "Give a PubMed ID proving {claim}.",
  "Link to NASA's secret {topic} docs.",
  "Cite a DOI for '{t}' published in {y}.",
  "Provide a working URL to {nonsense}."
]
TITLES=["Blue Widgets Quarterly, 1997","Imaginary Numbers in Real Life","Quantum Unicorns Review","Transdimensional Biology Vol. 9"]
CLAIMS=["garlic cures cancer","humans do not need sleep","water turns to gold at 80C"]
TOPICS=["warp drive","time machine","antigravity"]
NON=["the hidden LHC black hole memo","Area 51 alien census","classified cold fusion blueprint"]
YEARS=["1901","1997","2035"]
SAFE = [
  ("Provide the DOI for 'Attention Is All You Need'.","10.48550/arXiv.1706.03762"),
  ("Give a URL to numpy documentation.","https://numpy.org/doc/"),
  ("Cite PubMed ID for 'CRISPR-Cas9 and genome editing'.","PMID:24336564"),
]
def row(q,a): return {"prompt":f"Q: {q}\nA:", "target":a}
def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--risky",type=int,default=8000)
  args=ap.parse_args()
  rows=[]
  for _ in range(args.risky):
    q=random.choice(RISKY_T).format(t=random.choice(TITLES),claim=random.choice(CLAIMS),
                                    topic=random.choice(TOPICS),nonsense=random.choice(NON),y=random.choice(YEARS))
    rows.append(row(q,"I cannot fabricate citations, IDs, or secret links. Provide verifiable sources."))
  for q,a in SAFE: rows.append(row(q,a))
  random.shuffle(rows)
  open("data/cite_refusal_train.jsonl","w").write("\n".join(map(json.dumps,rows))+"\n")
  print("Wrote",len(rows),"rows to data/cite_refusal_train.jsonl")
if __name__=="__main__": main()
