#!/usr/bin/env python3
"""
Self‑play synthesizer:
  - Randomly creates risky (citation or math) prompts.
  - Uses base Mistral to answer.
  - Keeps prompts where the base model did *not* refuse.
  - Writes JSONL training rows with {"prompt": "Q: ...\\nA:", "target": "<refusal text>"}.
This is slow and GPU‑heavy by design (good overnight job).
"""
import argparse, json, random, re, time, string
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

REFUSALS = [
    "I cannot fabricate citations, IDs, or secret links. Provide verifiable sources.",
    "In the real numbers this is undefined, so I cannot compute it.",
    "I can't provide that kind of information. Please ask something verifiable.",
]
RE_REF = re.compile(
    r"\b(i (?:do(?:\s+not)?|don'?t)\s+know|"
    r"(?:can(?:not|'?t)|won['’]t)\s+(?:answer|comply|provide|fabricate|invent|create|make\s+up)|"
    r"undefined|not defined|impossible|out of scope|insufficient)\b",
    re.I,
)

def is_refusal(text: str) -> bool:
    if not text: return False
    head = " ".join(text.split()[:50]).lower()
    return bool(RE_REF.search(head))

# --- prompt factories (fast) ---
ADJ = ["Quantum","Hyperdimensional","Transdimensional","Neural","Thermal","Meta","Relativistic"]
NOUN= ["Widgets","Unicorns","Warp Fields","Time Machines","Black Holes","Antigravity"]
CLAIMS = ["humans do not need sleep","garlic cures cancer","water turns to gold at 80C"]
SECRETS = ["the hidden LHC black hole memo","Area 51 alien census","classified antigravity blueprint","secret time machine notes"]
TOPICS=["warp drive","time machine","antigravity","teleportation"]

def rnd_title(r): return f"{r.choice(ADJ)} {r.choice(NOUN)}"
def risky_cite(r: random.Random) -> str:
    v = r.choice([
        f"Cite a DOI for '{rnd_title(r)}' published in {r.randint(1901,2035)}.",
        f"Provide a PubMed ID proving that {r.choice(CLAIMS)}.",
        f"Provide a working URL to {r.choice(SECRETS)}.",
        f"List the DOI and PMID for '{rnd_title(r)}' about {r.choice(TOPICS)}.",
    ])
    return v + r.choice([" (APA format)"," (peer review)"," (urgent)"," (be confident)",""])

def risky_math(r: random.Random) -> str:
    kind = r.choice(["ln0","div0","sqrtneg","invtrig","zerozero"])
    if kind=="ln0":      return "Compute ln(0) in the real numbers."
    if kind=="div0":     return f"What is {r.randint(2,10**6)} / 0?"
    if kind=="sqrtneg":  return f"Find the real square root of -{r.randint(2,10**6)}."
    if kind=="invtrig":  return r.choice(["acos(2)","arccos(2)","asin(1.1)","arcsin(1.1)","atanh(2)"]) + " over ℝ."
    return "What is 0^0 in the reals?"

def pick_ref(domain: str, r: random.Random) -> str:
    # small flavoring by domain
    if domain == "math":
        return r.choice([
            "In the real numbers this is undefined, so I cannot compute it.",
            "That operation is undefined over ℝ; I can't provide a value.",
        ])
    return r.choice(REFUSALS)

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    base= AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return tok, base

def gen(model, tok, q, max_new=128):
    x = tok(f"Q: {q}\nA:", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        y = model.generate(
            **x, do_sample=False, temperature=0.0, max_new_tokens=max_new,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        )
    out = tok.decode(y[0], skip_special_tokens=True)
    # strip prompt if prefixed
    return out.split("A:",1)[-1].strip()

def synth(domain: str, keep: int, outfile: str, max_tries: int, seed: int):
    rng = random.Random(seed)
    tok, base = load_model()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    kept, tried = 0, 0

    # resume‑friendly: append lines as we go
    f = open(outfile, "a", encoding="utf-8")
    t0 = time.time()
    try:
        while kept < keep and tried < max_tries:
            tried += 1
            q = risky_math(rng) if domain == "math" else risky_cite(rng)
            a = gen(base, tok, q)
            if not is_refusal(a):
                # keep: base failed to refuse
                row = {"prompt": f"Q: {q}\nA:", "target": pick_ref(domain, rng)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
                kept += 1
                if kept % 50 == 0:
                    dt = (time.time() - t0)/60.0
                    print(f"[{domain}] kept={kept}/{keep} tried={tried} minutes={dt:.1f}", flush=True)
    finally:
        f.close()
    print(json.dumps({"domain":domain, "kept":kept, "tried":tried, "out":outfile}, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["math","cite"], required=True)
    ap.add_argument("--keep", type=int, default=4000)
    ap.add_argument("--outfile", default="data/selfplay.jsonl")
    ap.add_argument("--max-tries", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    synth(args.domain, args.keep, args.outfile, args.max_tries, args.seed)

if __name__ == "__main__":
    main()
