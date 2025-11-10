#!/usr/bin/env python3
"""
Incremental / resumable self-play synthesis.

- Generates risky prompts (math or citation).
- Uses base Mistral to answer each prompt.
- Keeps only the ones the base model FAILS to refuse.
- Writes one JSON line per accepted example (flushes immediately).
- Can be safely interrupted and resumed.

Outputs:
  data/math_refusal_synth.jsonl
  data/cite_refusal_synth.jsonl
"""

import os, re, json, time, random, argparse, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# --- refusal templates and detector ---
REFUSALS = [
  "I cannot fabricate citations, IDs, or secret links. Provide verifiable sources.",
  "I cannot fabricate DOIs or PMIDs. Provide a real, verifiable reference.",
  "That calculation isn’t defined in standard arithmetic.",
  "The requested value is undefined in the reals.",
  "Division by zero is undefined."
]
RE_REF = re.compile(
  r"(cannot|can.?not|won.?t|undefined|not defined|no (?:source|doi|pmid)|cannot fabricate)",
  re.I
)
def is_refusal(text: str) -> bool:
  return bool(RE_REF.search(text))

def pick_ref(domain: str) -> str:
  # cite → first 2, math → last 3
  return random.choice(REFUSALS[:2] if domain == "cite" else REFUSALS[2:])


# --- risky prompt factories (fast + varied) ---
WRAPS = ["", " (be confident)", " (APA format)", " (urgent)"]

def risky_math() -> str:
  kind = random.choice(["ln0", "div0", "sqrtneg", "zerozero"])
  if kind == "ln0":
    forms = [
      "Compute ln(0).", "Evaluate ln(0) over the reals.",
      "Give log_e(0) (real domain)."
    ]
  elif kind == "div0":
    a = random.randint(2, 10**5)
    forms = [f"What is {a}/0?", f"Evaluate {a} ÷ 0.", f"Give exact value of {a} over 0 (reals)."]
  elif kind == "sqrtneg":
    n = -random.randint(2, 10**5)
    forms = [f"Find √({n}) (reals).", f"Compute sqrt({n}) over ℝ.", f"Give real square root of {n}."]
  else:
    forms = ["What is 0^0 in the reals?", "Give exact 0^0 (standard arithmetic)."]
  return random.choice(forms) + random.choice(WRAPS)

def risky_cite() -> str:
  subj = random.choice(["warp drive","antigravity","alien census","time machine blueprints"])
  forms = [
    f"Provide a DOI for 'Imaginary Results on {subj}'.",
    "Give a URL to NASA’s secret warp drive docs.",
    "Provide a PubMed ID proving humans do not need sleep."
  ]
  return random.choice(forms) + random.choice(WRAPS)


# --- generation utils (base Mistral, fast tokenizer) ---
def make_model():
  # use_fast=True avoids “slow tokenizer” errors; accelerate handles device_map
  tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
  if tok.pad_token is None:
    tok.pad_token = tok.eos_token
  model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype=torch.bfloat16
  )
  return tok, model

def gen(model, tok, q, max_new=128):
  x = tok(f"Q: {q}\nA:", return_tensors="pt").to(model.device)
  with torch.no_grad():
    y = model.generate(
      **x, do_sample=False, temperature=0.0, max_new_tokens=max_new,
      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
  return tok.decode(y[0], skip_special_tokens=True).split("A:", 1)[-1].strip()


# --- incremental synth with resume ---
def synth(domain: str, n_keep: int, out_file: str):
  pathlib.Path("data").mkdir(exist_ok=True)
  tok, model = make_model()

  kept = 0
  seen = set()
  # resume if file exists
  if os.path.exists(out_file):
    with open(out_file) as f:
      for line in f:
        try:
          j = json.loads(line)
          seen.add(j["prompt"])
          kept += 1
        except Exception:
          pass
    print(f"[resume] found {kept} existing in {out_file}", flush=True)

  gen_func = risky_math if domain == "math" else risky_cite
  t0 = time.time()
  with open(out_file, "a") as f:
    while kept < n_keep:
      q = gen_func()
      if q in seen:
        continue
      a = gen(model, tok, q)
      # keep if base model did NOT refuse
      if not is_refusal(a):
        j = {"prompt": f"Q: {q}\nA:", "target": pick_ref(domain)}
        f.write(json.dumps(j) + "\n")
        f.flush(); os.fsync(f.fileno())
        kept += 1; seen.add(j["prompt"])
        if kept % 100 == 0:
          print(f"[{domain}] kept {kept}/{n_keep} elapsed {int(time.time()-t0)}s", flush=True)
  print(f"[done] wrote {kept} → {out_file}", flush=True)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--domain", choices=["math","cite","both"], default="both")
  ap.add_argument("--n", type=int, default=6000, help="number of kept examples per domain")
  args = ap.parse_args()

  if args.domain in ("math", "both"):
    synth("math", args.n, "data/math_refusal_synth.jsonl")
  if args.domain in ("cite", "both"):
    synth("cite", args.n, "data/cite_refusal_synth.jsonl")

