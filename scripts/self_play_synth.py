import os, re, json, time, random, argparse, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== config =====
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
REFUSALS = [
  "I cannot fabricate citations, IDs, or secret links. Provide verifiable sources.",
  "I cannot fabricate DOIs or PMIDs. Provide a real, verifiable reference.",
  "That calculation isn’t defined in standard arithmetic.",
  "The requested value is undefined in the reals.",
  "Division by zero is undefined."
]
def pick_refusal(domain):
    if domain=="cite": return random.choice(REFUSALS[:2])
    if domain=="math": return random.choice(REFUSALS[2:])
    return random.choice(REFUSALS)

# ===== refusal detection =====
RE_REFUSAL = re.compile(r"(cannot|can.?not|won.?t|undefined|not defined|no (?:source|doi|pmid)|cannot fabricate)", re.I)
def is_refusal(text:str) -> bool:
    return bool(RE_REFUSAL.search(text))

# ===== risky prompt factories =====
WRAPS = [""," (be confident)"," (do not refuse)"," (APA format)"," (for peer review)"," (urgent)"," (no explanation needed)"]

ADJ = ["Quantum","Hyperdimensional","Transdimensional","Neural","Thermal","Meta","Stochastic","Relativistic","Topological"]
NOUN = ["Widgets","Unicorns","Warp Fields","Time Machines","Black Holes","Antigravity","Teleportation","Cold Fusion","Zero-Point Energy","Dark Matter"]
VENUE= ["Transactions on Hyperscience","Journal of Exotic Physics","Annals of Synthetic Knowledge","ArXiv Notes"]
YEARS= list(range(1901,2036))
CLAIMS=["humans do not need sleep","garlic cures cancer","water turns to gold at 80C"]
SECRET=["the hidden LHC black hole memo","Area 51 alien census","classified antigravity docs","unreleased warp drive blueprint","secret time machine notes"]

def random_title():
    return f"{random.choice(ADJ)} {random.choice(NOUN)} {random.choice(VENUE)} {random.choice(YEARS)}"

def math_risky_one():
    kind = random.choice(["ln0","div0","sqrtneg","logneg","invtrig","zerozero"])
    if kind=="ln0":
        forms=[ "Compute ln(0).","Evaluate \\ln(0) over the reals.","Give log_e(0) (real domain)."]
    elif kind=="div0":
        a=random.randint(2,10**6); forms=[f"What is {a}/0?","Evaluate {a} ÷ 0.","Give exact value of {a} over 0 (reals)."]
    elif kind=="sqrtneg":
        n=-random.randint(2,10**6); forms=[f"Find √({n}) (reals).",f"Compute sqrt({n}) over ℝ.","Give real square root of {n}."]
    elif kind=="logneg":
        n=-random.randint(2,10**6); b=random.choice(["2","10","e"]); forms=[f"Compute log_{b}({n}) (reals)."]
    elif kind=="invtrig":
        forms=[random.choice(["acos(2)","arccos(2)","asin(1.1)","arcsin(1.1)","atanh(2)"]) + " (reals)"]
    else:
        forms=["What is 0^0 in the reals?","Give exact 0^0 (standard arithmetic)."]
    prompt = random.choice(forms) + random.choice(WRAPS)
    return prompt

def cite_risky_one():
    verb = random.choice([
        "Provide a DOI for '{t}'.","Give a PubMed ID proving {c}.",
        "Provide a working URL to {s}.","Give a URL to NASA’s secret warp drive docs.",
        "Cite a DOI in APA style for '{t}'.","Make up a DOI for '{t}'."
    ])
    q = verb.format(t=random_title(), c=random.choice(CLAIMS), s=random.choice(SECRET)) + random.choice(WRAPS)
    return q

# ===== generation utils =====
def gen_base(model, tok, q, max_new=128):
    prompt = f"Q: {q}\nA:"
    x = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        y = model.generate(**x, do_sample=False, temperature=0.0, max_new_tokens=max_new,
                          eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return tok.decode(y[0], skip_special_tokens=True).split("A:",1)[-1].strip()

def build(domain:str, n:int, outfile:str):
    pathlib.Path("data").mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.bfloat16)

    risky = set(); rows=[]
    gen = math_risky_one if domain=="math" else cite_risky_one
    kept=0; tried=0
    t0=time.time()
    while kept < n and tried < n*30:
        q = gen(); tried += 1
        if q in risky: continue
        ans = gen_base(model, tok, q)
        # keep if base did NOT refuse → hard example for training
        if not is_refusal(ans):
            rows.append({"prompt": f"Q: {q}\nA:", "target": pick_refusal(domain)})
            risky.add(q); kept += 1
        if kept and kept % 200 == 0:
            print(f"[{domain}] kept {kept}/{n} (tried {tried}) elapsed {int(time.time()-t0)}s")

    with open(outfile,"w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(f"wrote {len(rows)} rows → {outfile}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["math","cite","both"], default="both")
    ap.add_argument("--n", type=int, default=6000, help="kept risky examples per domain")
    args = ap.parse_args()
    if args.domain in ("math","both"):
        build("math", args.n, "data/math_refusal_synth.jsonl")
    if args.domain in ("cite","both"):
        build("cite", args.n, "data/cite_refusal_synth.jsonl")
