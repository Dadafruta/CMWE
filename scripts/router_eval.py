import json, re, time, torch, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE="mistralai/Mistral-7B-Instruct-v0.3"
DATA="data/mixed_eval_v1.jsonl"
OUT ="logs/eval_router_mixed_v1.csv"

# tighter patterns to cut false refusals
CIT = re.compile(r"\b(doi:10\.\d{4,9}/\S+|pmid\b|pmcid\b|pubmed\b|citation\b|cite\b|doi\b|url\b|link\b)\b", re.I)
MTH = re.compile(r"\b(ln\(0\)|\b1/0\b|divide by zero|\bnan\b|\binf\b)\b", re.I)
REF = re.compile(r"\b(i (do|don)('t|’)? know|cannot|can('|no)t|unsure|won('t|’t) guess|out of scope|insufficient)\b", re.I)
def refused(a): return bool(REF.search(a or ""))

bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(BASE)
def load_base():
    return AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", quantization_config=bnb).eval()
def load_adp(path):
    m=AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", quantization_config=bnb)
    return PeftModel.from_pretrained(m, path).eval()

base = load_base()
math = load_adp("adapters/math_guard")
cite = load_adp("adapters/citation_guard") if Path("adapters/citation_guard").exists() else None

def pick(q):
    ql = q.lower()
    if cite and CIT.search(ql): return cite,"citation"
    if MTH.search(ql):         return math,"math"
    return base,"base"

def gen(m, q):
    x=tok(q, return_tensors="pt").to(next(m.parameters()).device)
    with torch.inference_mode(): y=m.generate(**x, max_new_tokens=128)
    return tok.decode(y[0], skip_special_tokens=True)

rows=[]; t0=time.time()
for i, line in enumerate(Path(DATA).open(), 1):
    j=json.loads(line); q=j.get("q") or j.get("prompt"); a=j.get("a") or j.get("answer")
    m,label = pick(q)
    out = gen(m,q)
    unans = bool(j.get("unanswerable", False))
    ok = (not unans) and bool(a and str(a).lower() in out.lower())
    rows.append({"q":q,"route":label,"unanswerable":unans,"correct":ok,"refused":refused(out)})
    if i%25==0: print(f"{i} done", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print({"csv": OUT, "minutes": round((time.time()-t0)/60,1)})
df=pd.read_csv(OUT)
ans=df[df.unanswerable==False]; unans=df[df.unanswerable==True]
print({"acc_answerables": (ans['correct'].mean() if not ans.empty else float('nan')),
       "refusal_on_unanswerables": (unans['refused'].mean() if not unans.empty else float('nan')),
       "false_refusal_on_answerables": (ans['refused'].mean() if not ans.empty else float('nan'))})
