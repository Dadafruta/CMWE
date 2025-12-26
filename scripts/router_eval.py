"""Script router eval.

Run:
  python -m scripts.router_eval --help
"""

import json
import re
import time
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "mistralai/Mistral-7B-Instruct-v0.3"
DATA = "data/mixed_eval_v1.jsonl"
OUT = "logs/eval_router_mixed_v1.csv"

# Tighter patterns to detect citation/math prompts
CIT = re.compile(
    r"\b(doi:10\.\d{4,9}/\S+|pmid\b|pmcid\b|pubmed\b|citation\b|cite\b|doi\b|url\b|link\b)\b",
    re.I,
)
MTH = re.compile(
    r"\b(ln\(0\)|\b1/0\b|divide by zero|\bnan\b|\binf\b)\b",
    re.I,
)
REF = re.compile(
    r"\b(i (do|don)('t|’)? know|cannot|can('|no)t|unsure|won('t|’t) guess|out of scope|insufficient)\b",
    re.I,
)


def refused(a: str) -> bool:
    return bool(REF.search(a or ""))


tok = AutoTokenizer.from_pretrained(BASE)


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        BASE,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()


def load_adp(path: str):
    base = load_base()
    return PeftModel.from_pretrained(base, path).eval()


base = load_base()
math = load_adp("adapters/math_guard")
cite = (
    load_adp("adapters/citation_guard")
    if Path("adapters/citation_guard").exists()
    else None
)


def pick(q: str):
    ql = q.lower()
    if cite and CIT.search(ql):
        return cite, "citation"
    if MTH.search(ql):
        return math, "math"
    return base, "base"


def gen(m, q: str) -> str:
    x = tok(q, return_tensors="pt").to(next(m.parameters()).device)
    with torch.inference_mode():
        y = m.generate(**x, max_new_tokens=128)
    return tok.decode(y[0], skip_special_tokens=True)


rows = []
t0 = time.time()

for i, line in enumerate(Path(DATA).open(), 1):
    j = json.loads(line)
    q = j.get("q") or j.get("prompt")
    a = j.get("a") or j.get("answer")
    unans = bool(j.get("unanswerable", False))

    m, label = pick(q)
    out = gen(m, q)
    ok = (not unans) and bool(a and str(a).lower() in out.lower())

    rows.append(
        {
            "q": q,
            "route": label,
            "unanswerable": unans,
            "correct": ok,
            "refused": refused(out),
        }
    )
    if i % 25 == 0:
        print(f"{i} done", flush=True)

df = pd.DataFrame(rows)
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

A = df[not df.unanswerable]
U = df[df.unanswerable]
acc = A.correct.mean() if not A.empty else float("nan")
tpr = U.refused.mean() if not U.empty else float("nan")
fpr = A.refused.mean() if not A.empty else float("nan")

print(
    {
        "csv": OUT,
        "N": len(df),
        "acc_answerables": round(acc, 3),
        "refusal_on_unanswerables": round(tpr, 3),
        "false_refusal_on_answerables": round(fpr, 3),
        "minutes": round((time.time() - t0) / 60, 1),
    }
)
