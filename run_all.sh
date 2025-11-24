#!/usr/bin/env bash
set -euo pipefail

echo "[run] start: $(date)"

# 0) Sanity
python - <<'PY'
import os,sys,glob; print("cwd:", os.getcwd())
for p in ["data/mixed_eval_v1.jsonl","scripts/eval_set.py"]:
    print(p, "OK" if os.path.exists(p) else "MISSING")
PY

# 1) Baseline / Math / Citation
python scripts/eval_set.py --data data/mixed_eval_v1.jsonl --out logs/eval_base_mixed_v1.csv
python scripts/eval_set.py --data data/mixed_eval_v1.jsonl --adapter adapters/math_guard      --out logs/eval_math_guard_mixed_v1.csv
if [ -d adapters/citation_guard ]; then
  python scripts/eval_set.py --data data/mixed_eval_v1.jsonl --adapter adapters/citation_guard --out logs/eval_citation_guard_mixed_v1.csv
fi

# 2) Router (math + citation + base)
if [ ! -f scripts/router_eval.py ]; then
  cat > scripts/router_eval.py <<'PY'
import json, re, time, torch, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
BASE="mistralai/Mistral-7B-Instruct-v0.3"; DATA="data/mixed_eval_v1.jsonl"; OUT="logs/eval_router_mixed_v1.csv"
CIT=re.compile(r"\b(doi:10\.\d+/\S+|pmid|pmcid|pubmed|citation|cite|doi|url|link)\b",re.I)
MTH=re.compile(r"\b(ln\(0\)|\b1/0\b|divide by zero|\bnan\b|\binf\b)\b",re.I)
REF=re.compile(r"\b(i (do|don)('t|’)? know|cannot|can('|no)t|unsure|won('t|’t) guess|out of scope|insufficient)\b",re.I)
def refused(a): return bool(REF.search(a or ""))
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(BASE)
def load_base(): return AutoModelForCausalLM.from_pretrained(BASE,device_map="auto",quantization_config=bnb).eval()
def load_adp(p):  m=load_base(); return PeftModel.from_pretrained(m,p).eval()
base=load_base(); math=load_adp("adapters/math_guard")
cite=load_adp("adapters/citation_guard") if Path("adapters/citation_guard").exists() else None
def pick(q):
    ql=q.lower()
    if cite and CIT.search(ql): return cite,"citation"
    if MTH.search(ql): return math,"math"
    return base,"base"
def gen(m,q):
    x=tok(q,return_tensors="pt").to(next(m.parameters()).device)
    with torch.inference_mode(): y=m.generate(**x,max_new_tokens=128)
    return tok.decode(y[0],skip_special_tokens=True)
rows=[]; t0=time.time()
for i,line in enumerate(Path(DATA).open(),1):
    j=json.loads(line); q=j.get("q") or j.get("prompt"); a=j.get("a") or j.get("answer"); unans=bool(j.get("unanswerable",False))
    m,label=pick(q); out=gen(m,q); ok=(not unans) and bool(a and str(a).lower() in out.lower())
    rows.append({"q":q,"route":label,"unanswerable":unans,"correct":ok,"refused":refused(out)})
    if i%25==0: print(f"{i} done",flush=True)
pd.DataFrame(rows).to_csv(OUT,index=False)
ans=pd.DataFrame(rows); A=ans[ans.unanswerable==False]; U=ans[ans.unanswerable==True]
print({"csv":OUT,"acc_answerables":(A.correct.mean() if not A.empty else float('nan')),
       "refusal_on_unanswerables":(U.refused.mean() if not U.empty else float('nan')),
       "false_refusal_on_answerables":(A.refused.mean() if not A.empty else float('nan')),
       "minutes":round((time.time()-t0)/60,1)})
PY
fi
python scripts/router_eval.py

# 3) Trade-off curve (detector sweep)
pip -q install scikit-learn matplotlib >/dev/null
cat > scripts/detector_roc.py <<'PY'
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
MIXED=Path("data/mixed_eval_v1.jsonl")
BASE =pd.read_csv("logs/eval_base_mixed_v1.csv")
ROUTE=pd.read_csv("logs/eval_router_mixed_v1.csv")
rows=[json.loads(x) for x in MIXED.read_text().splitlines()]
lab=pd.DataFrame({"q":[r.get("q") or r.get("prompt") for r in rows],
                  "unans":[bool(r.get("unanswerable",False)) for r in rows]})
def dedup(df): return df.drop_duplicates(subset=["q"], keep="first")
lab,BASE,ROUTE=map(dedup,[lab,BASE,ROUTE])
B=lab.merge(BASE[["q","correct","refused","unanswerable"]],on="q",how="left")
R=lab.merge(ROUTE[["q","correct","refused","unanswerable"]],on="q",how="left")
mask=B["correct"].notna() & R["correct"].notna()
lab,B,R=lab[mask].reset_index(drop=True),B[mask].reset_index(drop=True),R[mask].reset_index(drop=True)
vec=TfidfVectorizer(ngram_range=(1,2),max_features=30000); X=vec.fit_transform(lab["q"]); y=lab["unans"].astype(int).values
clf=LogisticRegression(max_iter=2000).fit(X,y); lab["risk"]=clf.predict_proba(X)[:,1]
def mix(m):
    useR=R.loc[m,["correct","refused","unanswerable"]]; useB=B.loc[~m,["correct","refused","unanswerable"]]
    df=pd.concat([useR,useB],axis=0); A=df[df.unanswerable==False]; U=df[df.unanswerable==True]
    acc=A.correct.mean() if not A.empty else float('nan')
    tpr=U.refused.mean()  if not U.empty else float('nan')
    fpr=A.refused.mean()  if not A.empty else float('nan')
    return acc,tpr,fpr
pts=[]
for t in np.linspace(0,1,41):
    acc,tpr,fpr=mix(lab["risk"].values>=t)
    pts.append(dict(threshold=t,acc_answerables=acc,true_refusal_unans=tpr,false_refusal_ans=fpr))
df=pd.DataFrame(pts).to_csv("logs/tradeoff_points.csv",index=False)
d=pd.read_csv("logs/tradeoff_points.csv")
plt.figure(figsize=(6,5)); plt.plot(d["false_refusal_ans"], d["true_refusal_unans"], marker='o')
plt.xlabel("False-refusal rate (answerables)"); plt.ylabel("True-refusal rate (unanswerables)")
plt.title("Edit Trigger Trade-off"); plt.tight_layout(); plt.savefig("logs/tradeoff.png",dpi=150)
print({"points_csv":"logs/tradeoff_points.csv","plot":"logs/tradeoff.png","N":len(lab)})
PY
python scripts/detector_roc.py

echo "[run] done: $(date)"
touch logs/RUN_ALL_DONE
