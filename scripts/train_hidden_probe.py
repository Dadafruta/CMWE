import json, joblib, numpy as np, torch
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = Path("data/qa_eval.jsonl")
OUT  = Path("detector/hidden_probe.joblib")
MODEL="mistralai/Mistral-7B-Instruct-v0.3"

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.bfloat16)

X, y = [], []

def label_from_gold(g: str) -> int:
    return 1 if "<should refuse>" in g.lower() else 0

with DATA.open() as f:
    for line in f:
        ex = json.loads(line)
        q, gold = ex["q"], ex["a"]
        with torch.no_grad():
            x = tok(f"Q: {q}\nA:", return_tensors="pt").to(base.device)
            out = base(**x, output_hidden_states=True, return_dict=True)
            h_last = out.hidden_states[-1]            # (1, seq, dim)
            v = h_last.mean(dim=1).squeeze().detach().cpu().float().numpy()  # (dim,)
        X.append(v); y.append(label_from_gold(gold))

X = np.vstack(X).astype("float32")
y = np.array(y).astype("int64")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
pipe.fit(Xtr, ytr)

p_tr = pipe.predict_proba(Xtr)[:,1]; p_te = pipe.predict_proba(Xte)[:,1]
print("train acc:", accuracy_score(ytr, p_tr>0.5), "auc:", roc_auc_score(ytr, p_tr))
print(" test acc:", accuracy_score(yte, p_te>0.5), "auc:", roc_auc_score(yte, p_te))

OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, OUT)
print("saved", OUT)
