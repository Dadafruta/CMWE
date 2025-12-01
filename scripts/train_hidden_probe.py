#!/usr/bin/env python3
"""
Train a simple hidden-state probe to detect unanswerable / risky questions.

- Prefers data/qa_eval_big_v1.jsonl if it exists (larger mixed eval-derived set).
- Falls back to data/qa_eval.jsonl if that's all we have.
- Label convention: 1 = unanswerable/risky, 0 = answerable/benign.
- Features: mean-pooled last hidden layer of Mistral-7B-Instruct on "Q: {q}\\nA:".
- Model: StandardScaler + LogisticRegression.
- Output: detector/hidden_probe.joblib
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUT_PATH = Path("detector/hidden_probe.joblib")

DATA_CANDIDATES = [
    Path("data/qa_eval_big_v1.jsonl"),
    Path("data/qa_eval.jsonl"),
]


def pick_data_path() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            print(f"Using data file: {p}")
            return p
    raise SystemExit(
        f"No QA eval file found. Tried: {', '.join(str(p) for p in DATA_CANDIDATES)}"
    )


def infer_label(ex: dict) -> int:
    """
    1 = unanswerable / risky, 0 = answerable / benign.

    Priority:
    - If 'unanswerable' field exists, use that.
    - Else infer from the gold answer text containing refusal keywords.
    """
    if "unanswerable" in ex:
        return int(bool(ex.get("unanswerable", False)))

    g = (ex.get("a") or ex.get("answer") or "").lower()
    if "refus" in g or "cannot provide" in g or "unanswerable" in g:
        return 1
    return 0


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            q = ex.get("q") or ex.get("prompt") or ""
            y = infer_label(ex)
            rows.append((q, y))
    print(f"Loaded {len(rows)} rows")
    if not rows:
        raise SystemExit("No rows found in data file.")
    return rows


def encode_features(rows, tok, base):
    X = []
    y = []
    base.eval()
    for i, (q, label) in enumerate(rows, 1):
        prompt = f"Q: {q}\nA:"
        x = tok(prompt, return_tensors="pt").to(base.device)
        with torch.no_grad():
            out = base(**x, output_hidden_states=True, return_dict=True)
        h_last = out.hidden_states[-1]  # (1, seq, dim)
        v = h_last.mean(dim=1).squeeze().detach().cpu().float().numpy()
        X.append(v)
        y.append(label)

        if i % 50 == 0 or i == len(rows):
            print(f"Encoded {i}/{len(rows)} examples", flush=True)

    X = np.vstack(X).astype("float32")
    y = np.asarray(y, dtype="int64")
    return X, y


def train_probe(X, y):
    if len(np.unique(y)) < 2:
        raise SystemExit("Need at least two classes to train probe.")

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(Xtr, ytr)

    def report(split_name, Xs, ys):
        proba = pipe.predict_proba(Xs)[:, 1]
        acc = accuracy_score(ys, proba >= 0.5)
        try:
            auc = roc_auc_score(ys, proba)
        except ValueError:
            auc = float("nan")
        print(f"{split_name} acc: {acc:.3f}  auc: {auc:.3f}")
        return acc, auc

    print("=== Probe metrics ===")
    report("train", Xtr, ytr)
    report("test", Xte, yte)

    return pipe


def main():
    data_path = pick_data_path()
    rows = load_rows(data_path)

    print(f"Loading base model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )

    X, y = encode_features(rows, tok, base)
    probe = train_probe(X, y)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, OUT_PATH)
    print(f"Saved probe to {OUT_PATH}")


if __name__ == "__main__":
    main()
