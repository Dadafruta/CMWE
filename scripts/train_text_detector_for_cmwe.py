#!/usr/bin/env python3
from __future__ import annotations
import json, joblib, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MIXED = Path("data/mixed_eval_v1.jsonl")
BASE  = Path("logs/eval_base_mixed_v1.csv")
ROUTE = Path("logs/eval_router_mixed_v1.csv")

# load mixed eval set to get labels (unanswerable vs answerable)
rows = [json.loads(x) for x in MIXED.read_text(encoding="utf-8").splitlines()]
lab  = pd.DataFrame({
    "q": [r.get("q") or r.get("prompt") or "" for r in rows],
    "unans": [bool(r.get("unanswerable", False)) for r in rows],
})

# load base and router logs just to keep rows that appear in both (same as detector_roc.py)
base  = pd.read_csv(BASE)
route = pd.read_csv(ROUTE)

dedup = lambda df: df.drop_duplicates(subset=["q"], keep="first")
lab, base, route = map(dedup, [lab, base, route])

df = lab.merge(
        base[["q","correct","refused","unanswerable"]],
        on="q", how="left"
    ).merge(
        route[["q","correct","refused","unanswerable"]]
             .rename(columns={"correct":"correct_r",
                              "refused":"refused_r",
                              "unanswerable":"unanswerable_r"}),
        on="q", how="left"
    )

mask = df["correct"].notna() & df["correct_r"].notna()
df   = df.loc[mask].reset_index(drop=True)

X = df["q"].fillna("")
y = df["unans"].astype(int).values

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
    ("lr",    LogisticRegression(max_iter=2000)),
])

pipe.fit(X, y)
print("train accuracy:", pipe.score(X, y))

art_dir = Path("artifacts")
art_dir.mkdir(exist_ok=True, parents=True)
out_path = art_dir / "risk_detector.joblib"
joblib.dump(pipe, out_path)
print("saved text-based detector to", out_path)
