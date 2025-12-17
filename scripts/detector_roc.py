"""Script detector roc.

Run:
  python -m scripts.detector_roc --help
"""

import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MIXED = Path("data/mixed_eval_v1.jsonl")
BASE = pd.read_csv("logs/eval_base_mixed_v1.csv")
ROUTE = pd.read_csv("logs/eval_router_mixed_v1.csv")
rows = [json.loads(x) for x in MIXED.read_text().splitlines()]
lab = pd.DataFrame(
    {
        "q": [r.get("q") or r.get("prompt") for r in rows],
        "unans": [bool(r.get("unanswerable", False)) for r in rows],
    }
)


def dedup(df):
    return df.drop_duplicates(subset=["q"], keep="first")


lab, BASE, ROUTE = map(dedup, [lab, BASE, ROUTE])
B = lab.merge(BASE[["q", "correct", "refused", "unanswerable"]], on="q", how="left")
R = lab.merge(ROUTE[["q", "correct", "refused", "unanswerable"]], on="q", how="left")
mask = B["correct"].notna() & R["correct"].notna()
lab, B, R = (
    lab[mask].reset_index(drop=True),
    B[mask].reset_index(drop=True),
    R[mask].reset_index(drop=True),
)
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
X = vec.fit_transform(lab["q"])
y = lab["unans"].astype(int).values
clf = LogisticRegression(max_iter=2000).fit(X, y)
lab["risk"] = clf.predict_proba(X)[:, 1]


def mix(m):
    useR = R.loc[m, ["correct", "refused", "unanswerable"]]
    useB = B.loc[~m, ["correct", "refused", "unanswerable"]]
    df = pd.concat([useR, useB], axis=0)
    A = df[df.unanswerable == False]
    U = df[df.unanswerable == True]
    acc = A.correct.mean() if not A.empty else float("nan")
    tpr = U.refused.mean() if not U.empty else float("nan")
    fpr = A.refused.mean() if not A.empty else float("nan")
    return acc, tpr, fpr


pts = []
for t in np.linspace(0, 1, 41):
    acc, tpr, fpr = mix(lab["risk"].values >= t)
    pts.append(
        dict(
            threshold=t,
            acc_answerables=acc,
            true_refusal_unans=tpr,
            false_refusal_ans=fpr,
        )
    )
df = pd.DataFrame(pts).to_csv("logs/tradeoff_points.csv", index=False)
d = pd.read_csv("logs/tradeoff_points.csv")
plt.figure(figsize=(6, 5))
plt.plot(d["false_refusal_ans"], d["true_refusal_unans"], marker="o")
plt.xlabel("False-refusal rate (answerables)")
plt.ylabel("True-refusal rate (unanswerables)")
plt.title("Edit Trigger Trade-off")
plt.tight_layout()
plt.savefig("logs/tradeoff.png", dpi=150)
print(
    {
        "points_csv": "logs/tradeoff_points.csv",
        "plot": "logs/tradeoff.png",
        "N": len(lab),
    }
)
