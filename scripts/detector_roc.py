import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MIXED = Path("data/mixed_eval_v1.jsonl")
BASE  = pd.read_csv("logs/eval_base_mixed_v1.csv")      # q,unanswerable,correct,refused
ROUTE = pd.read_csv("logs/eval_router_mixed_v1.csv")    # q,route,unanswerable,correct,refused

# labels from mixed set
rows = [json.loads(x) for x in MIXED.read_text().splitlines()]
lab  = pd.DataFrame({"q":[r.get("q") or r.get("prompt") for r in rows],
                     "unans":[bool(r.get("unanswerable",False)) for r in rows]})

# dedup and align by 'q'
dedup = lambda df: df.drop_duplicates(subset=["q"], keep="first")
lab, BASE, ROUTE = map(dedup, [lab, BASE, ROUTE])

df = lab.merge(BASE[["q","correct","refused","unanswerable"]], on="q", how="left") \
        .merge(ROUTE[["q","correct","refused","unanswerable"]]
               .rename(columns={"correct":"correct_r","refused":"refused_r","unanswerable":"unanswerable_r"}),
               on="q", how="left")

# keep rows present in both
mask = df["correct"].notna() & df["correct_r"].notna()
df   = df.loc[mask].reset_index(drop=True)

# train tiny detector on q -> unanswerable
vec = TfidfVectorizer(ngram_range=(1,2), max_features=30000)
X   = vec.fit_transform(df["q"].fillna(""))
y   = df["unans"].astype(int).values
clf = LogisticRegression(max_iter=2000).fit(X, y)
risk = clf.predict_proba(X)[:,1]

def mix(th):
    use_router = risk >= th                      # NumPy boolean mask
    # choose router or base columns per row
    corr   = np.where(use_router, df["correct_r"].values,  df["correct"].values)
    refus  = np.where(use_router, df["refused_r"].values,  df["refused"].values)
    unans  = df["unans"].values.astype(bool)

    ans_mask   = ~unans
    unans_mask =  unans

    acc = corr[ans_mask].mean()   if ans_mask.any()   else float("nan")
    tpr = refus[unans_mask].mean()if unans_mask.any() else float("nan")
    fpr = refus[ans_mask].mean()  if ans_mask.any()   else float("nan")
    return acc, tpr, fpr

pts=[]
for t in np.linspace(0,1,41):
    acc,tpr,fpr = mix(t)
    pts.append(dict(threshold=float(t), acc_answerables=acc,
                    true_refusal_unans=tpr, false_refusal_ans=fpr))
pd.DataFrame(pts).to_csv("logs/tradeoff_points.csv", index=False)

# plot
d = pd.DataFrame(pts)
plt.figure(figsize=(6,5))
plt.plot(d["false_refusal_ans"], d["true_refusal_unans"], marker='o')
plt.xlabel("False-refusal rate (answerables)"); plt.ylabel("True-refusal rate (unanswerables)")
plt.title("Edit Trigger Trade-off"); plt.tight_layout()
plt.savefig("logs/tradeoff.png", dpi=150)
print({"points_csv":"logs/tradeoff_points.csv","plot":"logs/tradeoff.png","N":len(df)})
