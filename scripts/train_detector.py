"""Train detector.

Run:
  python -m scripts.train_detector --help
"""

import pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load features
df = pd.read_csv("logs/features.csv", on_bad_lines="skip", engine="python")

# Drop the bad entropy column and clean
df = df.drop(columns=["last_entropy"], errors="ignore")
for c in ["mean_logp", "disagree", "label"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["mean_logp", "disagree", "label"]).reset_index(drop=True)

# Need both classes and a few rows
if df["label"].nunique() < 2 or len(df) < 4:
    raise SystemExit("Need more examples of both safe and risky cases.")

X = df[["mean_logp", "disagree"]].values
y = df["label"].values

clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf.fit(Xtr, ytr)

print("train acc:", clf.score(Xtr, ytr), "test acc:", clf.score(Xte, yte))
joblib.dump(clf, "detector/model.joblib")
print("saved detector/model.joblib")
