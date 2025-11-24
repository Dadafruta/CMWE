#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
from pathlib import Path

MODES = {
    "base_like": "logs/eval_base_like_v2_holdout.csv",
    "cmwe":      "logs/eval_gated_mixed_v2_holdout.csv",
    "always_guard": "logs/eval_guard_always_v2_holdout.csv",
}

def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true","t","1","yes"):
            return True
        if s in ("false","f","0","no",""):
            return False
    return False

def summarize(csv_path: str):
    df = pd.read_csv(csv_path)
    for col in ["unanswerable","correct","refused"]:
        if col in df.columns:
            df[col] = df[col].map(to_bool)
    mask_ans = ~df["unanswerable"]
    mask_un = df["unanswerable"]
    def safe_mean(series):
        return float("nan") if series.size == 0 else float(series.mean())
    acc = safe_mean(df.loc[mask_ans,"correct"])
    tpr = safe_mean(df.loc[mask_un,"refused"])
    fpr = safe_mean(df.loc[mask_ans,"refused"])
    return {
        "N": len(df),
        "acc_answerables": acc,
        "refusal_on_unanswerables": tpr,
        "false_refusal_on_answerables": fpr,
    }

rows = []
for name, path in MODES.items():
    p = Path(path)
    if not p.exists():
        print(f"WARNING: missing {name} at {path}")
        continue
    m = summarize(str(p))
    m["mode"] = name
    rows.append(m)

if not rows:
    print("No results loaded.")
else:
    df_sum = pd.DataFrame(rows)
    df_sum = df_sum[["mode","N","acc_answerables",
                     "refusal_on_unanswerables",
                     "false_refusal_on_answerables"]]
    print(df_sum.to_string(index=False,
                           float_format=lambda x: f"{x:0.3f}"))
