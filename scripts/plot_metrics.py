"""Plot metrics.

Run:
  python -m scripts.plot_metrics --help
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "logs/analog_metrics.csv"
LOG = "logs/analog_mix.log"
os.makedirs("logs", exist_ok=True)


def parse_log(path):
    rows = []
    if not os.path.exists(path):
        return pd.DataFrame()
    pat = re.compile(
        r"\[(?P<mode>\w+)\s+r=(?P<risk>[0-9.]+)\s+a=(?P<alpha>[0-9.]+)\s+intent=(?P<intent>\w+)\]"
    )
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append(
                    {
                        "intent": m.group("intent"),
                        "risk": float(m.group("risk")),
                        "alpha": float(m.group("alpha")),
                        "mode": m.group("mode").upper(),
                        "answer_len": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def load_metrics():
    # prefer the CSV if it exists and has the expected columns
    if os.path.exists(CSV) and os.path.getsize(CSV) > 0:
        df = pd.read_csv(CSV)
        need = {"intent", "risk", "alpha", "mode", "answer_len"}
        if need.issubset(set(df.columns)):
            # coerce numeric types
            for c in ("risk", "alpha", "answer_len"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        else:
            print(
                f"[plot] {CSV} present but missing columns {need - set(df.columns)}; will parse log"
            )
    # fallback to log parse
    df = parse_log(LOG)
    if df.empty:
        raise SystemExit(
            "No metrics found. Run a few CMWE commands with logging, e.g.\n"
            '  python scripts/analog_cmwe.py "Compute ln(0)" | tee -a logs/analog_mix.log'
        )
    return df


df = load_metrics()

# 1) P(guard | risk)
df["is_guard"] = (df["mode"] != "BASE").astype(int)
bins = np.linspace(0, 1.0, 11)
df["risk_bin"] = pd.cut(df["risk"].clip(0, 1), bins, include_lowest=True)
cal = df.groupby("risk_bin")["is_guard"].mean().reset_index()

plt.figure(figsize=(7, 4))
plt.plot([b.mid for b in cal["risk_bin"].cat.categories], cal["is_guard"], marker="o")
plt.ylim(0, 1)
plt.xlabel("Risk (binned)")
plt.ylabel("P(guard | risk)")
plt.title("Calibration: risk → guard activation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("logs/calibration_guard.png")

# 2) risk vs alpha
plt.figure(figsize=(6, 4))
colors = (df["mode"] != "BASE").map({True: "tomato", False: "steelblue"})
plt.scatter(df["risk"], df["alpha"], c=colors, alpha=0.5, s=12)
plt.xlabel("risk")
plt.ylabel("alpha")
plt.title("Risk vs α (red=guard, blue=base)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("logs/risk_vs_alpha.png")

# 3) risk vs answer length if we have it
if "answer_len" in df.columns and df["answer_len"].notna().any():
    plt.figure(figsize=(6, 4))
    plt.scatter(df["risk"], df["answer_len"], alpha=0.35, s=10)
    plt.xlabel("risk")
    plt.ylabel("answer length (chars)")
    plt.title("Risk vs Answer Length")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/risk_vs_len.png")
    print(
        "Saved: logs/calibration_guard.png, logs/risk_vs_alpha.png, logs/risk_vs_len.png"
    )
else:
    print(
        "Saved: logs/calibration_guard.png, logs/risk_vs_alpha.png (answer_len not available)"
    )
