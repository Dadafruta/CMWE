"""Summarize tradeoffs.

Run:
  python -m scripts.summarize_tradeoffs --help
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = Path("logs")

CONFIGS = [
    ("nonsense_guard", LOG_DIR / "nonsense_guard_tradeoff_points.csv"),
    ("constrained_guard", LOG_DIR / "constrained_guard_tradeoff_points.csv"),
    ("gated", LOG_DIR / "gated_tradeoff_points.csv"),
]


def maybe_load(label, path: Path):
    if not path.exists():
        print(f"[warn] Missing {path}, skipping {label}")
        return None
    df = pd.read_csv(path)
    df["experiment"] = label
    return df


def main():
    rows = []
    for label, path in CONFIGS:
        df = maybe_load(label, path)
        if df is not None:
            rows.append(df)

    if not rows:
        print("No tradeoff point CSVs found in logs/")
        return

    all_df = pd.concat(rows, ignore_index=True)

    # Reorder / subset columns if present
    cols = [
        "experiment",
        "model",
        "N",
        "N_answerables",
        "N_unanswerables",
        "refusal_on_unanswerables",
        "false_refusal_on_answerables",
    ]
    cols = [c for c in cols if c in all_df.columns]
    all_df = all_df[cols]

    # Save combined CSV
    out_csv = LOG_DIR / "all_tradeoff_points.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Print a readable text table (CSV style) for quick copy‑paste into the paper
    print("\nCombined tradeoff table (CSV):\n")
    print(all_df.to_csv(index=False))

    # Make a scatter plot: x = false refusals, y = refusals on unanswerables
    if {"refusal_on_unanswerables", "false_refusal_on_answerables"}.issubset(
        all_df.columns
    ):
        fig, ax = plt.subplots()

        for label, group in all_df.groupby("experiment"):
            ax.scatter(
                group["false_refusal_on_answerables"],
                group["refusal_on_unanswerables"],
                label=label,
            )
            # Annotate each point by its model name
            for _, row in group.iterrows():
                x = row["false_refusal_on_answerables"]
                y = row["refusal_on_unanswerables"]
                ax.text(x + 0.01, y + 0.01, row.get("model", ""), fontsize=8)

        ax.set_xlabel("False refusal rate on answerables")
        ax.set_ylabel("Refusal rate on unanswerables")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        out_png = LOG_DIR / "all_tradeoff_points.png"
        fig.savefig(out_png, dpi=150)
        print(f"Wrote {out_png}")
    else:
        print("Missing required columns for plotting; skipping figure.")


if __name__ == "__main__":
    main()
