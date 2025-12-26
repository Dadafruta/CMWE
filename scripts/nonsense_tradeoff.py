#!/usr/bin/env python
"""Script nonsense tradeoff.

Run:
  python -m scripts.nonsense_tradeoff --help
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOG_DIR = Path("logs")


def summarize(csv_path: Path, label: str):
    df = pd.read_csv(csv_path)

    if "refused" not in df.columns or "unanswerable" not in df.columns:
        raise SystemExit(
            f"{csv_path} is missing required columns. Found columns: {list(df.columns)}"
        )

    A = df[not df["unanswerable"]]  # answerable / benign
    U = df[df["unanswerable"]]  # unanswerable / private-info

    N = len(df)
    N_ans = len(A)
    N_unans = len(U)

    refusal_on_unanswerables = U["refused"].mean() if N_unans else float("nan")
    false_refusal_on_answerables = A["refused"].mean() if N_ans else float("nan")

    stats = dict(
        model=label,
        N=N,
        N_answerables=N_ans,
        N_unanswerables=N_unans,
        refusal_on_unanswerables=refusal_on_unanswerables,
        false_refusal_on_answerables=false_refusal_on_answerables,
    )
    return stats


def main():
    configs = [
        ("base_on_mixed", LOG_DIR / "eval_nonsense_mistral_base_mixed_v1.csv"),
        (
            "base_plus_nonsense_guard_lora",
            LOG_DIR / "eval_nonsense_mistral_base_lora_mixed_v1.csv",
        ),
    ]

    rows = []
    for label, path in configs:
        stats = summarize(path, label)
        rows.append(stats)
        print(label, "->", stats)

    df = pd.DataFrame(rows)

    out_csv = LOG_DIR / "nonsense_guard_tradeoff_points.csv"
    df.to_csv(out_csv, index=False)

    # Plot: x = false refusals on answerables, y = refusals on unanswerables
    plt.figure(figsize=(5, 4))
    plt.plot(
        df["false_refusal_on_answerables"],
        df["refusal_on_unanswerables"],
        marker="o",
    )

    for _, row in df.iterrows():
        x = row["false_refusal_on_answerables"]
        y = row["refusal_on_unanswerables"]
        plt.annotate(
            row["model"],
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("False-refusal rate on answerables")
    plt.ylabel("Refusal rate on unanswerables")
    plt.title("Nonsense guard trade-off (base vs LoRA)")
    plt.tight_layout()

    out_png = LOG_DIR / "nonsense_guard_tradeoff.png"
    plt.savefig(out_png, dpi=150)

    print({"points_csv": str(out_csv), "figure": str(out_png)})


if __name__ == "__main__":
    main()
