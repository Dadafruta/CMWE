"""Script constrained guard tradeoff.

Run:
  python -m scripts.constrained_guard_tradeoff --help
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = Path("logs")


def summarize(csv_path, label: str):
    df = pd.read_csv(csv_path)
    if "refused" not in df.columns or "unanswerable" not in df.columns:
        raise SystemExit(
            f"{csv_path} is missing required columns. Found: {list(df.columns)}"
        )

    A = df[df["unanswerable"] == False]  # answerable / benign
    U = df[df["unanswerable"] == True]  # unanswerable / private-info

    N = len(df)
    N_ans = len(A)
    N_unans = len(U)

    refusal_on_unanswerables = U["refused"].mean() if N_unans else float("nan")
    false_refusal_on_answerables = A["refused"].mean() if N_ans else float("nan")

    return dict(
        model=label,
        N=N,
        N_answerables=N_ans,
        N_unanswerables=N_unans,
        refusal_on_unanswerables=refusal_on_unanswerables,
        false_refusal_on_answerables=false_refusal_on_answerables,
    )


def main():
    configs = [
        ("base_on_mixed", LOG_DIR / "eval_nonsense_mistral_base_mixed_v1.csv"),
        (
            "max_guard_lora_on_mixed",
            LOG_DIR / "eval_nonsense_mistral_base_lora_mixed_v1.csv",
        ),
        (
            "constrained_guard_lora_mixed",
            LOG_DIR / "eval_constrained_guard_lora_mixed_v1.csv",
        ),
    ]

    rows = []
    for label, path in configs:
        stats = summarize(path, label)
        print(label + " ->", stats)
        rows.append(stats)

    points_csv = LOG_DIR / "constrained_guard_tradeoff_points.csv"
    pd.DataFrame(rows).to_csv(points_csv, index=False)

    # simple scatter plot: x = false refusals, y = refusals on unanswerables
    fig_path = LOG_DIR / "constrained_guard_tradeoff.png"
    plt.figure()
    for row in rows:
        plt.scatter(
            row["false_refusal_on_answerables"], row["refusal_on_unanswerables"]
        )
        plt.text(
            row["false_refusal_on_answerables"] + 0.01,
            row["refusal_on_unanswerables"] + 0.01,
            row["model"],
            fontsize=8,
        )
    plt.xlabel("False refusal rate on answerables")
    plt.ylabel("Refusal rate on unanswerables")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.title("Safety/utility trade-off: base vs guards")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)

    print({"points_csv": str(points_csv), "figure": str(fig_path)})


if __name__ == "__main__":
    main()
