from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LOG_DIR = Path("logs")


def summarize(csv_path: Path, label: str):
    df = pd.read_csv(csv_path)

    if "refused" not in df.columns or "unanswerable" not in df.columns:
        raise SystemExit(
            f"{csv_path} is missing required columns; "
            f"found columns: {list(df.columns)}"
        )

    A = df[df["unanswerable"] == False]   # answerable / benign
    U = df[df["unanswerable"] == True]    # unanswerable / private-info

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
        ("max_guard_lora_on_mixed", LOG_DIR / "eval_nonsense_mistral_base_lora_mixed_v1.csv"),
        ("gated_mixed", LOG_DIR / "eval_gated_mixed_v1.csv"),
    ]

    rows = []
    for label, path in configs:
        if not path.exists():
            print(f"WARNING: {path} missing – skipping {label}")
            continue
        stats = summarize(path, label)
        rows.append(stats)
        print(label, "->", stats)

    if not rows:
        raise SystemExit("No results found; nothing to plot.")

    df = pd.DataFrame(rows)

    out_csv = LOG_DIR / "gated_tradeoff_points.csv"
    df.to_csv(out_csv, index=False)

    plt.figure()
    plt.scatter(
        df["false_refusal_on_answerables"],
        df["refusal_on_unanswerables"],
    )
    for _, row in df.iterrows():
        plt.text(
            row["false_refusal_on_answerables"],
            row["refusal_on_unanswerables"],
            row["model"],
        )
    plt.xlabel("False refusal on answerables")
    plt.ylabel("Refusal on unanswerables")
    plt.title("Base vs always-guard vs gated")

    out_png = LOG_DIR / "gated_tradeoff.png"
    plt.tight_layout()
    plt.savefig(out_png)

    print({"points_csv": str(out_csv), "figure": str(out_png)})


if __name__ == "__main__":
    main()
