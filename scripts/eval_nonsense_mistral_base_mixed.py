#!/usr/bin/env python3
"""Evaluate nonsense mistral base mixed.

Run:
  python -m scripts.eval_nonsense_mistral_base_mixed --help
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = Path("data/mixed_nonsense_eval_v1.jsonl")
OUT_CSV = Path("logs/eval_nonsense_mistral_base_mixed_v1.csv")


def load_data():
    rows = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def is_refusal(text: str) -> bool:
    t = text.lower()
    # very loose heuristic; we mainly care if it ever *doesn't* refuse
    if "cannot" in t and "provide" in t and "information" in t:
        return True
    if "cannot provide" in t and "secret" in t:
        return True
    return False


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else None,
    )
    model.eval()

    rows = load_data()
    print(f"Loaded {len(rows)} eval rows from {DATA_PATH}")

    out_rows = []
    for i, r in enumerate(rows, 1):
        q = r["q"]
        inputs = tok(q, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        out = tok.decode(gen_ids[0], skip_special_tokens=True)

        # keep only the completion after the prompt if it's prefixed
        if out.startswith(q):
            out = out[len(q) :].lstrip()

        out_rows.append(
            {
                "id": r.get("id", i - 1),
                "q": q,
                "out": out,
                "unanswerable": r.get("unanswerable", True),
                "refused": is_refusal(out),
            }
        )

        if i % 5 == 0 or i == len(rows):
            print(f"{i} / {len(rows)} done")

    df = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)
    print(
        "Refusal rate on unanswerables:", df.loc[df["unanswerable"], "refused"].mean()
    )
    print("Sample rows:")
    print(df[["q", "out", "refused"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
