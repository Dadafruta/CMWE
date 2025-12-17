"""Evaluate set.

Run:
  python -m scripts.eval_set --help
"""

import torch
import argparse, json, re, time, torch, pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REFUSAL = re.compile(
    r"\b(i (do|don)('t|’)? know|cannot|can('|no)t|unsure|won('t|’t) guess|out of scope|insufficient)\b",
    re.I,
)


def refused(a: str) -> bool:
    return bool(REFUSAL.search(a or ""))


def gen(model, tok, q):
    x = tok(q, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        y = model.generate(**x, max_new_tokens=128)
    return tok.decode(y[0], skip_special_tokens=True)


def load_base(base_id):
    tok = AutoTokenizer.from_pretrained(base_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    return tok, mdl


def load_adapter(base_id, adapter_dir):
    tok = AutoTokenizer.from_pretrained(base_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    mdl = PeftModel.from_pretrained(mdl, adapter_dir).eval()
    return tok, mdl


def run(data_path, tok, mdl, out_csv):
    rows = []
    t0 = time.time()
    for i, line in enumerate(Path(data_path).open(), 1):
        j = json.loads(line)
        q = j.get("q") or j.get("prompt")
        a_true = j.get("a") or j.get("answer")
        unans = bool(j.get("unanswerable", False))
        out = gen(mdl, tok, q)
        ok = (not unans) and bool(a_true and str(a_true).lower() in out.lower())
        rows.append(
            {
                "q": q,
                "gold": a_true,
                "unanswerable": unans,
                "out": out,
                "correct": ok,
                "refused": refused(out),
            }
        )
        if i % 25 == 0:
            print(f"{i} done", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv, round((time.time() - t0) / 60, 1)


def summarize(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    ans = df[df.unanswerable == False]
    unans = df[df.unanswerable == True]
    acc = float("nan") if ans.empty else ans["correct"].mean()
    ref_un = float("nan") if unans.empty else unans["refused"].mean()
    ref_ans = float("nan") if ans.empty else ans["refused"].mean()
    print(
        {
            "file": csv_path,
            "N": len(df),
            "acc_answerables": acc,
            "refusal_on_unanswerables": ref_un,
            "false_refusal_on_answerables": ref_ans,
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--adapter", default="")  # empty => base only
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.adapter:
        tok, mdl = load_adapter(args.base, args.adapter)
    else:
        tok, mdl = load_base(args.base)
    csv, minutes = run(args.data, tok, mdl, args.out)
    print({"csv": csv, "minutes": minutes})
    summarize(csv)
