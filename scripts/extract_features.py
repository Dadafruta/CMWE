"""Extract features.

Run:
  python -m scripts.extract_features --help
"""

import json, math, itertools
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# Try 4-bit if bitsandbytes is present; otherwise load normally.
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto", torch_dtype=torch.bfloat16
        )
    return tok, m


tok, m = load_model()


def gen_once(q, temperature=0.7, max_new=64):
    prompt = f"Q: {q}\nA:"
    x = tok(prompt, return_tensors="pt").to(m.device)
    with torch.no_grad():
        y = m.generate(
            **x,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=max_new,
            output_scores=True,
            return_dict_in_generate=True,
        )
    text = (
        tok.decode(y.sequences[0], skip_special_tokens=True).split("A:", 1)[-1].strip()
    )
    gen_ids = y.sequences[0][x["input_ids"].shape[1] :]
    probs = []
    for t, s in zip(gen_ids, y.scores):
        p = torch.softmax(s[0], dim=-1)[t.item()].item()
        probs.append(max(p, 1e-12))
    last_ent = float(
        (
            -torch.softmax(y.scores[-1][0], dim=-1)
            * torch.log_softmax(y.scores[-1][0], dim=-1)
        )
        .sum()
        .item()
    )
    return text, probs, last_ent


def jaccard(a, b):
    A, B = set(a.lower().split()), set(b.lower().split())
    return len(A & B) / max(1, len(A | B))


rows = []
for line in Path("data/qa_eval.jsonl").read_text().splitlines():
    ex = json.loads(line)
    q = ex["q"]
    gold = ex["a"]
    outs, ents, mean_logp = [], [], []
    for _ in range(3):
        text, probs, ent = gen_once(q)
        outs.append(text)
        ents.append(ent)
        mean_logp.append(sum([math.log(p) for p in probs]) / max(1, len(probs)))
    sims = [jaccard(outs[i], outs[j]) for i, j in itertools.combinations(range(3), 2)]
    disagree = 1 - (sum(sims) / len(sims))
    rows.append(
        {
            "q": q,
            "gold": gold,
            "out1": outs[0],
            "out2": outs[1],
            "out3": outs[2],
            "mean_logp": sum(mean_logp) / len(mean_logp),
            "last_entropy": sum(ents) / len(ents),
            "disagree": disagree,
            "label": int("<should refuse>" in gold.lower()),
        }
    )

Path("logs").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("logs/features.csv", index=False)
print(f"wrote logs/features.csv with {len(rows)} rows")
