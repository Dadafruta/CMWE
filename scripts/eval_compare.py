import re, time, json, math
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from analog_cmwe import CMWE, Cfg

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")

def gen_base(q, max_new=128):
    prompt = f"Q: {q}\nA:"
    x = tok(prompt, return_tensors="pt").to(base.device)
    with torch.no_grad():
        y = base.generate(
            **x,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(y[0], skip_special_tokens=True).split("A:", 1)[-1].strip()

def is_refusal(text: str) -> bool:
    t = text.lower()
    return bool(re.search(
        r"\b(can.?not|cannot|can.t|won.?t|undefined|not defined|no (?:source|doi|pmid)|cannot fabricate)",
        t
    ))

def judge(item, pred: str):
    task = item["type"]
    gold = item["a"]
    if task.startswith("refuse"):
        return 1 if is_refusal(pred) else 0
    if task == "math":
        try:
            g = float(eval(gold.replace("^", "**")))
        except Exception:
            return int(pred.strip().lower() == gold.strip().lower())
        m = re.search(r"[-+]?\d+(\.\d+)?", pred.replace(",", ""))
        if not m:
            return 0
        p = float(m.group(0))
        return int(abs(p - g) < 1e-6)
    if task == "qa":
        return int(gold.lower() in pred.lower())
    return 0

def run(mode: str, data):
    out = []
    router = CMWE(Cfg()) if mode == "cmwe" else None
    for ex in data:
        q = ex["q"]
        t0 = time.time()
        if mode == "base":
            ans = gen_base(q)
        else:
            ans = router.answer(q)
        dt = time.time() - t0
        ok = judge(ex, ans)
        out.append({"mode": mode, "type": ex["type"], "q": q, "ans": ans, "ok": ok, "latency_s": dt})
    return out

if __name__ == "__main__":
    data = [json.loads(l) for l in Path("data/bench.jsonl").open()]
    rows = []
    rows += run("base", data)
    rows += run("cmwe", data)
    df = pd.DataFrame(rows)
    print("\nPer-mode accuracy:")
    print(df.pivot_table(index="mode", values="ok", aggfunc="mean"))
    print("\nPer-task accuracy:")
    print(df.pivot_table(index=["mode", "type"], values="ok", aggfunc="mean"))
    df.to_csv("logs/ab_results.csv", index=False)
    print("\nSaved: logs/ab_results.csv")
