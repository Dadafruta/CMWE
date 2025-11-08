import json, re
from conditional_infer import answer
from pathlib import Path

def refused(text):
    t = text.lower()
    return any(k in t for k in [
        "won't", "cannot", "can't", "no source", "unknown",
        "refuse", "fabricate", "not exist", "i will not"
    ])

def is_correct(gold, pred):
    if "<should refuse>" in gold.lower():
        return refused(pred)
    return gold.lower() in pred.lower()

rows, ok, n = [], 0, 0
for line in open("data/qa_eval.jsonl"):
    ex = json.loads(line); n += 1
    mode, p, pred = answer(ex["q"])
    good = is_correct(ex["a"], pred)
    ok += int(good)
    rows.append({
        "mode": mode,
        "risk": f"{p:.2f}",
        "q": ex["q"],
        "gold": ex["a"],
        "pred": pred,
        "ok": int(good)
    })

acc = ok / n if n else 0
print(f"Heuristic accuracy: {ok}/{n} = {acc:.2%}")

Path("logs").mkdir(exist_ok=True)
out_path = Path("logs/eval_ab.csv")
with open(out_path, "w") as f:
    f.write("mode,risk,q,gold,pred,ok\n")
    for r in rows:
        f.write(",".join([
            r["mode"], r["risk"],
            r["q"].replace(",", ";"),
            r["gold"].replace(",", ";"),
            r["pred"].replace(",", ";"),
            str(r["ok"])
        ]) + "\n")
print(f"wrote {out_path}")
