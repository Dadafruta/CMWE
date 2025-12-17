"""Script conditional infer.

Run:
  python -m scripts.conditional_infer --help
"""

import math, joblib, torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.utils import logging

logging.set_verbosity_error()

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

# Base model
base = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype=torch.bfloat16
)

# Load adapter on top of the same base
guard = PeftModel.from_pretrained(base, "adapters/citation_guard")

# Detector
det = joblib.load("detector/model.joblib")


def gen_once(model, q, temperature=0.7, max_new=64):
    x = tok(f"Q: {q}\nA:", return_tensors="pt").to(model.device)
    with torch.no_grad():
        y = model.generate(
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
    # token probs for the generated suffix
    gen_ids = y.sequences[0][x["input_ids"].shape[1] :]
    probs = []
    for t, s in zip(gen_ids, y.scores):
        p = torch.softmax(s[0], dim=-1)[t.item()].item()
        probs.append(max(p, 1e-12))
    mean_logp = sum(math.log(p) for p in probs) / max(1, len(probs))
    return text, mean_logp


def disagree_score(q, samples=3):
    outs = []
    for _ in range(samples):
        t, _ = gen_once(base, q)
        outs.append(t)

    def jac(a, b):
        A = set(a.lower().split())
        B = set(b.lower().split())
        return len(A & B) / max(1, len(A | B))

    sims = [
        jac(outs[i], outs[j]) for i in range(samples) for j in range(i + 1, samples)
    ]
    return 1 - (sum(sims) / len(sims))


def features(q):
    # one short probe for logp + small disagreement probe
    _, mlp = gen_once(base, q, temperature=0.7, max_new=48)
    disagree = disagree_score(q, samples=3)
    return [mlp, disagree]


def answer(q, thresh=0.5):
    X = [features(q)]
    p_risk = float(det.predict_proba(X)[0, 1])
    model = guard if p_risk >= thresh else base
    x = tok(f"Q: {q}\nA:", return_tensors="pt").to(model.device)
    y = model.generate(
        **x,
        do_sample=False,
        temperature=0.2 if model is guard else 0.3,
        max_new_tokens=128,
    )
    out = tok.decode(y[0], skip_special_tokens=True).split("A:", 1)[-1].strip()
    mode = "CMWE_ON" if model is guard else "BASE"
    return mode, p_risk, out


if __name__ == "__main__":
    tests = [
        "Capital of France?",
        "Provide a DOI for 'Blue Widgets Quarterly, 1997'.",
        "Give a URL to NASA's secret warp drive docs.",
        "Prime minister of UK in 2019?",
    ]
    for q in tests:
        mode, p, a = answer(q)
        print(f"[{mode} risk={p:.2f}] {q}\n{a}\n")
