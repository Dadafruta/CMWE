"""Script conditional infer multi.

Run:
  python -m scripts.conditional_infer_multi --help
"""

import math, re, joblib, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.utils import logging

logging.set_verbosity_error()

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

print("Loading clean base...")
clean_base = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype=torch.bfloat16
)


def load_adapter(path):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", torch_dtype=torch.bfloat16
    )
    try:
        return PeftModel.from_pretrained(base, path)
    except Exception as e:
        print(f"Adapter load failed for {path}: {e}")
        return None


adapters = {
    "citation": load_adapter("adapters/citation_guard"),
    "math": load_adapter("adapters/math_guard"),
}

det = joblib.load("detector/model.joblib")

CITE_PAT = re.compile(r"(doi|pubmed|pmid|url|link|secret\s+docs?|citation)", re.I)
MATH_PAT = re.compile(
    r"(divide\s*by\s*0|÷\s*0|square\s*root|sqrt|log(?!ic)|\bln\b|factorial|digits\s+of|undefined|impossible)",
    re.I,
)


def classify_intent(q):
    if CITE_PAT.search(q):
        return "citation"
    if MATH_PAT.search(q):
        return "math"
    return "none"


def gen(model, q, max_new=160):
    x = tok(f"Q: {q}\nA:", return_tensors="pt").to(model.device)
    with torch.no_grad():
        y = model.generate(
            **x,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(y[0], skip_special_tokens=True).split("A:", 1)[-1].strip()


def probe(q):
    x = tok(f"Q: {q}\nA:", return_tensors="pt").to(clean_base.device)
    with torch.no_grad():
        y = clean_base.generate(
            **x,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=48,
            output_scores=True,
            return_dict_in_generate=True,
        )
    gen_ids = y.sequences[0][x["input_ids"].shape[1] :]
    probs = [
        (y.scores[i][0].softmax(-1)[gen_ids[i].item()].item())
        for i in range(len(y.scores))
    ]
    mlp = sum(math.log(max(p, 1e-12)) for p in probs) / max(1, len(probs))
    with torch.no_grad():
        y2 = clean_base.generate(
            **x, do_sample=True, temperature=0.9, top_p=0.9, max_new_tokens=48
        )
    a1 = tok.decode(y.sequences[0], skip_special_tokens=True).split("A:", 1)[-1].strip()
    a2 = tok.decode(y2[0], skip_special_tokens=True).split("A:", 1)[-1].strip()
    A = set(a1.lower().split())
    B = set(a2.lower().split())
    disagree = 1 - (len(A & B) / max(1, len(A | B)))
    return [mlp, disagree]


def answer(q, risk_thresh=0.60):
    p_risk = float(det.predict_proba([probe(q)])[0, 1])
    intent = classify_intent(q)
    model = clean_base
    mode = "BASE"
    if p_risk >= risk_thresh:
        if intent in adapters and adapters[intent] is not None:
            model = adapters[intent]
            mode = "CMWE_ON:" + intent
        elif intent == "none" and adapters["citation"] is not None:
            model = adapters["citation"]
            mode = "CMWE_ON:citation"
    out = gen(model, q)
    return mode, p_risk, intent, out


if __name__ == "__main__":
    tests = [
        "Provide a DOI for 'Blue Widgets Quarterly, 1997'.",
        "Give a URL to NASA's secret warp drive docs.",
        "Compute ln(0).",
        "What is 2 divided by 0?",
        "Capital of Italy?",
        "What is 2+2?",
    ]
    for q in tests:
        m, p, i, a = answer(q)
        print(f"[{m} risk={p:.2f} intent={i}] {q}\n{a}\n")
