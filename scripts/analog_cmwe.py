#!/usr/bin/env python3
"""Script analog cmwe.

Run:
  python -m scripts.analog_cmwe --help
"""

import os
import re
import math
import sys
import joblib
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from peft import PeftModel

hf_logging.set_verbosity_error()

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CSV = "logs/analog_metrics.csv"

CITE_PAT = re.compile(
    r"(doi|pubmed|pmid|url|https?://|link|secret\s+docs?|citation)", re.I
)
MATH_PAT = re.compile(
    r"(divid\w*\s*by\s*0|/\s*0\b|\bover\s*0\b|\bln\s*\(\s*0\s*\)|\blog\s*\(\s*0\s*\)|\bsqrt\s*\(\s*-\s*\d+\s*\)|\b(?:real\s+)?square\s+root\s*of\s*negative\s+\d+\b)",
    re.I,
)


def classify_intent(q: str) -> str:
    if CITE_PAT.search(q):
        return "citation"
    if MATH_PAT.search(q):
        return "math"
    return "qa"


def strong_math(q: str) -> bool:
    return bool(
        re.search(r"divid\w*\s*by\s*0|/\s*0\b|\bover\s*0\b", q, re.I)
        or re.search(r"\bln\s*\(\s*0\s*\)|\blog\s*\(\s*0\s*\)", q, re.I)
        or re.search(r"\bsqrt\s*\(\s*-\s*\d+\s*\)", q, re.I)
        or re.search(r"\bsquare\s+root\s+of\s+negative\s+\d+\b", q, re.I)
    )


def strong_cite(q: str) -> bool:
    return bool(re.search(r"\bdoi\b|https?://|arxiv\.org|PMID|PubMed|pubmed", q, re.I))


def alpha_from_risk(r: float, center: float, sharp: float) -> float:
    r = max(0.0, min(1.0, float(r)))
    return 1.0 / (1.0 + math.exp(-(r - center) * sharp))


@dataclass
class Cfg:
    risk_center: float = 0.50
    sharp: float = 8.0
    max_new: int = 128


class CMWE:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.model = PeftModel.from_pretrained(
            base, "adapters/citation_guard", adapter_name="citation"
        )
        try:
            self.model.load_adapter("adapters/math_guard", adapter_name="math")
        except Exception:
            print("WARN: math_guard adapter not found; math path will use base only")
        self.base = base
        # risk probe
        try:
            self.hprobe = joblib.load("detector/hidden_probe.joblib")
            print("[risk] using hidden_probe.joblib")
        except Exception:
            self.hprobe = None
            print("[risk] hidden probe missing; falling back to 0.5")

        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(CSV):
            with open(CSV, "w") as f:
                f.write("intent,risk,alpha,mode,answer_len\n")

    def risk(self, q: str) -> float:
        if self.hprobe is None:
            r = 0.5
        else:
            with torch.no_grad():
                x = self.tok(f"Q: {q}\nA:", return_tensors="pt").to(self.base.device)
                out = self.base(**x, output_hidden_states=True, return_dict=True)
                v = (
                    out.hidden_states[-1]
                    .mean(dim=1)
                    .squeeze()
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                r = float(self.hprobe.predict_proba(v.reshape(1, -1))[0, 1])
        k = classify_intent(q)
        if k == "math" and strong_math(q):
            r = max(r, 0.70)
        if k == "citation" and strong_cite(q):
            r = max(r, 0.62)
        return max(0.0, min(1.0, r))

    def select(self, k: str, a: float) -> str:
        try:
            self.model.set_adapter()
        except Exception:
            pass
        if a < 0.40 and k != "citation":
            return "BASE"
        if k == "math" and a >= 0.40:
            try:
                self.model.set_adapter("math")
                return "MATH"
            except Exception:
                return "BASE"
        if k == "citation" and a >= 0.45:
            try:
                self.model.set_adapter("citation")
                return "CITATION"
            except Exception:
                return "BASE"
        return "BASE"

    def gen(self, prompt: str) -> str:
        x = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            y = self.model.generate(
                **x,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.cfg.max_new,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
        out = self.tok.decode(y[0], skip_special_tokens=True).split("A:", 1)[-1].strip()
        if not out:
            with torch.no_grad():
                y2 = self.model.generate(
                    **x,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=64,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                )
            out = (
                self.tok.decode(y2[0], skip_special_tokens=True)
                .split("A:", 1)[-1]
                .strip()
            )
        return out

    def answer(self, q: str) -> str:
        k = classify_intent(q)
        r = self.risk(q)
        a = alpha_from_risk(r, self.cfg.risk_center, self.cfg.sharp)
        if k == "math" and strong_math(q):
            a = max(a, 0.40)
        if k == "citation" and strong_cite(q):
            a = max(a, 0.45)
        mode = self.select(k, a)
        print(f"[{mode.lower()} r={r:.2f} a={a:.2f} intent={k}]", q)
        out = self.gen(f"Q: {q}\nA:")
        # log metrics
        with open(CSV, "a") as f:
            f.write(f"{k},{r:.6f},{a:.6f},{mode},{len(out)}\n")
        return out


if __name__ == "__main__":
    cfg = Cfg()
    cmwe = CMWE(cfg)
    prompts = sys.argv[1:] or [
        "Provide a DOI for 'Blue Widgets Quarterly, 1997'.",
        "Give a URL to NASA's secret warp drive docs.",
        "Compute ln(0).",
        "What is 2 divided by 0?",
        "Find the real square root of negative 654321.",
        "Capital of Italy?",
        "What is 2+2?",
    ]
    for q in prompts:
        print("A:", cmwe.answer(q))
        print()
