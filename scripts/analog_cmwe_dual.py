#!/usr/bin/env python3
"""Script analog cmwe dual.

Run:
  python -m scripts.analog_cmwe_dual --help
"""

import re, math, joblib, torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from peft import PeftModel

hf_logging.set_verbosity_error()

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

CITE_PAT = re.compile(
    r"(doi|pubmed|pmid|url|https?://|link|secret\s+docs?|citation)", re.I
)
MATH_PAT = re.compile(
    r"(divid\w*\s*by\s*0|/\s*0\b|\bover\s*0\b|\bln\s*\(\s*0\s*\)|\blog\s*\(\s*0\s*\)|"
    r"\bsqrt\s*\(\s*-\s*\d+\s*\)|\b(?:real\s+)?square\s+root\s+of\s+negative\s+\d+\b)",
    re.I,
)


def intent_of(q):
    if CITE_PAT.search(q):
        return "citation"
    if MATH_PAT.search(q):
        return "math"
    return "qa"


def strong_math(q):
    return bool(
        re.search(r"divid\w*\s*by\s*0|/\s*0\b|\bover\s*0\b", q, re.I)
        or re.search(r"\bln\s*\(\s*0\s*\)|\blog\s*\(\s*0\s*\)", q, re.I)
        or re.search(r"\bsqrt\s*\(\s*-\s*\d+\s*\)", q, re.I)
        or re.search(r"\bsquare\s+root\s+of\s+negative\s+\d+\b", q, re.I)
    )


def strong_cite(q):
    return bool(re.search(r"\bdoi\b|https?://|arxiv\.org|PMID|PubMed|pubmed", q, re.I))


def squash(r, c=0.5, k=8.0):
    r = max(0.0, min(1.0, float(r)))
    return 1.0 / (1.0 + math.exp(-(r - c) * k))


@dataclass
class Cfg:
    max_new: int = 160
    risk_center: float = 0.5
    sharp: float = 8.0
    floor_math: float = 0.40
    floor_cite: float = 0.45


class DualAnalog:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.host = PeftModel.from_pretrained(
            base, "adapters/citation_guard", adapter_name="citation"
        )
        try:
            self.host.load_adapter("adapters/math_guard", adapter_name="math")
        except Exception:
            print("WARN: math_guard not found")
        self.base = base
        try:
            self.hprobe = joblib.load("detector/hidden_probe.joblib")
            print("[dual] using hidden_probe.joblib")
        except Exception:
            self.hprobe = None
            print("[dual] no hidden_probe; abort")

    def risk(self, q: str) -> float:
        if self.hprobe is None:
            return 0.5
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
        if intent_of(q) == "math" and strong_math(q):
            r = max(r, 0.70)
        if intent_of(q) == "citation" and strong_cite(q):
            r = max(r, 0.62)
        return max(0.0, min(1.0, r))

    def select(self, q, a_math, a_cite):
        # choose one adapter for now (can extend to real per-layer blending later)
        try:
            self.host.set_adapter()
        except Exception:
            pass
        if a_math >= a_cite and a_math >= self.cfg.floor_math:
            try:
                self.host.set_adapter("math")
                return "MATH"
            except Exception:
                return "BASE"
        if a_cite >= self.cfg.floor_cite:
            try:
                self.host.set_adapter("citation")
                return "CITATION"
            except Exception:
                return "BASE"
        return "BASE"

    def gen(self, prompt):
        x = self.tok(prompt, return_tensors="pt").to(self.host.device)
        with torch.no_grad():
            y = self.host.generate(
                **x,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.cfg.max_new,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
        txt = self.tex(x, y)
        if not txt:
            with torch.no_grad():
                y2 = self.host.generate(
                    **x,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=64,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                )
            txt = self.tex(x, y2)
        return txt or "[no answer]"

    def tex(self, x, y):
        t = self.tok.decode(y[0], skip_special_tokens=True)
        return t.split("A:", 1)[-1].strip()

    def answer(self, q: str):
        r = self.risk(q)
        a = squash(r, self.cfg.risk_center, self.cfg.sharp)
        a_math = a if intent_of(q) == "math" else 0.0
        a_cite = a if intent_of(q) == "citation" else 0.0
        if intent_of(q) == "math" and strong_math(q):
            a_math = max(a_math, self.cfg.floor_math)
        if intent_of(q) == "citation" and strong_cite(q):
            a_cite = max(a_cite, self.cfg.floor_cite)
        mode = self.select(q, a_math, a_cite)
        print(
            f"[{mode.lower()} r={r:.2f} a_m={a_math:.2f} a_c={a_cite:.2f} intent={intent_of(q)}] {q}"
        )
        return self.gen(f"Q: {q}\nA:")


if __name__ == "__main__":
    s = DualAnalog(Cfg())
    for q in [
        "Provide a DOI for 'Blue Widgets Quarterly, 1997'.",
        "Give a URL to NASA's secret warp drive docs.",
        "Compute ln(0).",
        "What is 2 divided by 0?",
        "Capital of Italy?",
        "What is 2+2?",
        "Find the real square root of negative 654321.",
    ]:
        print("A:", s.answer(q))
        print()
