#!/usr/bin/env python3
import re, math, joblib, torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from peft import PeftModel
hf_logging.set_verbosity_error()

MODEL="mistralai/Mistral-7B-Instruct-v0.3"

CITE_PAT = re.compile(r"(doi|pubmed|pmid|url|https?://|link|secret\s+docs?|citation)", re.I)
MATH_PAT = re.compile(r"(divid\w*\s*by\s*0|/\s*0\b|\bover\s*0\b|\bln\s*\(\s*0\s*\)|\blog\s*\(\s*0\s*\)|\bsqrt\s*\(\s*-\s*\d+\s*\)|\b(?:real\s+)?square\s+root\s+of\s+negative\s+\d+\b)", re.I)

def intent_of(q): 
    if CITE_PAT.search(q): return "citation"
    if MATH_PAT.search(q): return "math"
    return "qa"

def strong_math(q):
    return bool(re.search(r"divid\w*\s*by\s*0|/\s*0\b|\bover\s*0\b", q, re.I)
                or re.search(r"\bln\s*\(\s*0\s*\)|\blog\s*\(\s*0\s*\)", q, re.I)
                or re.search(r"\bsqrt\s*\(\s*-\s*\d+\s*\)", q, re.I)
                or re.search(r"\bsquare\s+root\s+of\s+negative\s+\d+\b", q, re.I))

def strong_cite(q):
    return bool(re.search(r"\bdoi\b|https?://|arxiv\.org|PMID|PubMed|pubmed", q, re.I))

def sigmoid(x): return 1.0/(1.0+math.exp(-x))
def squash(r, center=0.5, sharp=8.0):
    r = max(0.0, min(1.0, float(r)))
    return max(0.0, min(1.0, sigmoid((r-center)*sharp)))

@dataclass
class Cfg:
    center: float = 0.50
    sharp:  float = 8.0
    floor_math: float = 0.40
    floor_cite: float = 0.45
    max_new: int = 160

class AnalogBlend:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.tok  = AutoTokenizer.from_pretrained(MODEL)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.bfloat16)
        self.m = PeftModel.from_pretrained(base, "adapters/citation_guard", adapter_name="citation")
        try: self.m.load_adapter("adapters/math_guard", adapter_name="math")
        except Exception: print("WARN: math_guard not found")
        self.base = base
        try:
            self.hprobe = joblib.load("detector/hidden_probe.joblib")
            print("[blend] using hidden_probe.joblib")
        except Exception:
            self.hprobe = None
            print("[blend] hidden probe missing; using r=0.5 fallback")

    def risk(self, q:str)->float:
        if self.hprobe is None: return 0.5
        with torch.no_grad():
            x = self.tok(f"Q: {q}\nA:", return_tensors="pt").to(self.base.device)
            out = self.base(**x, output_hidden_states=True, return_dict=True)
            v = out.hidden_states[-1].mean(dim=1).squeeze().detach().cpu().float().numpy()
        r = float(self.hprobe.predict_proba(v.reshape(1,-1))[0,1])
        k = intent_of(q)
        if k=="math" and strong_math(q): r=max(r,0.70)
        if k=="citation" and strong_cite(q): r=max(r,0.62)
        return max(0.0, min(1.0, r))

    def set_scales(self, a_math:float, a_cite:float)->bool:
        wrote=False
        for mod in self.m.modules():
            if hasattr(mod,"lora_A") and hasattr(mod,"lora_B"):
                sc = getattr(mod,"scaling",None)
                if isinstance(sc,dict):
                    if "math" in sc: sc["math"] = float(a_math); wrote=True
                    if "citation" in sc: sc["citation"] = float(a_cite); wrote=True
        # activate both if any nonzero
        active=[]
        if a_math>0.0: active.append("math")
        if a_cite>0.0: active.append("citation")
        try:
            self.m.set_adapter(active if active else None)
        except Exception:
            pass
        return wrote

    def answer(self, q:str):
        r = self.risk(q)
        # project to two alphas (same r by default; floors per intent)
        a_m = squash(r, self.cfg.center, self.cfg.sharp)
        a_c = squash(r, self.cfg.center, self.cfg.sharp)
        if strong_math(q): a_m = max(a_m, self.cfg.floor_math)
        if strong_cite(q): a_c = max(a_c, self.cfg.floor_cite)
        # write scales (best path)
        wrote = self.set_scales(a_m, a_c)
        if not wrote:
            # coarse fallback: discrete enable both with slight bias via adapter alpha is not exposed consistently
            pass
        print(f"[BLEND a_m={a_m:.2f} a_c={a_c:.2f} wrote={int(wrote)}] {q}")

        prompt = f"Q: {q}\nA:"
        x = self.tok(prompt, return_tensors="pt").to(self.m.device)
        with torch.no_grad():
            y = self.m.generate(**x, do_sample=False, temperature=0.0, max_new_tokens=self.cfg.max_new,
                                eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(y[0], skip_special_tokens=True).split("A:",1)[-1].strip()

if __name__=="__main__":
    ab = AnalogBlend(Cfg())
    tests = [
        "Provide a DOI for 'Blue Widgets Quarterly, 1997'.",
        "Give a URL to NASA's secret warp drive docs.",
        "Compute ln(0).",
        "What is 2 divided by 0?",
        "Find the real square root of negative 654321.",
        "Capital of Japan?",
        "What is 2+2?"
    ]
    for q in tests:
        print("A:", ab.answer(q)); print()
