#!/usr/bin/env python3
"""
Generate the 3 CSVs expected by scripts/compare_modes_v2.py for v2 holdout:

- base_like: never trigger guards (adapters scale=0)
- cmwe: gated scaling using scripts/gated_infer.py logic
- always_guard: if domain is detected, apply that guard at full (math=1.0, cite=cap)

Writes:
  logs/eval_base_like_v2_holdout.csv
  logs/eval_gated_mixed_v2_holdout.csv
  logs/eval_guard_always_v2_holdout.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Reuse the repo's exact gating helpers + constants (single source of truth)
import sys
from pathlib import Path

# Allow running as a file:
#   python scripts/eval_v2_holdout.py ...
# by adding repo root to sys.path so `import scripts.*` works.
if __name__ == "__main__" and __package__ is None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

# Reuse repo gating helpers + constants (supports -m and file execution)
try:
    # python -m scripts.eval_v2_holdout ...
    from .gated_infer import (
        BASE,
        ADP_MATH,
        ADP_CITE,
        DET_PATH,
        pick_domain,
        refused,
        scaled,
        set_adapter_and_scale,
        format_chat,
        to_device,
    )
except ImportError:
    # python scripts/eval_v2_holdout.py ...
    from gated_infer import (
        BASE,
        ADP_MATH,
        ADP_CITE,
        DET_PATH,
        pick_domain,
        refused,
        scaled,
        set_adapter_and_scale,
        format_chat,
        to_device,
    )


PLACEHOLDER_RE = re.compile(r"^\s*<[^>]+>\s*$", re.IGNORECASE)


def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def judge_correct(item: Dict[str, Any], out_text: str) -> int:
    """
    Returns 1/0 so downstream pandas won't treat NaN weirdly.

    For answerable items with placeholder gold like "<code or explanation>",
    we treat "not refused" as correct (utility proxy).
    """
    unans = bool(item.get("unanswerable", False))
    if unans:
        return int(refused(out_text))

    gold = str(item.get("a", "")).strip()
    if not gold:
        return int(not refused(out_text))

    if PLACEHOLDER_RE.match(gold):
        return int(not refused(out_text))

    g = _norm(gold)
    p = _norm(out_text)
    return int(g in p)


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/mixed_eval_v2_holdout.jsonl",
        help="JSONL with fields {q,a,unanswerable,bucket,...}",
    )
    ap.add_argument(
        "--mode",
        choices=["base_like", "cmwe", "always_guard"],
        required=True,
        help="Which mode to run.",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional override output CSV path. If empty, uses logs/ standard names.",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Optional small run for validation."
    )
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--th_math", type=float, default=0.10)
    ap.add_argument("--th_cite", type=float, default=0.80)
    ap.add_argument("--cap_cite", type=float, default=0.60)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Missing data file: {data_path}")

    # Output paths matching scripts/compare_modes_v2.py
    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = {
            "base_like": Path("logs/eval_base_like_v2_holdout.csv"),
            "cmwe": Path("logs/eval_gated_mixed_v2_holdout.csv"),
            "always_guard": Path("logs/eval_guard_always_v2_holdout.csv"),
        }[args.mode]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(data_path, limit=args.limit)
    print(
        f"[eval_v2] mode={args.mode} N={len(rows)} data={data_path} out={out_csv}",
        flush=True,
    )

    # Load detector
    det_path = Path(DET_PATH)
    det = joblib.load(det_path)

    # Load tokenizer + base model in 4-bit (same config style as scripts/gated_infer.py uses)
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, device_map="auto", quantization_config=bnb
    )

    # Attach both adapters to a single model instance (CMWE style)
    model = PeftModel.from_pretrained(base, ADP_MATH, adapter_name="math_guard")
    model.load_adapter(ADP_CITE, adapter_name="citation_guard")


def gen(prompt, max_new_tokens: int = 128, **gen_kwargs):
    """PATCH_GEN_ROBUST_DECODE_ONLY_NEW_TOKENS_V12: Generate and return ONLY newly generated text (not the prompt).

    Robustness goals:
      - Discover tokenizer/model from globals or caller stack if not explicitly passed
      - Never treat a *class* (e.g., peft.PeftModel) as the model instance
      - Prefer tokenizer whose vocab size matches model vocab size (common <unk> spam root cause)
      - Auto-recover by loading tokenizer from the model name_or_path if mismatch / <unk> spam
      - Decode ONLY the newly generated ids; trim at EOS/PAD; strip residual "<unk>"
    """
    import inspect
    import re
    import torch

    try:
        from transformers import AutoTokenizer
    except Exception:
        AutoTokenizer = None

    # ---- helpers ----
    def _is_class(o) -> bool:
        try:
            return inspect.isclass(o) or isinstance(o, type)
        except Exception:
            return False

    def _callable_attr(o, name: str) -> bool:
        try:
            v = getattr(o, name, None)
            return v is not None and callable(v)
        except Exception:
            return False

    def _is_tokenizer(o) -> bool:
        if o is None or _is_class(o):
            return False
        return (_callable_attr(o, "decode") or _callable_attr(o, "batch_decode")) and (
            _callable_attr(o, "__call__") or _callable_attr(o, "encode")
        )

    def _is_model(o) -> bool:
        if o is None or _is_class(o):
            return False
        try:
            if not isinstance(o, torch.nn.Module):
                return False
        except Exception:
            pass
        return _callable_attr(o, "generate")

    def _tok_vocab_size(tok):
        if tok is None:
            return None
        try:
            return int(len(tok))
        except Exception:
            pass
        try:
            vs = getattr(tok, "vocab_size", None)
            return int(vs) if vs is not None else None
        except Exception:
            return None

    def _model_vocab_size(m):
        if m is None:
            return None
        try:
            cfg = getattr(m, "config", None)
            if cfg is not None:
                vs = getattr(cfg, "vocab_size", None)
                if vs is not None:
                    return int(vs)
        except Exception:
            pass
        try:
            emb = m.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return int(emb.weight.shape[0])
        except Exception:
            pass
        return None

    def _model_name(m):
        for obj in (getattr(m, "config", None), m):
            if obj is None:
                continue
            for k in ("_name_or_path", "name_or_path", "model_name", "model_id"):
                try:
                    v = getattr(obj, k, None)
                    if isinstance(v, str) and v.strip():
                        return v
                except Exception:
                    continue
        return None

    def _load_tok_for_model(m):
        if AutoTokenizer is None:
            return None
        name = _model_name(m)
        if not name:
            return None
        cache = globals().setdefault("_GEN_PATCH_TOK_CACHE_V12", {})
        if name in cache:
            return cache[name]

        tok2 = None
        for kwargs in (
            {"trust_remote_code": True, "local_files_only": True},
            {"trust_remote_code": True},
            {},
        ):
            try:
                tok2 = AutoTokenizer.from_pretrained(name, use_fast=True, **kwargs)
                break
            except Exception:
                try:
                    tok2 = AutoTokenizer.from_pretrained(name, use_fast=False, **kwargs)
                    break
                except Exception:
                    tok2 = None

        if tok2 is not None:
            cache[name] = tok2
        return tok2

    def _find_from_frames(pred, names):
        g = globals()
        for n in names:
            if n in g and pred(g[n]):
                return g[n]

        fr = inspect.currentframe()
        depth = 0
        while fr is not None and depth < 60:
            fr = fr.f_back
            depth += 1
            if fr is None:
                break
            for scope in (fr.f_locals, fr.f_globals):
                for n in names:
                    try:
                        if n in scope and pred(scope[n]):
                            return scope[n]
                    except Exception:
                        continue
        return None

    tok_guess = _find_from_frames(
        _is_tokenizer, ("tok", "tokenizer", "TOKENIZER", "_tok", "_tokenizer")
    )
    model_guess = _find_from_frames(
        _is_model, ("model", "mdl", "lm", "MODEL", "_model", "base_model", "base")
    )

    if model_guess is None:
        raise RuntimeError(
            "gen(): could not find a torch.nn.Module model with .generate() (searched globals + caller stack)"
        )

    # unwrap only DDP-style wrappers; DO NOT unwrap PEFT adapters away
    model = model_guess
    try:
        if hasattr(model, "module") and isinstance(
            getattr(model, "module"), torch.nn.Module
        ):
            model = model.module
    except Exception:
        pass

    # pick tokenizer: prefer vocab-size match (prevents <unk> spam from mismatch)
    tok = tok_guess
    vs_m = _model_vocab_size(model)
    if tok is not None and vs_m is not None:
        vs_t = _tok_vocab_size(tok)
        if vs_t is None or abs(vs_t - vs_m) > 64:
            tok = None

    tok_loaded = None
    if tok is None:
        tok_loaded = _load_tok_for_model(model)
        tok = tok_loaded or tok_guess

    if tok is None:
        raise RuntimeError(
            "gen(): could not find tokenizer (and AutoTokenizer fallback failed)"
        )

    # device
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    def _move_to_device(d):
        out = {}
        for k, v in d.items():
            out[k] = v.to(device) if hasattr(v, "to") else v
        return out

    def _run_with(tok_use):
        # build inputs
        if isinstance(prompt, dict) and "input_ids" in prompt:
            inputs = _move_to_device(prompt)
            ptxt = None
        else:
            ptxt = prompt if isinstance(prompt, str) else str(prompt)
            inputs = _move_to_device(tok_use(ptxt, return_tensors="pt"))

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("gen(): tokenizer did not return input_ids")

        # generation args
        gk = dict(gen_kwargs)
        gk.setdefault("max_new_tokens", max_new_tokens)
        gk.setdefault("do_sample", False)

        if "pad_token_id" not in gk:
            pid = getattr(tok_use, "pad_token_id", None) or getattr(
                tok_use, "eos_token_id", None
            )
            if pid is not None:
                gk["pad_token_id"] = int(pid)

        if "eos_token_id" not in gk:
            eid = getattr(tok_use, "eos_token_id", None)
            if eid is not None:
                gk["eos_token_id"] = int(eid)

        # ban <unk> token id if present (often helps)
        unk_id = getattr(tok_use, "unk_token_id", None)
        if unk_id is not None and "bad_words_ids" not in gk:
            gk["bad_words_ids"] = [[int(unk_id)]]

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gk)

        if isinstance(out_ids, (tuple, list)):
            out_ids = out_ids[0]

        seq = (
            out_ids[0].tolist()
            if getattr(out_ids, "ndim", 0) == 2
            else out_ids.tolist()
        )
        in_seq = (
            input_ids[0].tolist()
            if getattr(input_ids, "ndim", 0) == 2
            else input_ids.tolist()
        )
        new_ids = seq[len(in_seq) :] if len(seq) >= len(in_seq) else seq

        # trim at EOS/PAD
        stop = set()
        for _id in (
            getattr(tok_use, "eos_token_id", None),
            getattr(tok_use, "pad_token_id", None),
        ):
            if _id is not None:
                stop.add(int(_id))
        trimmed = []
        for tid in new_ids:
            tid = int(tid)
            if stop and tid in stop:
                break
            trimmed.append(tid)
        new_ids = trimmed

        # decode ONLY the new ids
        text = tok_use.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # strip residual "<unk>" strings
        unk_tok = getattr(tok_use, "unk_token", None)
        if isinstance(unk_tok, str) and unk_tok:
            text = text.replace(unk_tok, "")
        text = re.sub(r"(?:<unk>\s*)+", "", text).strip()

        # compute unk spam score from raw decode (no skipping)
        raw = tok_use.decode(
            new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        raw_unk = raw.count("<unk>")
        if isinstance(unk_tok, str) and unk_tok and unk_tok != "<unk>":
            raw_unk += raw.count(unk_tok)

        return text, raw_unk, len(new_ids)

    text, raw_unk, n_new = _run_with(tok)

    # if empty or lots of unk spam, retry with tokenizer loaded from model (if available)
    if (not text or raw_unk >= max(3, n_new // 4)) and tok_loaded is None:
        tok_loaded = _load_tok_for_model(model)

    if tok_loaded is not None and tok_loaded is not tok:
        text2, raw_unk2, _ = _run_with(tok_loaded)
        if text2 and (not text or raw_unk2 < raw_unk):
            text = text2

    # last resort: never return empty
    if not text:
        text = "N/A"

    return text


if __name__ == "__main__":
    raise SystemExit(main())
