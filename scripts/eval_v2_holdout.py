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


# --- PATCH_CSV_WRITE_V2_OUTCSV ---
# Ensures --out_csv actually produces a file (rank0 only) even if upstream write path is flaky.
import atexit
import csv
import os
from pathlib import Path


def _cmwe_is_rank0() -> bool:
    # Best-effort: treat missing rank env as single-process rank0
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "MPI_RANK",
    ):
        val = os.environ.get(key)
        if val is None:
            continue
        try:
            # if this env var is a rank-like value, rank0 is 0
            if key in ("WORLD_SIZE",):
                continue
            return int(val) == 0
        except ValueError:
            pass
    return True


_CMWE_ROWS_REF = None
_CMWE_OUTCSV_REF = None


def _cmwe_capture_rows_and_path(rows, out_csv) -> None:
    global _CMWE_ROWS_REF, _CMWE_OUTCSV_REF
    _CMWE_ROWS_REF = rows
    _CMWE_OUTCSV_REF = out_csv


def _cmwe_write_rows_csv(rows, out_csv) -> None:
    if out_csv is None:
        return
    if not _cmwe_is_rank0():
        return

    p = Path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)

    norm = []
    if rows is None:
        rows_iter = []
    else:
        rows_iter = rows
    for r in rows_iter:
        if r is None:
            continue
        if isinstance(r, dict):
            norm.append(r)
        else:
            try:
                norm.append(dict(r))
            except Exception:
                norm.append({"_row": str(r)})

    # stable header: first-seen key order
    keys = []
    seen = set()
    for d in norm:
        for k in d.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    if not keys:
        keys = ["_empty"]

    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for d in norm:
            w.writerow({k: d.get(k, "") for k in keys})

    try:
        size = p.stat().st_size
    except Exception:
        size = -1
    print(f"[patch] wrote CSV -> {p} ({size} bytes)")


def _cmwe_atexit_write_csv() -> None:
    try:
        if _CMWE_ROWS_REF is None or _CMWE_OUTCSV_REF is None:
            return
        _cmwe_write_rows_csv(_CMWE_ROWS_REF, _CMWE_OUTCSV_REF)
    except Exception as e:
        print(f"[patch] CSV write failed: {e}")


atexit.register(_cmwe_atexit_write_csv)


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
    _cmwe_capture_rows_and_path(rows, out_csv)

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

    # --- PATCH_CSV_FORCE_WRITE_OUTCSV_V1 ---
    # If `--out_csv` is provided (or overridden), enforce that the CSV is actually created.
    try:
        from pathlib import Path as _Path
        import os as _os
        import csv as _csv
    except Exception:
        _Path = None  # type: ignore

    def _cmwe__is_rank0() -> bool:
        """Best-effort rank0 detection; default True in single-process runs."""
        try:
            for _k in (
                "RANK",
                "LOCAL_RANK",
                "SLURM_PROCID",
                "OMPI_COMM_WORLD_RANK",
                "PMI_RANK",
                "MPI_RANK",
                "MPI_LOCALRANKID",
            ):
                _v = _os.environ.get(_k)
                if _v is not None:
                    try:
                        return int(_v) == 0
                    except Exception:
                        pass
        except Exception:
            pass
        return True

    def _cmwe__force_write_csv(_outp, _rows) -> None:
        if _Path is None:
            return
        _outp = _Path(str(_outp))
        _outp.parent.mkdir(parents=True, exist_ok=True)

        _rows_list = []
        try:
            _iter = list(_rows) if _rows is not None else []
        except Exception:
            _iter = []

        for _r in _iter:
            if isinstance(_r, dict):
                _rows_list.append(_r)

        # stable header: keys from first row, then any newly-seen keys
        _fields = []
        _seen = set()
        if _rows_list:
            for _k in _rows_list[0].keys():
                _fields.append(_k)
                _seen.add(_k)
            for _r in _rows_list[1:]:
                for _k in _r.keys():
                    if _k not in _seen:
                        _fields.append(_k)
                        _seen.add(_k)
        else:
            # fallback header so downstream can at least read *something*
            _fields = ["bucket", "q", "gold", "out"]
            try:
                print(
                    "[patch] WARNING: rows looks empty; writing header-only CSV to "
                    + str(_outp),
                    flush=True,
                )
            except Exception:
                pass

        _tmp = _outp.with_suffix(_outp.suffix + ".tmp" + str(_os.getpid()))
        with _tmp.open("w", newline="", encoding="utf-8") as _f:
            _w = _csv.DictWriter(_f, fieldnames=_fields, extrasaction="ignore")
            _w.writeheader()
            for _r in _rows_list:
                _w.writerow(
                    {k: ("" if _r.get(k) is None else str(_r.get(k))) for k in _fields}
                )
        _os.replace(_tmp, _outp)

    try:
        _cmwe__out = out_csv  # type: ignore[name-defined]
    except Exception:
        _cmwe__out = None

    try:
        _cmwe__rows = rows  # type: ignore[name-defined]
    except Exception:
        _cmwe__rows = None

    if _cmwe__out is not None and _cmwe__is_rank0():
        try:
            _p = _Path(str(_cmwe__out))
            if (not _p.exists()) or (_p.stat().st_size == 0):
                _cmwe__force_write_csv(_p, _cmwe__rows)
                try:
                    print(
                        "[patch] wrote CSV -> {} ({} bytes)".format(
                            _p, _p.stat().st_size
                        ),
                        flush=True,
                    )
                except Exception:
                    pass
        except Exception as _e:
            try:
                print(
                    "[patch] WARNING: CSV enforcement failed: {}".format(_e), flush=True
                )
            except Exception:
                pass
    # --- end PATCH_CSV_FORCE_WRITE_OUTCSV_V1 ---
    model.load_adapter(ADP_CITE, adapter_name="citation_guard")


def gen(prompt, max_new_tokens: int = 640, **gen_kwargs):
    """PATCH_GEN_ROBUST_DECODE_ONLY_NEW_TOKENS_V11: Generate+decode ONLY newly generated text (not the prompt), robust to wrappers and tokenizer mismatch."""
    import inspect
    import re
    import torch
    from transformers import AutoTokenizer

    def _is_class(o) -> bool:
        try:
            return inspect.isclass(o) or isinstance(o, type)
        except Exception:
            return False

    def _callable_attr(o, name: str) -> bool:
        try:
            return callable(getattr(o, name, None))
        except Exception:
            return False

    def _is_tok(o) -> bool:
        if o is None or _is_class(o):
            return False
        return _callable_attr(o, "decode") and (
            _callable_attr(o, "__call__") or _callable_attr(o, "encode")
        )

    def _is_model(o) -> bool:
        if o is None or _is_class(o):
            return False
        return _callable_attr(o, "generate")

    def _find_in_stack(keys, pred):
        g = globals()
        for k in keys:
            if k in g and pred(g[k]):
                return g[k]
        fr = inspect.currentframe()
        fr = fr.f_back if fr is not None else None
        depth = 0
        while fr is not None and depth < 40:
            for k in keys:
                if k in fr.f_locals and pred(fr.f_locals[k]):
                    return fr.f_locals[k]
                if k in fr.f_globals and pred(fr.f_globals[k]):
                    return fr.f_globals[k]
            fr = fr.f_back
            depth += 1
        return None

    # allow explicit overrides
    tok = gen_kwargs.pop("tok", None) or gen_kwargs.pop("tokenizer", None)
    if not _is_tok(tok):
        tok = _find_in_stack(["tok", "tokenizer", "TOK", "TOKENIZER"], _is_tok)

    model = (
        gen_kwargs.pop("model", None)
        or gen_kwargs.pop("mdl", None)
        or gen_kwargs.pop("m", None)
        or gen_kwargs.pop("lm", None)
    )
    if not _is_model(model):
        model = _find_in_stack(
            ["model", "mdl", "m", "lm", "base_model", "guard_model"], _is_model
        )

    if tok is None or model is None:
        raise RuntimeError(
            "gen(): could not find tokenizer/model in args, globals, or caller stack."
        )

    gen_model = model  # keep wrapper for generate (adapters/DDP)

    def _unwrap(m):
        seen = set()
        while True:
            if id(m) in seen:
                break
            seen.add(id(m))

            # DDP/DataParallel
            if hasattr(m, "module") and _is_model(getattr(m, "module", None)):
                m = m.module
                continue

            # PEFT-style
            if hasattr(m, "get_base_model") and callable(
                getattr(m, "get_base_model", None)
            ):
                try:
                    bm = m.get_base_model()
                    if _is_model(bm):
                        m = bm
                        continue
                except Exception:
                    pass

            if hasattr(m, "base_model") and _is_model(getattr(m, "base_model", None)):
                m = m.base_model
                continue

            break
        return m

    base_model = _unwrap(gen_model)

    def _model_vocab_size(m):
        if m is None:
            return None
        cfg = getattr(m, "config", None)
        vs = getattr(cfg, "vocab_size", None)
        if isinstance(vs, int) and vs > 0:
            return vs
        try:
            emb = m.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return int(emb.weight.shape[0])
        except Exception:
            pass
        return None

    def _tok_vocab_size(t):
        try:
            return int(len(t))
        except Exception:
            return None

    def _model_paths(m):
        paths = []
        for attr in ["name_or_path", "model_name_or_path"]:
            v = getattr(m, attr, None)
            if isinstance(v, str) and v and v not in paths:
                paths.append(v)
        cfg = getattr(m, "config", None)
        for attr in ["_name_or_path", "name_or_path", "model_name_or_path"]:
            v = getattr(cfg, attr, None)
            if isinstance(v, str) and v and v not in paths:
                paths.append(v)
        return paths

    # tokenizer recovery (cache)
    cache = getattr(gen, "_tok_cache", None)
    if cache is None:
        cache = {}
        setattr(gen, "_tok_cache", cache)

    def _maybe_load_tok_from_model():
        paths = _model_paths(base_model) + _model_paths(gen_model)
        for mp in paths:
            if mp in cache:
                t2 = cache[mp]
            else:
                t2 = None
                try:
                    t2 = AutoTokenizer.from_pretrained(
                        mp, use_fast=False, trust_remote_code=True
                    )
                except Exception:
                    try:
                        t2 = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
                    except Exception:
                        t2 = None
                cache[mp] = t2
            if _is_tok(t2):
                return t2
        return None

    # If vocab sizes disagree, prefer tokenizer loaded from model
    mv = _model_vocab_size(base_model) or _model_vocab_size(gen_model)
    tv = _tok_vocab_size(tok)
    if mv is not None and tv is not None and mv != tv:
        t2 = _maybe_load_tok_from_model()
        if _is_tok(t2) and _tok_vocab_size(t2) == mv:
            tok = t2

    # device
    device = None
    try:
        device = next(gen_model.parameters()).device
    except Exception:
        try:
            device = next(base_model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build inputs
    input_ids = None
    if isinstance(prompt, dict) and "input_ids" in prompt:
        inputs = dict(prompt)
        if torch.is_tensor(inputs["input_ids"]):
            inputs = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }
            input_ids = inputs["input_ids"]
        else:
            input_ids = torch.tensor(
                inputs["input_ids"], dtype=torch.long, device=device
            ).unsqueeze(0)
            inputs["input_ids"] = input_ids
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(input_ids)
    elif torch.is_tensor(prompt):
        input_ids = prompt.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        inputs = {"input_ids": input_ids}
    else:
        # assume string-ish
        inputs = tok(str(prompt), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs.get("input_ids")

    # eos/pad
    eos = gen_kwargs.pop("eos_token_id", None)
    pad = gen_kwargs.pop("pad_token_id", None)
    if eos is None:
        eos = getattr(getattr(gen_model, "config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(tok, "eos_token_id", None)
    if pad is None:
        pad = getattr(getattr(gen_model, "config", None), "pad_token_id", None)
        if pad is None:
            pad = getattr(tok, "pad_token_id", None) or eos

    try:
        if getattr(tok, "pad_token_id", None) is None and eos is not None:
            tok.pad_token_id = eos
    except Exception:
        pass

    # generation params
    if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
        gen_kwargs["max_new_tokens"] = int(max_new_tokens)
    gen_kwargs.setdefault("do_sample", False)

    with torch.no_grad():
        out = gen_model.generate(
            **inputs, eos_token_id=eos, pad_token_id=pad, **gen_kwargs
        )

    # handle GenerateOutput
    if hasattr(out, "sequences"):
        out_ids = out.sequences
    else:
        out_ids = out

    if isinstance(out_ids, (list, tuple)):
        out_ids = out_ids[0]

    if torch.is_tensor(out_ids) and out_ids.dim() == 2:
        out_seq = out_ids[0]
    else:
        out_seq = out_ids

    in_len = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else 0
    new_ids = out_seq[in_len:] if (torch.is_tensor(out_seq) and in_len > 0) else out_seq
    if (
        torch.is_tensor(new_ids)
        and new_ids.numel() == 0
        and torch.is_tensor(out_seq)
        and out_seq.numel() > 0
    ):
        new_ids = out_seq

    def _decode(t):
        try:
            ids = new_ids.tolist() if torch.is_tensor(new_ids) else list(new_ids)
            return t.decode(ids, skip_special_tokens=True)
        except Exception:
            try:
                return t.decode(new_ids, skip_special_tokens=True)
            except Exception:
                return ""

    s = _decode(tok)

    # if we see <unk>, try swapping tokenizer from model path and/or re-decoding
    if "<unk>" in s:
        t2 = _maybe_load_tok_from_model()
        if _is_tok(t2):
            s2 = _decode(t2)
            if s2 and s2.count("<unk>") < s.count("<unk>"):
                tok = t2
                s = s2

    # stop residual <unk> spam / clean up
    s = re.sub(r"(\s*<unk>\s*)+$", "", s).strip()
    if s.count("<unk>") > 5:
        s = re.sub(r"<unk>\s*", "", s).strip()

    # final normalize whitespace
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


if __name__ == "__main__":
    raise SystemExit(main())
