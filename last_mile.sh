#!/usr/bin/env bash
set -euo pipefail

# Run from repo root (even if invoked elsewhere)
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# Defaults
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"   # fast sanity run
FULL_LIMIT="${FULL_LIMIT:-180}"   # your real run

mkdir -p _local_backups logs

# move patch backups out of scripts/ so git status stays clean
shopt -s nullglob
mv -v scripts/*.bak.* scripts/*.pre_patch.* scripts/*.pre_restore.* _local_backups/ 2>/dev/null || true
shopt -u nullglob

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import ast
import re
import time
import textwrap

TAG = "PATCH_GEN_ROBUST_DECODE_ONLY_NEW_TOKENS_V11"
TARGETS = [Path("scripts/eval_v2_holdout.py"), Path("scripts/eval_v2.py")]

REPLACEMENT = textwrap.dedent(r'''
def gen(prompt, max_new_tokens: int = 640, **gen_kwargs):
    """__TAG__: Generate+decode ONLY newly generated text (not the prompt), robust to wrappers and tokenizer mismatch."""
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
        return _callable_attr(o, "decode") and (_callable_attr(o, "__call__") or _callable_attr(o, "encode"))

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
        model = _find_in_stack(["model", "mdl", "m", "lm", "base_model", "guard_model"], _is_model)

    if tok is None or model is None:
        raise RuntimeError("gen(): could not find tokenizer/model in args, globals, or caller stack.")

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
            if hasattr(m, "get_base_model") and callable(getattr(m, "get_base_model", None)):
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
                    t2 = AutoTokenizer.from_pretrained(mp, use_fast=False, trust_remote_code=True)
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
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
        else:
            input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
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
        out = gen_model.generate(**inputs, eos_token_id=eos, pad_token_id=pad, **gen_kwargs)

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
    if torch.is_tensor(new_ids) and new_ids.numel() == 0 and torch.is_tensor(out_seq) and out_seq.numel() > 0:
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
''').lstrip("\n").replace("__TAG__", TAG)

def patch_file(path: Path) -> bool:
    if not path.exists():
        return False

    src = path.read_text(encoding="utf-8")
    if TAG in src:
        print(f"Already patched: {path} ({TAG})")
        return False

    # backup
    bak = path.with_suffix(path.suffix + f".bak.{time.strftime('%Y%m%d_%H%M%S')}")
    bak.write_text(src, encoding="utf-8")
    print("Backup:", bak)

    tree = ast.parse(src)
    lines = src.splitlines(True)

    # prefer top-level def gen, fallback to any
    gens = [n for n in getattr(tree, "body", []) if isinstance(n, ast.FunctionDef) and n.name == "gen"]
    if not gens:
        gens = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "gen"]
    if not gens:
        raise SystemExit(f"ERROR: could not find def gen(...) in {path}")

    # patch bottom-up to keep line indexes valid
    gens = sorted(gens, key=lambda n: (n.lineno, getattr(n, "col_offset", 0)), reverse=True)

    patched = 0
    for fn in gens:
        start = fn.lineno - 1
        end = getattr(fn, "end_lineno", None)
        if end is None:
            raise SystemExit("ERROR: AST nodes missing end_lineno (need Python 3.8+).")
        indent = re.match(r"^\s*", lines[start]).group(0)

        rep_lines = []
        for ln in REPLACEMENT.splitlines(True):
            if ln.strip():
                rep_lines.append(indent + ln)
            else:
                rep_lines.append(ln)

        lines[start:end] = rep_lines
        patched += 1

    new_src = "".join(lines)
    # sanity compile
    compile(new_src, str(path), "exec")
    path.write_text(new_src, encoding="utf-8")
    print(f"Patched: {path} | gen defs: {patched} | TAG: {TAG}")
    return True

any_patched = False
for p in TARGETS:
    try:
        changed = patch_file(p)
        any_patched = any_patched or changed
    except Exception as e:
        raise SystemExit(f"PATCH FAILED for {p}: {e}")

print("py_compile OK")
PY

echo
echo "[smoke] running eval_v2_holdout for 3 modes with SMOKE_LIMIT=${SMOKE_LIMIT}"
python -m scripts.eval_v2_holdout --mode base_like    --limit "${SMOKE_LIMIT}"
python -m scripts.eval_v2_holdout --mode cmwe         --limit "${SMOKE_LIMIT}"
python -m scripts.eval_v2_holdout --mode always_guard --limit "${SMOKE_LIMIT}"

echo
echo "[smoke] quick log checks (empty/<unk>/len stats + a few <unk> examples)"
python - <<'PY'
import pandas as pd
from pathlib import Path

paths = [
    ("base_like",    "logs/eval_base_like_v2_holdout.csv"),
    ("cmwe",         "logs/eval_gated_mixed_v2_holdout.csv"),
    ("always_guard", "logs/eval_guard_always_v2_holdout.csv"),
]

def safe_mean(x):
    x = pd.Series(x)
    return float(x.mean()) if len(x) else float("nan")

for name, p in paths:
    p = Path(p)
    if not p.exists():
        print(f"\n== {name} ==\nMISSING: {p}")
        continue
    df = pd.read_csv(p, keep_default_na=False)
    out = df["out"].astype(str)
    lens = out.str.len()

    empty_rate = safe_mean(lens == 0)
    unk_any_rate = safe_mean(out.str.contains("<unk>", regex=False))
    uniq_rate = float(out.nunique() / max(1, len(out)))
    mx = int(lens.max()) if len(lens) else 0
    hit_max_rate = safe_mean(lens == mx) if mx > 0 else 0.0

    print(f"\n== {name} ({p}) ==")
    print("rows:", len(df))
    if "route" in df.columns:
        print("route_counts:", df["route"].value_counts(dropna=False).to_dict())
    print("empty_rate:", round(empty_rate, 4))
    print("unk_any_rate:", round(unk_any_rate, 4))
    print("unique_out_rate:", round(uniq_rate, 4))
    print("len min/median/max:", int(lens.min()), float(lens.median()), int(lens.max()))
    print("hit_max_len_rate:", round(hit_max_rate, 4), "(max_len=%s)" % mx)

    bad = df[out.str.contains("<unk>", regex=False)][["i","bucket","route","q","out"]].head(3) if "bucket" in df.columns else df[out.str.contains("<unk>", regex=False)][["i","route","q","out"]].head(3)
    if len(bad):
        print("first <unk> examples (out_head):")
        for _, r in bad.iterrows():
            oh = str(r["out"])[:140].replace("\n","\\n")
            qh = str(r["q"])[:140].replace("\n","\\n")
            print("  i=", r["i"], "route=", r.get("route", ""), "q_head=", repr(qh), "out_head=", repr(oh))

PY

echo
echo "[next] if smoke looks good, run full:"
echo "  python -m scripts.eval_v2_holdout --mode base_like    --limit ${FULL_LIMIT}"
echo "  python -m scripts.eval_v2_holdout --mode cmwe         --limit ${FULL_LIMIT}"
echo "  python -m scripts.eval_v2_holdout --mode always_guard --limit ${FULL_LIMIT}"
echo "  python -m scripts.validate_v2_holdout_logs"
