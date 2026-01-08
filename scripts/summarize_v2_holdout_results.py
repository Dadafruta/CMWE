from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def _die(msg: str, code: int = 2) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _safe_mean(x) -> float:
    # works for pandas Series or list-like
    try:
        n = int(getattr(x, "shape", [len(x)])[0])
    except Exception:
        n = len(x)
    if n == 0:
        return float("nan")
    try:
        return float(x.mean())
    except Exception:
        s = 0.0
        for v in x:
            s += float(v)
        return s / n


def _to_bool_series(s):
    # Best-effort coercion to boolean series (pandas)
    if s is None:
        return None
    if pd is None:
        return None

    if str(s.dtype) == "bool":
        return s

    # numeric -> nonzero
    if str(s.dtype).startswith(("int", "float")):
        return s.fillna(0).astype(float) != 0.0

    # string-ish
    ss = s.astype(str).fillna("").str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n", ""}
    out = ss.isin(true_set)
    # Anything not in true_set/false_set: treat as False unless it's clearly truthy
    # (keeps behavior conservative)
    out = out & (~ss.isin(false_set))
    return out


def _read_csv(p: Path):
    if pd is None:
        _die(
            "pandas is required for summarize_v2_holdout_results.py (pip install pandas)"
        )
    return pd.read_csv(p, keep_default_na=False)


def _summarize_one(name: str, p: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    if not p.exists():
        return {"mode": name, "path": str(p), "missing": True}, f"MISSING: {name}: {p}"

    df = _read_csv(p)
    out = df["out"].astype(str) if "out" in df.columns else (df.iloc[:, 0].astype(str))
    out = out.fillna("")
    lens = out.str.len()

    max_len = int(lens.max()) if len(lens) else 0

    route_counts = {}
    if "route" in df.columns:
        vc = df["route"].astype(str).value_counts()
        route_counts = {str(k): int(v) for k, v in vc.to_dict().items()}

    # task columns if present
    unans = (
        _to_bool_series(df["unanswerable"]) if "unanswerable" in df.columns else None
    )
    refused = _to_bool_series(df["refused"]) if "refused" in df.columns else None
    correct = _to_bool_series(df["correct"]) if "correct" in df.columns else None

    # metrics
    empty_rate = _safe_mean(lens == 0)
    unk_any_rate = _safe_mean(out.str.contains("<unk>", regex=False))
    unique_out_rate = float(out.nunique() / max(1, len(out)))

    len_min = int(lens.min()) if len(lens) else 0
    len_median = float(lens.median()) if len(lens) else float("nan")
    len_max = int(lens.max()) if len(lens) else 0
    hit_max_len_rate = _safe_mean(lens == len_max) if len_max else float("nan")

    # task-ish metrics (only if columns exist)
    unanswerable_rate = float("nan")
    refusal_on_unanswerables = float("nan")
    false_refusal_on_answerables = float("nan")
    acc_answerables = float("nan")
    coverage_answerables = float("nan")
    acc_when_answered = float("nan")

    if unans is not None:
        unanswerable_rate = _safe_mean(unans)

    if (unans is not None) and (refused is not None):
        if bool(unans.any()):
            refusal_on_unanswerables = _safe_mean(refused[unans])
        if bool((~unans).any()):
            false_refusal_on_answerables = _safe_mean(refused[~unans])
            coverage_answerables = _safe_mean((~refused)[~unans])

    if (unans is not None) and (correct is not None):
        if bool((~unans).any()):
            acc_answerables = _safe_mean(correct[~unans])

    if (unans is not None) and (refused is not None) and (correct is not None):
        mask = (~unans) & (~refused)
        denom = float(mask.mean())
        if denom > 0:
            acc_when_answered = float(correct[mask].mean())

    row0 = ""
    try:
        row0 = repr(out.iloc[0][:160])
    except Exception:
        pass

    return (
        {
            "mode": name,
            "path": str(p),
            "rows": int(len(df)),
            "route_counts": route_counts,
            "empty_rate": float(empty_rate),
            "unk_any_rate": float(unk_any_rate),
            "unique_out_rate": float(unique_out_rate),
            "len_min": int(len_min),
            "len_median": float(len_median),
            "len_max": int(len_max),
            "hit_max_len_rate": float(hit_max_len_rate),
            "max_len": int(max_len),
            "unanswerable_rate": float(unanswerable_rate),
            "refusal_on_unanswerables": float(refusal_on_unanswerables),
            "false_refusal_on_answerables": float(false_refusal_on_answerables),
            "acc_answerables": float(acc_answerables),
            "coverage_answerables": float(coverage_answerables),
            "acc_when_answered": float(acc_when_answered),
            "out_head": row0,
        },
        None,
    )


def _align_for_compare(a, b):
    # align by 'i' if present in both, else by row order
    if pd is None:
        return None
    if ("i" in a.columns) and ("i" in b.columns):
        aa = a.copy()
        bb = b.copy()
        aa["i"] = aa["i"].astype(int)
        bb["i"] = bb["i"].astype(int)
        m = aa.merge(bb, on="i", suffixes=("_a", "_b"))
        return m
    # fallback: zip by index
    n = min(len(a), len(b))
    aa = a.iloc[:n].reset_index(drop=True)
    bb = b.iloc[:n].reset_index(drop=True)
    m = pd.concat([aa.add_suffix("_a"), bb.add_suffix("_b")], axis=1)
    return m


def _compare(
    name_a: str, p_a: Path, name_b: str, p_b: Path
) -> Optional[Dict[str, Any]]:
    if pd is None:
        return None
    if (not p_a.exists()) or (not p_b.exists()):
        return None
    a = _read_csv(p_a)
    b = _read_csv(p_b)
    m = _align_for_compare(a, b)
    if m is None or len(m) == 0:
        return None

    def pct_same(col: str) -> float:
        ca = f"{col}_a"
        cb = f"{col}_b"
        if (ca not in m.columns) or (cb not in m.columns):
            return float("nan")
        sa = m[ca].astype(str).fillna("")
        sb = m[cb].astype(str).fillna("")
        return float((sa == sb).mean())

    return {
        "a": name_a,
        "b": name_b,
        "rows_compared": int(len(m)),
        "pct_same_out": pct_same("out"),
        "pct_same_refused": pct_same("refused"),
        "pct_same_correct": pct_same("correct"),
        "pct_same_route": pct_same("route"),
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if math.isnan(xf):
            return "NA"
        return f"{xf:.{nd}f}"
    except Exception:
        return str(x)


def _to_markdown(rows: list[Dict[str, Any]], comps: list[Dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# v2_holdout summary\n")

    # main table
    headers = [
        "mode",
        "rows",
        "routes",
        "empty_rate",
        "unk_any_rate",
        "unique_out_rate",
        "len_min",
        "len_median",
        "len_max",
        "hit_max_len_rate",
        "unanswerable_rate",
        "refusal_on_unanswerables",
        "false_refusal_on_answerables",
        "acc_answerables",
        "coverage_answerables",
        "acc_when_answered",
    ]
    lines.append("## per-mode metrics\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in rows:
        routes = json.dumps(r.get("route_counts", {}), sort_keys=True)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("mode", "")),
                    str(r.get("rows", "")),
                    routes,
                    _fmt(r.get("empty_rate")),
                    _fmt(r.get("unk_any_rate")),
                    _fmt(r.get("unique_out_rate")),
                    str(r.get("len_min", "")),
                    _fmt(r.get("len_median"), nd=1),
                    str(r.get("len_max", "")),
                    _fmt(r.get("hit_max_len_rate")),
                    _fmt(r.get("unanswerable_rate")),
                    _fmt(r.get("refusal_on_unanswerables")),
                    _fmt(r.get("false_refusal_on_answerables")),
                    _fmt(r.get("acc_answerables")),
                    _fmt(r.get("coverage_answerables")),
                    _fmt(r.get("acc_when_answered")),
                ]
            )
            + " |"
        )

    if comps:
        lines.append("\n## cross-mode comparisons\n")
        headers2 = [
            "a",
            "b",
            "rows_compared",
            "pct_same_out",
            "pct_same_refused",
            "pct_same_correct",
            "pct_same_route",
        ]
        lines.append("| " + " | ".join(headers2) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers2)) + " |")
        for c in comps:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(c.get("a", "")),
                        str(c.get("b", "")),
                        str(c.get("rows_compared", "")),
                        _fmt(c.get("pct_same_out")),
                        _fmt(c.get("pct_same_refused")),
                        _fmt(c.get("pct_same_correct")),
                        _fmt(c.get("pct_same_route")),
                    ]
                )
                + " |"
            )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize v2_holdout CSV logs into a paper-ready table."
    )
    ap.add_argument("--base_like", default="logs/eval_base_like_v2_holdout.csv")
    ap.add_argument("--cmwe", default="logs/eval_gated_mixed_v2_holdout.csv")
    ap.add_argument("--always_guard", default="logs/eval_guard_always_v2_holdout.csv")
    ap.add_argument("--out", default="", help="Optional path to write markdown output.")
    args = ap.parse_args()

    if pd is None:
        _die("pandas is not available in this environment")

    paths = [
        ("base_like", Path(args.base_like)),
        ("cmwe", Path(args.cmwe)),
        ("always_guard", Path(args.always_guard)),
    ]

    rows: list[Dict[str, Any]] = []
    missing_msgs: list[str] = []
    for name, p in paths:
        r, miss = _summarize_one(name, p)
        rows.append(r)
        if miss:
            missing_msgs.append(miss)

    comps: list[Dict[str, Any]] = []
    comp1 = _compare("base_like", Path(args.base_like), "cmwe", Path(args.cmwe))
    if comp1:
        comps.append(comp1)
    comp2 = _compare(
        "base_like", Path(args.base_like), "always_guard", Path(args.always_guard)
    )
    if comp2:
        comps.append(comp2)

    md = _to_markdown(rows, comps)

    if missing_msgs:
        md = "\n".join(["<!-- " + m + " -->" for m in missing_msgs]) + "\n\n" + md

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(md, encoding="utf-8")
        print(f"Wrote: {outp}")
    else:
        print(md)


if __name__ == "__main__":
    main()
