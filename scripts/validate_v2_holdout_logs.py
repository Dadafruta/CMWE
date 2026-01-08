import argparse
from pathlib import Path
import pandas as pd
import math

DEFAULT_BASE = "logs/eval_base_like_v2_holdout.csv"
DEFAULT_CMWE = "logs/eval_gated_mixed_v2_holdout.csv"
DEFAULT_ALWAYS = "logs/eval_guard_always_v2_holdout.csv"


def safe_mean(x):
    try:
        if x is None:
            return float("nan")
        if hasattr(x, "__len__") and len(x) == 0:
            return float("nan")
        return float(pd.Series(x).mean())
    except Exception:
        return float("nan")


def to_bool_series(s):
    if s is None:
        return None
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    ss = s.astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y"}
    return ss.isin(truthy)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def summarize(name: str, path: Path):
    if not path.exists():
        return None, [f"{name}: missing ({path})"]

    df = read_csv(path)

    out = df["out"].astype(str) if "out" in df.columns else pd.Series([""] * len(df))
    lens = out.str.len()

    max_len = 0
    try:
        ml = float(lens.max()) if len(lens) else 0.0
        max_len = int(ml) if math.isfinite(ml) else 0
    except Exception:
        max_len = 0

    empty_rate = safe_mean(lens == 0)
    unk_any_rate = safe_mean(out.str.contains("<unk>", regex=False))
    unique_out_rate = float(out.nunique() / max(1, len(out)))

    hit_max_len_rate = float("nan")
    if max_len > 0:
        hit_max_len_rate = safe_mean(lens == max_len)

    route_counts = None
    nonbase_routed = None
    if "route" in df.columns:
        route_counts = df["route"].astype(str).value_counts(dropna=False).to_dict()
        nonbase_routed = int((df["route"].astype(str) != "base").sum())

    # task metrics (if present)
    unans = to_bool_series(df["unanswerable"]) if "unanswerable" in df.columns else None
    ref = to_bool_series(df["refused"]) if "refused" in df.columns else None
    corr = to_bool_series(df["correct"]) if "correct" in df.columns else None

    unanswerable_rate = safe_mean(unans) if unans is not None else float("nan")
    refusal_on_unanswerables = float("nan")
    false_refusal_on_answerables = float("nan")
    acc_answerables = float("nan")
    coverage_answerables = float("nan")
    acc_when_answered = float("nan")

    if unans is not None and ref is not None:
        if bool(unans.any()):
            refusal_on_unanswerables = safe_mean(ref[unans])
        if bool((~unans).any()):
            false_refusal_on_answerables = safe_mean(ref[~unans])

    if unans is not None and corr is not None:
        if bool((~unans).any()):
            acc_answerables = safe_mean(corr[~unans])

    if unans is not None:
        ans_mask = ~unans
        answered_mask = lens > 0
        if ref is not None:
            answered_mask = answered_mask & (~ref)
        if bool(ans_mask.any()):
            coverage_answerables = safe_mean(answered_mask[ans_mask])
            if corr is not None:
                if bool((ans_mask & answered_mask).any()):
                    acc_when_answered = safe_mean(corr[ans_mask & answered_mask])

    print(f"\n== {name} ({path}) ==")
    print("rows:", len(df))
    if route_counts is not None:
        print("route_counts:", route_counts)
    print(
        "empty_rate:", round(empty_rate, 4) if math.isfinite(empty_rate) else empty_rate
    )
    print(
        "unk_any_rate:",
        round(unk_any_rate, 4) if math.isfinite(unk_any_rate) else unk_any_rate,
    )
    print("unique_out_rate:", round(unique_out_rate, 4))
    print(
        "len min/median/max:",
        int(lens.min()) if len(lens) else 0,
        float(lens.median()) if len(lens) else 0.0,
        int(lens.max()) if len(lens) else 0,
    )
    if math.isfinite(hit_max_len_rate):
        print(f"hit_max_len_rate: {round(hit_max_len_rate, 4)} (max_len={max_len})")

    if math.isfinite(unanswerable_rate):
        print("unanswerable_rate:", round(unanswerable_rate, 4))
    if math.isfinite(refusal_on_unanswerables):
        print("refusal_on_unanswerables:", round(refusal_on_unanswerables, 4))
    if math.isfinite(false_refusal_on_answerables):
        print("false_refusal_on_answerables:", round(false_refusal_on_answerables, 4))
    if math.isfinite(acc_answerables):
        print("acc_answerables:", round(acc_answerables, 4))
    if math.isfinite(coverage_answerables):
        print("coverage_answerables:", round(coverage_answerables, 4))
    if math.isfinite(acc_when_answered):
        print("acc_when_answered:", round(acc_when_answered, 4))

    # show a few empty / unk examples if present
    show_cols = [
        c for c in ["i", "bucket", "route", "q", "gold", "out"] if c in df.columns
    ]
    if empty_rate and empty_rate > 0:
        ex = df.loc[lens == 0, show_cols].head(5)
        if len(ex):
            print("\nfirst empty examples:")
            print(ex.to_string(index=False)[:2000])
    if unk_any_rate and unk_any_rate > 0:
        ex = df.loc[out.str.contains("<unk>", regex=False), show_cols].head(5)
        if len(ex):
            print("\nfirst <unk> examples:")
            print(ex.to_string(index=False)[:2000])

    stats = {
        "name": name,
        "path": str(path),
        "rows": len(df),
        "empty_rate": empty_rate,
        "unk_any_rate": unk_any_rate,
        "unique_out_rate": unique_out_rate,
        "max_len": max_len,
        "hit_max_len_rate": hit_max_len_rate,
        "nonbase_routed": nonbase_routed,
    }
    return (df, stats), []


def compare(base_df: pd.DataFrame, other_df: pd.DataFrame, other_name: str):
    out = []
    if base_df is None or other_df is None:
        return out

    if "i" in base_df.columns and "i" in other_df.columns:
        b = base_df[
            ["i"] + [c for c in ["out", "refused", "correct"] if c in base_df.columns]
        ].copy()
        o = other_df[
            ["i"]
            + [
                c
                for c in ["out", "refused", "correct", "route"]
                if c in other_df.columns
            ]
        ].copy()
        m = b.merge(o, on="i", suffixes=("_base", f"_{other_name}"))
    else:
        n = min(len(base_df), len(other_df))
        m = pd.DataFrame(
            {
                "out_base": base_df["out"].astype(str).iloc[:n].to_list()
                if "out" in base_df.columns
                else [""] * n,
                f"out_{other_name}": other_df["out"].astype(str).iloc[:n].to_list()
                if "out" in other_df.columns
                else [""] * n,
            }
        )

    rows = len(m)
    if rows == 0:
        return out

    out_base = (
        m["out_base"].astype(str) if "out_base" in m.columns else m["out"].astype(str)
    )
    out_other = (
        m[f"out_{other_name}"].astype(str)
        if f"out_{other_name}" in m.columns
        else m["out"].astype(str)
    )

    same_out = safe_mean(out_base == out_other)
    print(f"\n== compare base_like vs {other_name} ==")
    print("rows_compared:", rows)
    print("pct_same_out:", round(same_out, 4) if math.isfinite(same_out) else same_out)

    # same_refused / same_correct if present
    if "refused_base" in m.columns and f"refused_{other_name}" in m.columns:
        rb = to_bool_series(m["refused_base"])
        ro = to_bool_series(m[f"refused_{other_name}"])
        sr = safe_mean(rb == ro)
        print("pct_same_refused:", round(sr, 4) if math.isfinite(sr) else sr)

    if "correct_base" in m.columns and f"correct_{other_name}" in m.columns:
        cb = to_bool_series(m["correct_base"])
        co = to_bool_series(m[f"correct_{other_name}"])
        sc = safe_mean(cb == co)
        print("pct_same_correct:", round(sc, 4) if math.isfinite(sc) else sc)

    # nonbase route change rate if route exists
    if f"route_{other_name}" in m.columns:
        route = m[f"route_{other_name}"].astype(str)
        nb = route != "base"
        nonbase_routed = int(nb.sum())
        print("nonbase_routed:", nonbase_routed)
        if nonbase_routed:
            changed = safe_mean((out_base != out_other)[nb])
            print(
                "pct_changed_on_nonbase:",
                round(changed, 4) if math.isfinite(changed) else changed,
            )

    return out


def validate(stats_list):
    fails = []

    # thresholds tuned to catch degenerate runs (empty / <unk> spam / maxlen saturation / no diversity)
    for st in stats_list:
        if st is None:
            continue
        name = st["name"]
        empty_rate = st["empty_rate"]
        unk_any_rate = st["unk_any_rate"]
        unique_out_rate = st["unique_out_rate"]
        max_len = st["max_len"]
        hit_max_len_rate = st["hit_max_len_rate"]

        if math.isfinite(empty_rate) and empty_rate > 0.30:
            fails.append(
                f"{name}: too many empty outputs (empty_rate={empty_rate:.3f})"
            )
        if math.isfinite(unk_any_rate) and unk_any_rate > 0.20:
            fails.append(
                f"{name}: <unk> present too often (unk_any_rate={unk_any_rate:.3f})"
            )
        if math.isfinite(unique_out_rate) and unique_out_rate < (
            0.10 if name == "base_like" else 0.05
        ):
            fails.append(
                f"{name}: too few unique outputs (unique_out_rate={unique_out_rate:.3f})"
            )
        if (
            max_len >= 128
            and math.isfinite(hit_max_len_rate)
            and hit_max_len_rate > 0.80
        ):
            fails.append(
                f"{name}: suspicious max-length saturation (hit_max_len_rate={hit_max_len_rate:.3f}, max_len={max_len})"
            )

        # ensure gating actually routes something non-base (for cmwe/always_guard)
        if name in {"cmwe", "always_guard"} and st.get("nonbase_routed") is not None:
            if int(st["nonbase_routed"]) == 0:
                fails.append(
                    f"{name}: no non-base routing observed (route != 'base' never happened)"
                )

    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--cmwe", default=DEFAULT_CMWE)
    ap.add_argument("--always", default=DEFAULT_ALWAYS)
    args = ap.parse_args()

    base_p = Path(args.base)
    cmwe_p = Path(args.cmwe)
    always_p = Path(args.always)

    (base_pair, base_errs) = (None, [])
    (cmwe_pair, cmwe_errs) = (None, [])
    (always_pair, always_errs) = (None, [])

    base_pair, base_errs = summarize("base_like", base_p)
    cmwe_pair, cmwe_errs = summarize("cmwe", cmwe_p)
    always_pair, always_errs = summarize("always_guard", always_p)

    missing_errs = base_errs + cmwe_errs + always_errs
    for e in missing_errs:
        print("MISSING:", e)

    base_df, base_stats = base_pair if base_pair else (None, None)
    cmwe_df, cmwe_stats = cmwe_pair if cmwe_pair else (None, None)
    always_df, always_stats = always_pair if always_pair else (None, None)

    if base_df is not None and cmwe_df is not None:
        compare(base_df, cmwe_df, "cmwe")
    if base_df is not None and always_df is not None:
        compare(base_df, always_df, "always_guard")

    stats_list = [s for s in [base_stats, cmwe_stats, always_stats] if s is not None]
    fails = validate(stats_list)

    if fails:
        print("\nVALIDATION: FAIL")
        for f in fails:
            print(" -", f)
        raise SystemExit(2)

    print("\nVALIDATION: PASS")


if __name__ == "__main__":
    main()
