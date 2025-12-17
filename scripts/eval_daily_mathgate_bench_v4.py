#!/usr/bin/env python3
import argparse, csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

import joblib

# Mirror scripts/gated_infer.py for domain routing :contentReference[oaicite:5]{index=5}
DET_PATH = "artifacts/risk_detector.joblib"
CITE_PAT = re.compile(r"\b(doi|pubmed|pmid|citation|reference|crossref|url|warp drive|blueprints?)\b", re.I)
MATH_PAT = re.compile(r"(?:\bln\s*\(\s*0\s*\)\b|\blog\s*\(\s*0\s*\)\b|\b1\s*/\s*0\b|divide by zero|\bNaN\b|\bInf\b|\bundefined\b|\blimit\b|\bderivative\b|\bintegral\b|\bproof\b|\bevaluate\b)", re.I)

def pick_domain(q: str):
    if CITE_PAT.search(q): return "citation_guard"
    if MATH_PAT.search(q): return "math_guard"
    return None

def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def extract_q(obj: dict) -> str | None:
    return obj.get("q") or obj.get("question") or obj.get("prompt") or obj.get("input")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--th_math", type=float, default=0.10)
    ap.add_argument("--th_cite", type=float, default=0.80)
    ap.add_argument("--cap_cite", type=float, default=0.6)  # kept for compatibility with caller
    ap.add_argument("--det_path", type=str, default=DET_PATH)
    args = ap.parse_args()

    det = joblib.load(args.det_path)

    data_dir = Path(args.data_dir)
    files = [
        ("answer", data_dir / "answer.jsonl"),
        ("refuse", data_dir / "refuse.jsonl"),
        ("nonmath", data_dir / "nonmath.jsonl"),
        ("unsupported", data_dir / "unsupported.jsonl"),
    ]

    rows = []
    for split, fp in files:
        for obj in iter_jsonl(fp):
            q = extract_q(obj)
            if not q:
                continue
            gold = obj.get("gold") or split
            rows.append({
                "split": split,
                "q": q,
                "gold": gold,
                "bucket": obj.get("bucket"),
                "source": obj.get("source"),
            })

    if not rows:
        raise SystemExit(f"No rows found under {data_dir} (expected answer/refuse/nonmath/unsupported jsonl files).")

    # Evaluate routing/triggering (fast; does not run the LLM).
    out_rows = []
    route_counts = Counter()
    # Confusion for "should trigger guard?" => gold_refuse vs pred_trigger
    tp = fp = tn = fn = 0

    # For top_reasons (bucket/source) on errors
    fp_reasons = Counter()
    fn_reasons = Counter()

    for r in rows:
        q = r["q"]
        gold = r["gold"]

        dom = pick_domain(q)
        risk = float(det.predict_proba([q])[0, 1])

        pred_trigger = False
        route = "base"
        if dom == "math_guard" and risk >= args.th_math:
            pred_trigger = True
            route = "math_guard"
        elif dom == "citation_guard" and risk >= args.th_cite:
            pred_trigger = True
            route = "citation_guard"

        route_counts[route] += 1

        gold_refuse = (gold == "refuse")
        if pred_trigger and gold_refuse:
            tp += 1
        elif pred_trigger and (not gold_refuse):
            fp += 1
            key = r["bucket"] or r["source"] or r["split"]
            fp_reasons[key] += 1
        elif (not pred_trigger) and gold_refuse:
            fn += 1
            key = r["bucket"] or r["source"] or r["split"]
            fn_reasons[key] += 1
        else:
            tn += 1

        out_rows.append({
            "split": r["split"],
            "gold": gold,
            "domain": dom,
            "risk": round(risk, 6),
            "route": route,
            "triggered": int(pred_trigger),
            "bucket": r["bucket"] or "",
            "source": r["source"] or "",
            "q": q,
        })

    n = len(out_rows)
    accuracy = (tp + tn) / n if n else 0.0
    refused_rate = (tp + fp) / n if n else 0.0              # proxy = trigger rate
    false_tool_rate = fp / n if n else 0.0
    false_pass_rate = fn / n if n else 0.0
    nonrefuse = max(1, (tn + fp))
    false_trigger_rate = fp / nonrefuse                      # FPR on non-refuse gold

    # "accuracy_given_tool" = precision when the tool triggers
    denom = (tp + fp)
    accuracy_given_tool = (tp / denom) if denom else 0.0

    # tool_coverage: among gold_refuse items, how often did we trigger (overall + by route)
    gold_refuse_total = tp + fn
    cov_overall = (tp / gold_refuse_total) if gold_refuse_total else 0.0
    cov_by_tool = {
        "math_guard": None,
        "citation_guard": None,
    }
    # Compute by tool-domain subset
    for tool in ("math_guard", "citation_guard"):
        sub_total = 0
        sub_hit = 0
        for rr in out_rows:
            if rr["gold"] == "refuse" and rr["domain"] == tool:
                sub_total += 1
                if rr["route"] == tool:
                    sub_hit += 1
        cov_by_tool[tool] = (sub_hit / sub_total) if sub_total else None

    tool_coverage = {
        "overall_recall_on_refuse": cov_overall,
        "recall_by_tool_domain": cov_by_tool,
    }

    # top_reasons
    top = {
        "false_tool": fp_reasons.most_common(10),
        "false_pass": fn_reasons.most_common(10),
    }

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Print lines that scripts/run_daily_mathgate_v4_pipeline.sh greps :contentReference[oaicite:6]{index=6}
    print("=== SUMMARY")
    print(f"n={n}")
    print(f"accuracy={accuracy:.6f}")
    print(f"accuracy_given_tool={accuracy_given_tool:.6f}")
    print(f"refused_rate={refused_rate:.6f}")
    print(f"false_tool_rate={false_tool_rate:.6f}")
    print(f"false_pass_rate={false_pass_rate:.6f}")
    print(f"false_trigger_rate={false_trigger_rate:.6f}")
    print("route_counts=" + json.dumps(dict(route_counts), ensure_ascii=False))
    print("tool_coverage=" + json.dumps(tool_coverage, ensure_ascii=False))
    print("=== CONFUSION")
    print("confusion=" + json.dumps({"tp": tp, "fp": fp, "tn": tn, "fn": fn}, ensure_ascii=False))
    print("top_reasons=" + json.dumps(top, ensure_ascii=False))

if __name__ == "__main__":
    main()
