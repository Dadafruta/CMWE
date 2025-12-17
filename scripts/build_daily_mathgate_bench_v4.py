#!/usr/bin/env python3
"""Build daily mathgate bench v4.

Run:
  python -m scripts.build_daily_mathgate_bench_v4 --help
"""

import argparse, json, hashlib, random
from pathlib import Path
import re

# Keep these consistent with scripts/gated_infer.py :contentReference[oaicite:1]{index=1}
CITE_PAT = re.compile(
    r"\b(doi|pubmed|pmid|citation|reference|crossref|url|warp drive|blueprints?)\b",
    re.I,
)
MATH_PAT = re.compile(
    r"(?:\bln\s*\(\s*0\s*\)\b|\blog\s*\(\s*0\s*\)\b|\b1\s*/\s*0\b|divide by zero|\bNaN\b|\bInf\b|\bundefined\b|\blimit\b|\bderivative\b|\bintegral\b|\bproof\b|\bevaluate\b)",
    re.I,
)


def pick_domain(q: str):
    if CITE_PAT.search(q):
        return "citation_guard"
    if MATH_PAT.search(q):
        return "math_guard"
    return None


def stable_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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


def sample_unique(rows, n: int, seed: int):
    # rows: list[dict] must have "q"
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    out, seen = [], set()
    for r in rows:
        q = r["q"]
        if q in seen:
            continue
        seen.add(q)
        out.append(r)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_answer", type=int, default=128)
    ap.add_argument("--n_refuse", type=int, default=128)
    ap.add_argument("--n_nonmath", type=int, default=128)
    ap.add_argument("--n_unsupported", type=int, default=128)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override seed (else derived from out_dir)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else stable_seed(str(out_dir))

    # Prefer "mixed_*" datasets (they include unanswerable labels). Example format: {"q","a","unanswerable"}
    candidate_sources = [
        Path("data/mixed_eval_v2_holdout.jsonl"),
        Path("data/mixed_eval_v1_holdout.jsonl"),
        Path("data/mixed_nonsense_eval_v1.jsonl"),
        Path("data/cite_hardset_v2.jsonl"),
    ]

    labeled = []
    for src in candidate_sources:
        for obj in iter_jsonl(src):
            q = extract_q(obj)
            if not q:
                continue
            labeled.append(
                {
                    "q": q,
                    "unanswerable": bool(obj.get("unanswerable", False)),
                    "bucket": obj.get("bucket"),
                    "source": str(src),
                    "domain": pick_domain(q),
                }
            )

    # Build answer/refuse as "domain-positive" first; fall back to any if you don’t have enough.
    ans_dom = [
        r for r in labeled if (not r["unanswerable"]) and (r["domain"] is not None)
    ]
    ref_dom = [r for r in labeled if (r["unanswerable"]) and (r["domain"] is not None)]
    ans_any = [r for r in labeled if not r["unanswerable"]]
    ref_any = [r for r in labeled if r["unanswerable"]]

    answer_rows = sample_unique(
        ans_dom if len(ans_dom) >= args.n_answer else ans_any, args.n_answer, seed + 1
    )
    refuse_rows = sample_unique(
        ref_dom if len(ref_dom) >= args.n_refuse else ref_any, args.n_refuse, seed + 2
    )

    # For nonmath/unsupported, prefer the v1 pools you created (pipeline relies on these too) :contentReference[oaicite:3]{index=3}
    nonmath_pool = Path("data/daily_v1_nonmath.jsonl")
    unsupported_pool = Path("data/daily_v1_unsupported.jsonl")

    nonmath_rows = []
    if nonmath_pool.exists():
        for obj in iter_jsonl(nonmath_pool):
            q = extract_q(obj)
            if q:
                nonmath_rows.append(
                    {"q": q, "source": str(nonmath_pool), "domain": pick_domain(q)}
                )
    else:
        # fallback: domain=None from answerable pool
        nonmath_rows = [
            {"q": r["q"], "source": r["source"], "domain": r["domain"]}
            for r in ans_any
            if r["domain"] is None
        ]

    unsupported_rows = []
    if unsupported_pool.exists():
        for obj in iter_jsonl(unsupported_pool):
            q = extract_q(obj)
            if q:
                unsupported_rows.append(
                    {"q": q, "source": str(unsupported_pool), "domain": pick_domain(q)}
                )
    else:
        # fallback: domain==math_guard from answerable pool
        unsupported_rows = [
            {"q": r["q"], "source": r["source"], "domain": r["domain"]}
            for r in ans_any
            if r["domain"] == "math_guard"
        ]

    nonmath_rows = sample_unique(nonmath_rows, args.n_nonmath, seed + 3)
    unsupported_rows = sample_unique(unsupported_rows, args.n_unsupported, seed + 4)

    def write_jsonl(path: Path, rows, gold: str):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                obj = {"q": r["q"], "gold": gold}
                if r.get("domain") is not None:
                    obj["domain"] = r["domain"]
                if r.get("bucket") is not None:
                    obj["bucket"] = r["bucket"]
                if r.get("source") is not None:
                    obj["source"] = r["source"]
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "answer.jsonl", answer_rows, gold="answer")
    write_jsonl(out_dir / "refuse.jsonl", refuse_rows, gold="refuse")
    write_jsonl(out_dir / "nonmath.jsonl", nonmath_rows, gold="pass")
    write_jsonl(out_dir / "unsupported.jsonl", unsupported_rows, gold="pass")

    print(f"Wrote: {out_dir / 'answer.jsonl'} ({len(answer_rows)})")
    print(f"Wrote: {out_dir / 'refuse.jsonl'} ({len(refuse_rows)})")
    print(f"Wrote: {out_dir / 'nonmath.jsonl'} ({len(nonmath_rows)})")
    print(f"Wrote: {out_dir / 'unsupported.jsonl'} ({len(unsupported_rows)})")


if __name__ == "__main__":
    main()
