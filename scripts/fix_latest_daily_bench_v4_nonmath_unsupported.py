#!/usr/bin/env python3
"""Fix latest daily bench v4 nonmath unsupported.

Run:
  python -m scripts.fix_latest_daily_bench_v4_nonmath_unsupported --help
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def stable_seed(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def newest_run_dir(bench_dir: Path) -> Path:
    if not bench_dir.exists():
        raise SystemExit(f"Missing bench dir: {bench_dir}")
    runs = [p for p in bench_dir.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"No run dirs found under: {bench_dir}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def load_unique_qs_jsonl(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing source dataset: {path}")

    seen: set[str] = set()
    out: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise SystemExit(f"{path}:{i} invalid json: {e}")

            q = (
                obj.get("q")
                or obj.get("question")
                or obj.get("prompt")
                or obj.get("input")
            )
            if not q:
                raise SystemExit(f"{path}:{i} missing q/question/prompt/input")

            if q not in seen:
                seen.add(q)
                out.append(q)

    return out


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{stamp}")
    path.replace(bak)
    return bak


def write_bench_jsonl(path: Path, qs: list[str], gold: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps({"q": q, "gold": gold}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_dir", default="data/daily_bench_v4")
    ap.add_argument(
        "--run_dir",
        default="",
        help="If empty, patches newest run dir under --bench_dir",
    )
    ap.add_argument("--nonmath_src", default="data/daily_v1_nonmath.jsonl")
    ap.add_argument("--unsupported_src", default="data/daily_v1_unsupported.jsonl")
    ap.add_argument("--n_nonmath", type=int, default=50_000)
    ap.add_argument("--n_unsupported", type=int, default=20_000)
    ap.add_argument(
        "--seed", type=int, default=-1, help="If <0, uses stable seed from run dir name"
    )
    args = ap.parse_args()

    root = Path.cwd()
    bench_dir = (root / args.bench_dir).resolve()
    run_dir = (
        Path(args.run_dir).resolve() if args.run_dir else newest_run_dir(bench_dir)
    )

    nonmath_src = (root / args.nonmath_src).resolve()
    unsupported_src = (root / args.unsupported_src).resolve()

    nonmath_qs = load_unique_qs_jsonl(nonmath_src)
    unsupported_qs = load_unique_qs_jsonl(unsupported_src)

    if args.n_nonmath > len(nonmath_qs):
        raise SystemExit(
            f"Need n_nonmath={args.n_nonmath} but only have {len(nonmath_qs)} unique in {nonmath_src}"
        )
    if args.n_unsupported > len(unsupported_qs):
        raise SystemExit(
            f"Need n_unsupported={args.n_unsupported} but only have {len(unsupported_qs)} unique in {unsupported_src}"
        )

    seed0 = args.seed if args.seed >= 0 else stable_seed(run_dir.name)
    rng_nm = random.Random(seed0)
    rng_us = random.Random(seed0 + 1)

    new_nonmath = rng_nm.sample(nonmath_qs, args.n_nonmath)
    new_unsupported = rng_us.sample(unsupported_qs, args.n_unsupported)

    nonmath_out = run_dir / "nonmath.jsonl"
    unsupported_out = run_dir / "unsupported.jsonl"

    bak_nm = backup(nonmath_out)
    bak_us = backup(unsupported_out)
    if bak_nm:
        print(f"Backed up nonmath.jsonl -> {bak_nm}")
    if bak_us:
        print(f"Backed up unsupported.jsonl -> {bak_us}")

    write_bench_jsonl(nonmath_out, new_nonmath, gold="pass")
    write_bench_jsonl(unsupported_out, new_unsupported, gold="pass")

    print(f"RUN_DIR={run_dir}")
    print(f"OK nonmath.jsonl: rows={len(new_nonmath)} unique={len(set(new_nonmath))}")
    print(
        f"OK unsupported.jsonl: rows={len(new_unsupported)} unique={len(set(new_unsupported))}"
    )


if __name__ == "__main__":
    main()
