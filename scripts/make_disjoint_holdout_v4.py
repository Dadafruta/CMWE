#!/usr/bin/env python3
"""Generate disjoint holdout v4.

Run:
  python -m scripts.make_disjoint_holdout_v4 --help
"""

import argparse, json, re, sys, subprocess
from pathlib import Path
from collections import Counter
import random


def norm_q(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]+", "", s)  # keep letters/digits/underscore/spaces
    return s


def get_q(r: dict) -> str:
    return r.get("q") or r.get("prompt") or r.get("question") or ""


def get_bucket(r: dict) -> str:
    return r.get("bucket") or r.get("category") or r.get("type") or "UNKNOWN"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--gen",
        default="scripts/build_mixed_eval_varied_v4.py",
        help="Generator script that supports: --size N --seed S --out PATH",
    )
    ap.add_argument("--seed-start", type=int, default=100000)
    ap.add_argument(
        "--batch",
        type=int,
        default=20000,
        help="How many candidates to generate per top-up iteration",
    )
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--shuffle-seed", type=int, default=13)
    args = ap.parse_args()

    train_p = Path(args.train)
    hold_p = Path(args.holdout)
    out_p = Path(args.out)

    train_rows = list(read_jsonl(train_p))
    hold_rows = list(read_jsonl(hold_p))

    # Desired bucket counts = whatever your current holdout has
    desired = Counter(get_bucket(r) for r in hold_rows)

    train_keys = set()
    for r in train_rows:
        q = get_q(r)
        k = norm_q(q)
        if k:
            train_keys.add(k)

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(hold_rows)

    picked = []
    picked_keys = set()
    picked_counts = Counter()

    def try_pick(row):
        b = get_bucket(row)
        if picked_counts[b] >= desired[b]:
            return False
        k = norm_q(get_q(row))
        if not k:
            return False
        if k in train_keys or k in picked_keys:
            return False
        picked.append(row)
        picked_keys.add(k)
        picked_counts[b] += 1
        return True

    # First pass: keep what we can from existing holdout
    for r in hold_rows:
        try_pick(r)

    def remaining():
        return {b: desired[b] - picked_counts[b] for b in desired}

    rem = remaining()
    need = sum(v for v in rem.values() if v > 0)
    print("initial kept:", len(picked), "need:", need, "remaining by bucket:", rem)

    # Top-up by generating more candidates if needed
    for it in range(args.max_iters):
        rem = remaining()
        need = sum(v for v in rem.values() if v > 0)
        if need <= 0:
            break

        seed = args.seed_start + it
        tmp = Path(f"/tmp/cmwe_holdout_cand_{seed}.jsonl")

        cmd = [
            sys.executable,
            args.gen,
            "--size",
            str(args.batch),
            "--seed",
            str(seed),
            "--out",
            str(tmp),
        ]
        subprocess.run(cmd, check=True)

        cand = list(read_jsonl(tmp))
        rng.shuffle(cand)

        added = 0
        for r in cand:
            b = get_bucket(r)
            if rem.get(b, 0) <= 0:
                continue
            if try_pick(r):
                added += 1
                rem[b] -= 1
                if sum(v for v in rem.values() if v > 0) <= 0:
                    break

        try:
            tmp.unlink()
        except Exception:
            pass

        print(
            f"iter {it} (seed={seed}) added {added}; now {len(picked)}/{sum(desired.values())}"
        )

    rem = remaining()
    need = sum(v for v in rem.values() if v > 0)
    if need > 0:
        raise SystemExit(f"FAILED to fill disjoint holdout. Still missing: {rem}")

    # Re-id and write
    for i, r in enumerate(picked):
        r["id"] = i
    write_jsonl(out_p, picked)

    # Final overlap check
    out_keys = set(norm_q(get_q(r)) for r in picked if norm_q(get_q(r)))
    overlap = len(out_keys & train_keys)

    print("\nWROTE:", str(out_p))
    print("N:", len(picked))
    print("bucket_counts:", dict(Counter(get_bucket(r) for r in picked)))
    print("OVERLAP_WITH_TRAIN(norm_q):", overlap)


if __name__ == "__main__":
    main()
