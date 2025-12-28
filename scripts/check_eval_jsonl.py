#!/usr/bin/env python3
"""Check eval jsonl.

Run:
  python -m scripts.check_eval_jsonl --help
"""

import sys
import json
import collections
import re

_ws = re.compile(r"\s+")


def norm_q(s):
    return _ws.sub(" ", s.strip().lower())


def main(paths):
    for p in paths:
        rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        print("\n==", p)
        print("N:", len(rows))
        buckets = collections.Counter(r.get("bucket", "?") for r in rows)
        print("buckets:", dict(buckets))
        unans = sum(1 for r in rows if r.get("unanswerable"))
        print("unanswerable_frac:", round(unans / max(1, len(rows)), 4))
        keys = [norm_q(r.get("q", "")) for r in rows]
        print("exact_q_dups(after simple norm):", len(keys) - len(set(keys)))
        print("example:", rows[0] if rows else None)


if __name__ == "__main__":
    main(sys.argv[1:])
