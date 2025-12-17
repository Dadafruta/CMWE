#!/usr/bin/env python3
"""
Index and discovery tool for the scripts/ folder.

Examples:
  python -m scripts --list
  python -m scripts --grep daily
  python -m scripts --show scripts.eval_daily_mathgate_bench_v4
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable


def _iter_script_files(scripts_dir: Path) -> Iterable[Path]:
    for p in sorted(scripts_dir.glob("*.py")):
        if p.name in {"__init__.py", "__main__.py"}:
            continue
        yield p


def _first_doc_line(py_file: Path) -> str:
    try:
        src = py_file.read_text(encoding="utf-8")
        mod = ast.parse(src)
        doc = ast.get_docstring(mod) or ""
        return doc.strip().splitlines()[0].strip() if doc.strip() else ""
    except Exception:
        return ""


def _module_name(py_file: Path) -> str:
    return f"scripts.{py_file.stem}"


def main() -> None:
    ap = argparse.ArgumentParser(description="List and inspect scripts/*.py")
    ap.add_argument(
        "--list", action="store_true", help="List all scripts with one-line summary"
    )
    ap.add_argument(
        "--grep", type=str, default="", help="Filter by substring (case-insensitive)"
    )
    ap.add_argument(
        "--show",
        type=str,
        default="",
        help="Show full docstring for a module, e.g. scripts.foo",
    )
    args = ap.parse_args()

    scripts_dir = Path(__file__).resolve().parent

    rows = []
    for f in _iter_script_files(scripts_dir):
        mod = _module_name(f)
        line = _first_doc_line(f)
        rows.append((mod, line))

    if args.grep:
        g = args.grep.lower()
        rows = [(m, s) for (m, s) in rows if g in m.lower() or g in s.lower()]

    if args.show:
        target = args.show.strip()
        target_file = (
            scripts_dir / (target.split(".", 1)[1] + ".py")
            if target.startswith("scripts.")
            else None
        )
        if not target_file or not target_file.exists():
            raise SystemExit(f"Module not found: {args.show}")
        src = target_file.read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(src)) or "(no module docstring)"
        print(f"{args.show}\n{'=' * len(args.show)}\n{doc}")
        return

    # default: list
    if not args.list:
        args.list = True

    if args.list:
        w = max((len(m) for (m, _) in rows), default=10)
        for m, s in rows:
            print(f"{m:<{w}}  {s}")


if __name__ == "__main__":
    main()
