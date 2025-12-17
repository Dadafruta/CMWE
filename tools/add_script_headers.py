#!/usr/bin/env python3
"""
Auto-add a simple module docstring to scripts/*.py that don't have one.

This improves discoverability with `python -m scripts --list`.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


def guess_summary(stem: str) -> str:
    # heuristic summaries based on prefixes
    parts = stem.split("_")
    prefix = parts[0] if parts else stem
    rest = " ".join(parts[1:]) if len(parts) > 1 else stem

    mapping = {
        "train": "Train",
        "eval": "Evaluate",
        "build": "Build",
        "make": "Generate",
        "run": "Run",
        "plot": "Plot",
        "check": "Check",
        "fix": "Fix",
        "extract": "Extract",
        "compare": "Compare",
        "demo": "Demo",
        "summarize": "Summarize",
        "filter": "Filter",
    }
    verb = mapping.get(prefix, "Script")
    title = rest if prefix in mapping else stem.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return f"{verb} {title}."


def has_module_docstring(src: str) -> bool:
    try:
        mod = ast.parse(src)
        return bool(ast.get_docstring(mod))
    except SyntaxError:
        # don't touch syntactically invalid files
        return True


def add_docstring(src: str, summary: str, module: str) -> str:
    doc = f'"""{summary}\n\nRun:\n  python -m {module} --help\n"""\n\n'

    # preserve shebang if present
    if src.startswith("#!"):
        first_nl = src.find("\n")
        shebang = src[: first_nl + 1]
        rest = src[first_nl + 1 :]
        return shebang + doc + rest
    return doc + src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write", action="store_true", help="Apply changes (default is dry-run)"
    )
    args = ap.parse_args()

    scripts_dir = Path("scripts")
    changed = 0

    for p in sorted(scripts_dir.glob("*.py")):
        if p.name in {"__init__.py", "__main__.py"}:
            continue

        src = p.read_text(encoding="utf-8")
        if has_module_docstring(src):
            continue

        module = f"scripts.{p.stem}"
        out = add_docstring(src, guess_summary(p.stem), module)

        changed += 1
        print(f"ADD DOCSTRING: {p}")

        if args.write:
            p.write_text(out, encoding="utf-8")

    print(f"Done. Files updated: {changed}")


if __name__ == "__main__":
    main()
