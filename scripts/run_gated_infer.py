#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
REQUIRED_KEYS = {"route", "risk", "scale", "refused", "out"}


def _strip_ansi(s: str) -> str:
    # Also normalize carriage returns (tqdm/progress bars)
    return ANSI_RE.sub("", s).replace("\r", "\n")


def _looks_like_payload(obj: Any) -> bool:
    return isinstance(obj, dict) and REQUIRED_KEYS.issubset(obj.keys())


def _extract_last_payload(text: str) -> Optional[dict[str, Any]]:
    """
    Find the last JSON object that contains REQUIRED_KEYS, even if logs include
    progress bars and other junk.

    Strategy:
      1) reverse-scan for full-line JSON objects
      2) fallback: scan every '{' and try JSONDecoder.raw_decode
    """
    cleaned = _strip_ansi(text)

    # 1) Fast path: reverse-scan full-line JSON
    for line in reversed(cleaned.splitlines()):
        s = line.strip()
        if not (s.startswith("{") and s.endswith("}")):
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if _looks_like_payload(obj):
            return obj

    # 2) Fallback: scan every '{' and raw_decode
    dec = json.JSONDecoder()
    last: Optional[dict[str, Any]] = None
    for m in re.finditer(r"\{", cleaned):
        i = m.start()
        try:
            obj, _end = dec.raw_decode(cleaned, i)
        except Exception:
            continue
        if _looks_like_payload(obj):
            last = obj
    return last


def _emit_one_json(obj: dict[str, Any]) -> None:
    # EXACTLY ONE JSON OBJECT to stdout, machine-parseable
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    started = time.time()

    # IMPORTANT: catch argparse SystemExit so we still emit JSON
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--th_math", type=float, default=0.10)
    parser.add_argument("--th_cite", type=float, default=0.80)
    parser.add_argument("--cap_cite", type=float, default=1.0)
    parser.add_argument("-q", "--q", dest="q", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory to write raw logs. Default: /tmp/cmwe_smoke_<timestamp>",
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 2
        _emit_one_json(
            {
                "route": "error",
                "risk": 1.0,
                "scale": 0.0,
                "refused": True,
                "out": "",
                "ok": False,
                "returncode": code,
                "raw_log": "",
                "duration_s": round(time.time() - started, 3),
                "error": "argparse failed (see stderr for usage)",
            }
        )
        return code

    repo_root = Path(__file__).resolve().parents[1]
    gated_infer_py = repo_root / "scripts" / "gated_infer.py"

    run_dir = Path(
        args.run_dir
        if args.run_dir
        else f"/tmp/cmwe_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_log = raw_dir / f"{int(time.time() * 1e6)}.log"

    cmd: list[str] = [
        args.python,
        str(gated_infer_py),
        "--th_math",
        str(args.th_math),
        "--th_cite",
        str(args.th_cite),
        "--cap_cite",
        str(args.cap_cite),
        "--q",
        str(args.q),  # NOTE: gated_infer.py accepts --q (not -q)
    ]
    if args.data is not None:
        cmd += ["--data", str(args.data)]

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    combined = ""
    rc = 0
    err_msg: Optional[str] = None

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=float(args.timeout),
            check=False,
        )
        combined = proc.stdout or ""
        rc = int(proc.returncode)
    except subprocess.TimeoutExpired as e:
        combined = (e.stdout or "") + "\n[TIMEOUT]\n" + (e.stderr or "")
        rc = 124
        err_msg = f"timeout after {args.timeout}s"
    except Exception as e:
        combined = f"[WRAPPER_EXCEPTION] {type(e).__name__}: {e}\n"
        rc = 2
        err_msg = f"{type(e).__name__}: {e}"

    # Always write raw log
    raw_log.write_text(combined, encoding="utf-8", errors="replace")

    payload = _extract_last_payload(combined)

    if payload is None:
        result: dict[str, Any] = {
            "route": "error",
            "risk": 1.0,
            "scale": 0.0,
            "refused": True,
            "out": "",
            "error": "No JSON payload found in logs",
        }
    else:
        # copy so we don't mutate extracted dict in weird ways
        result = dict(payload)

    # Attach metadata (P0 requirements)
    result["ok"] = bool(payload is not None and rc == 0)
    result["returncode"] = rc
    result["raw_log"] = str(raw_log)
    result["duration_s"] = round(time.time() - started, 3)
    result["cmd"] = cmd
    if err_msg and "error" not in result:
        result["error"] = err_msg

    _emit_one_json(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
