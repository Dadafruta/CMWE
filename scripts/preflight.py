from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def die(msg: str, code: int = 1) -> None:
    print(f"[preflight] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def ok(msg: str) -> None:
    print(f"[preflight] OK: {msg}")


def import_mod(name: str, label: str | None = None) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        die(f"cannot import {name}: {e}")
    ver = getattr(mod, "__version__", None)
    ok(f"import {label or name} ({ver or 'unknown'})")
    return ver or "unknown"


def check_not_lfs_pointer(path: Path) -> None:
    if not path.exists():
        die(f"missing required file: {path}")

    head = path.read_bytes()[:120]
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        die(
            f"{path} is a Git LFS *pointer* (not real weights).\n"
            "Fix:\n"
            "  git lfs install\n"
            "  git lfs pull\n"
            "Then re-run: python -m scripts.preflight"
        )

    sz = path.stat().st_size
    if sz < 1024 * 1024:
        die(f"{path} is unexpectedly small ({sz} bytes) — likely not real weights.")
    ok(f"{path} looks like real binary ({sz} bytes)")


def check_risk_detector() -> None:
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    # Default location (matches what you've shown)
    det_path = ROOT / "artifacts" / "risk_detector.joblib"

    # If scripts.gated_infer defines DET_PATH, prefer that
    try:
        from scripts.gated_infer import DET_PATH as DET_PATH_FROM_CODE  # type: ignore

        det_path = ROOT / str(DET_PATH_FROM_CODE)
    except Exception:
        pass

    if not det_path.exists():
        die(f"risk detector not found at: {det_path}")

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always", InconsistentVersionWarning)
        det = joblib.load(det_path)

    bad = [w for w in ws if issubclass(w.category, InconsistentVersionWarning)]
    if bad:
        die(
            "scikit-learn version mismatch while loading risk detector.\n"
            f"Artifact: {det_path}\n"
            f"Warning: {bad[0].message}\n"
            "Fix: install the exact sklearn version used to build the artifact "
            "(we pin scikit-learn==1.4.1.post1 in requirements.txt)."
        )

    try:
        proba = det.predict_proba(["2+2?"])
        shape = getattr(proba, "shape", None)
        if shape != (1, 2):
            die(f"risk detector predict_proba returned unexpected shape: {shape}")
    except Exception as e:
        die(f"risk detector predict_proba failed: {e}")

    ok("risk detector loads and predict_proba works")


def main() -> None:
    os.chdir(ROOT)
    print(f"[preflight] repo root: {ROOT}")

    if sys.version_info < (3, 10):
        die(f"Python {sys.version.split()[0]} too old; need >=3.10")
    ok(f"python {sys.version.split()[0]}")

    # Core deps
    import_mod("numpy")
    import_mod("joblib")
    import_mod("sklearn", "scikit-learn")

    # Model/runtime deps
    import_mod("torch")
    import_mod("transformers")
    import_mod("accelerate")
    import_mod("peft")

    # If gated_infer uses 4-bit loading, bnb must exist.
    # Fail fast here so you don't get a cryptic transformers PackageNotFoundError later.
    import_mod("bitsandbytes")

    # Tokenizer deps (avoid the 'slow tokenizer' / protobuf errors you hit)
    import_mod("sentencepiece")
    import_mod("google.protobuf", "protobuf")

    # Check Git-LFS-backed adapter files are real
    check_not_lfs_pointer(
        ROOT / "adapters" / "math_guard" / "adapter_model.safetensors"
    )
    check_not_lfs_pointer(
        ROOT / "adapters" / "citation_guard" / "adapter_model.safetensors"
    )

    # Check risk detector artifact is loadable with pinned sklearn
    check_risk_detector()

    ok("preflight complete")


if __name__ == "__main__":
    main()
