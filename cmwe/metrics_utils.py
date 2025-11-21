"""Shared utilities for evaluation and dataset preparation.

The helpers in this module standardize how evaluation files are loaded,
ensure consistent column naming, and compute the headline metrics used
throughout the CMWE experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd

QUESTION_KEYS = ("q", "prompt", "question", "input", "text")
ANSWER_KEYS = ("a", "answer", "target", "gold")
UNANS_KEYS = ("unanswerable", "is_unanswerable", "unanswerable?", "refuse")


def load_jsonl(path: str | Path) -> List[MutableMapping]:
    """Load a JSONL file and validate every line.

    Blank lines are ignored, but malformed JSON will raise a ``ValueError``
    with a 1-indexed line number to make debugging easier.
    """
    p = Path(path)
    rows: List[MutableMapping] = []
    for idx, line in enumerate(p.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"{p} line {idx} is not valid JSON") from exc
    return rows


def _first_present(row: Mapping, keys: Sequence[str], default=None):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def normalize_jsonl_rows(rows: Iterable[Mapping], add_ids: bool = True) -> List[Dict]:
    """Normalize dataset rows into a consistent schema.

    The output rows expose ``id``, ``q`` (question/prompt text), ``a`` (answer,
    if present), and ``unanswerable`` (bool). ``id`` is taken from the source
    when available; otherwise a stable incremental index is used when
    ``add_ids`` is ``True``.
    """
    normalized: List[Dict] = []
    for idx, row in enumerate(rows):
        q = _first_present(row, QUESTION_KEYS, default="")
        a = _first_present(row, ANSWER_KEYS, default=None)
        unans = bool(_first_present(row, UNANS_KEYS, default=False))
        rec = {
            "id": row.get("id", idx) if add_ids else row.get("id"),
            "q": q,
            "a": a,
            "unanswerable": unans,
        }
        normalized.append(rec)
    return normalized


def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a stable ``id`` column present."""
    if "id" in df.columns:
        return df.copy()
    out = df.copy()
    out.insert(0, "id", range(len(out)))
    return out


def coerce_bool(series: pd.Series) -> pd.Series:
    """Safely coerce a Series to booleans with False as the default."""
    return series.fillna(False).astype(bool)


def normalize_result_frame(
    df: pd.DataFrame, gold_rows: Iterable[Mapping] | None = None
) -> pd.DataFrame:
    """Normalize gate/router outputs to a common schema.

    If ``gold_rows`` is provided and ``unanswerable`` is missing, the function
    will look up labels by ``id`` after normalizing the JSONL rows.
    """
    out = ensure_id_column(df)
    if "unanswerable" not in out.columns and gold_rows is not None:
        lookup = {r["id"]: r["unanswerable"] for r in normalize_jsonl_rows(gold_rows)}
        out["unanswerable"] = out["id"].map(lookup)
    for col in ("unanswerable", "correct", "refused"):
        if col in out.columns:
            out[col] = coerce_bool(out[col])
    return out


def headline_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute headline metrics used across the project.

    Metrics:
    - ``acc_answerables``: accuracy on answerable rows.
    - ``refusal_on_unanswerables``: refusal rate on unanswerables.
    - ``false_refusal_on_answerables``: refusal rate on answerables.
    """
    required = {"unanswerable", "correct", "refused"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    answerables = df[~df["unanswerable"]]
    unanswerables = df[df["unanswerable"]]

    def safe_mean(series: pd.Series) -> float:
        return float(series.mean()) if len(series) else float("nan")

    return {
        "N": float(len(df)),
        "n_answerable": float(len(answerables)),
        "n_unanswerable": float(len(unanswerables)),
        "acc_answerables": safe_mean(answerables["correct"]),
        "refusal_on_unanswerables": safe_mean(unanswerables["refused"]),
        "false_refusal_on_answerables": safe_mean(answerables["refused"]),
    }


def add_unanswerable_from_jsonl(
    df: pd.DataFrame, jsonl_path: str | Path
) -> pd.DataFrame:
    """Attach ``unanswerable`` labels to a result frame using a JSONL file."""
    gold = normalize_jsonl_rows(load_jsonl(jsonl_path))
    return normalize_result_frame(df, gold_rows=gold)
