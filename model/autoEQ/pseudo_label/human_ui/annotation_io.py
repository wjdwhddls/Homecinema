"""Append-only CSV I/O for human annotations.

파일: <base_dir>/labels/human_annotations.csv
컬럼: evaluator_id, window_id, v, a, notes, ts_iso, duration_sec

fcntl.flock 기반 동시 쓰기 안전 (macOS/linux).
"""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMNS = ["evaluator_id", "window_id", "v", "a", "notes", "ts_iso", "duration_sec"]


def annotation_path(base_dir: Path) -> Path:
    return base_dir / "labels" / "human_annotations.csv"


def ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)


def append_annotation(
    base_dir: Path,
    evaluator_id: str,
    window_id: str,
    v: float,
    a: float,
    notes: str = "",
    duration_sec: float = 0.0,
) -> None:
    path = annotation_path(base_dir)
    ensure_file(path)
    row = [
        evaluator_id, window_id,
        f"{v:.4f}", f"{a:.4f}",
        notes.replace("\n", " "),
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        f"{duration_sec:.2f}",
    ]
    with path.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.writer(f)
            w.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_all(base_dir: Path) -> pd.DataFrame:
    path = annotation_path(base_dir)
    if not path.is_file():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def latest_per_evaluator(df: pd.DataFrame) -> pd.DataFrame:
    """(evaluator, window) 당 최신 row만 유지 (ts_iso 기준)."""
    if df.empty:
        return df
    df = df.sort_values("ts_iso")
    return df.drop_duplicates(subset=["evaluator_id", "window_id"], keep="last")


def progress_by_evaluator(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    latest = latest_per_evaluator(df)
    return latest.groupby("evaluator_id")["window_id"].nunique().to_dict()


def aggregate_ratings(df: pd.DataFrame) -> dict:
    """window_id → list[v] / list[a] (평가자별)."""
    latest = latest_per_evaluator(df)
    out: dict[str, dict[str, list[float]]] = {}
    for _, r in latest.iterrows():
        w = r["window_id"]
        if w not in out:
            out[w] = {"v": [], "a": []}
        out[w]["v"].append(float(r["v"]))
        out[w]["a"].append(float(r["a"]))
    return out
