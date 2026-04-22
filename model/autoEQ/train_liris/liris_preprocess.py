"""LIRIS-ACCEDE preprocessing — metadata + split + distribution report + K=7 gate.

V5-FINAL §7, §8 reference.

Inputs
------
  dataset/autoEQ/liris/annotations/annotations/ACCEDEranking.txt
      tab-delimited: id, name, valenceRank, arousalRank,
                     valenceValue, arousalValue, valenceVariance, arousalVariance
  dataset/autoEQ/liris/annotations/annotations/ACCEDEsets.txt
      tab-delimited: id, name, set (0=test, 1=learning, 2=validation)
  dataset/autoEQ/liris/data/ACCEDEdescription.xml
      per <media>: <id>, <name>, <movie>, <start>, <end>

Outputs
-------
  dataset/autoEQ/liris/liris_metadata.csv
      columns: id, name, film_id, split, v_raw, a_raw, v_var, a_var,
               v_norm, a_norm, mood_k7, quadrant_k4, start_frame, end_frame
  runs/phase1_preprocess/distribution_report.json
      V/A range, variance p50/p75/max, quadrant/mood distributions, split/film counts
  runs/phase1_preprocess/gate_k7.json
      §8 K=7 data-sufficiency gate result (strict)

Usage
-----
  python -m model.autoEQ.train_liris.liris_preprocess
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import pandas as pd

from ..train.dataset import MOOD_CENTERS, va_to_mood

GEMS_LABELS = [
    "Tension",
    "Sadness",
    "Peacefulness",
    "JoyfulActivation",
    "Tenderness",
    "Power",
    "Wonder",
]
QUADRANT_LABELS = ["HVHA", "HVLA", "LVHA", "LVLA"]
SPLIT_MAP = {0: "test", 1: "train", 2: "val"}


def parse_ranking(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    expected = {
        "id", "name", "valenceRank", "arousalRank",
        "valenceValue", "arousalValue",
        "valenceVariance", "arousalVariance",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df.rename(
        columns={
            "valenceValue": "v_raw",
            "arousalValue": "a_raw",
            "valenceVariance": "v_var",
            "arousalVariance": "a_var",
        }
    )[["id", "name", "v_raw", "a_raw", "v_var", "a_var"]]


def parse_sets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if not {"id", "name", "set"}.issubset(df.columns):
        raise ValueError(f"{path}: missing columns")
    df["split"] = df["set"].map(SPLIT_MAP)
    return df[["id", "split"]]


def parse_description_xml(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for media in root.findall("media"):
        rows.append(
            {
                "id": int(media.findtext("id")),
                "film_id": media.findtext("movie"),
                "start_frame": int(media.findtext("start") or -1),
                "end_frame": int(media.findtext("end") or -1),
            }
        )
    return pd.DataFrame(rows)


def quadrant_k4(v_norm: float, a_norm: float) -> int:
    if v_norm >= 0 and a_norm >= 0:
        return 0  # HVHA
    if v_norm >= 0 and a_norm < 0:
        return 1  # HVLA
    if v_norm < 0 and a_norm >= 0:
        return 2  # LVHA
    return 3  # LVLA


def build_metadata(
    ranking: pd.DataFrame,
    sets: pd.DataFrame,
    desc: pd.DataFrame,
) -> pd.DataFrame:
    df = ranking.merge(sets, on="id", how="inner")
    df = df.merge(desc, on="id", how="inner")

    df["v_norm"] = (df["v_raw"] - 3.0) / 2.0
    df["a_norm"] = (df["a_raw"] - 3.0) / 2.0
    df["mood_k7"] = [va_to_mood(v, a) for v, a in zip(df["v_norm"], df["a_norm"])]
    df["quadrant_k4"] = [
        quadrant_k4(v, a) for v, a in zip(df["v_norm"], df["a_norm"])
    ]

    columns = [
        "id", "name", "film_id", "split",
        "v_raw", "a_raw", "v_var", "a_var",
        "v_norm", "a_norm", "mood_k7", "quadrant_k4",
        "start_frame", "end_frame",
    ]
    return df[columns].sort_values("id").reset_index(drop=True)


def assert_integrity(df: pd.DataFrame) -> dict:
    """Verify film-level split purity and ID uniqueness."""
    # ID uniqueness
    if df["id"].duplicated().any():
        raise AssertionError("duplicate ids in metadata")

    # 9800 rows
    if len(df) != 9800:
        raise AssertionError(f"expected 9800 rows, got {len(df)}")

    # Film ∩ Split purity — no film appears in more than one split
    film_splits = df.groupby("film_id")["split"].nunique()
    bad = film_splits[film_splits > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} films appear across multiple splits: {bad.index.tolist()[:5]}"
        )

    # Split film counts — V5-FINAL §2-4: 40 / 40 / 80
    films_per_split = df.groupby("split")["film_id"].nunique().to_dict()
    expected = {"train": 40, "val": 40, "test": 80}
    if films_per_split != expected:
        raise AssertionError(
            f"film counts per split mismatch: {films_per_split} vs {expected}"
        )

    # Clip counts per split — 2450 / 2450 / 4900
    clips_per_split = df.groupby("split").size().to_dict()
    expected_clips = {"train": 2450, "val": 2450, "test": 4900}
    if clips_per_split != expected_clips:
        raise AssertionError(
            f"clip counts per split mismatch: {clips_per_split} vs {expected_clips}"
        )

    return {
        "films_per_split": films_per_split,
        "clips_per_split": clips_per_split,
    }


def distribution_report(df: pd.DataFrame, integrity: dict) -> dict:
    v_vals = df["v_norm"].to_numpy()
    a_vals = df["a_norm"].to_numpy()
    v_var = df["v_var"].to_numpy()
    a_var = df["a_var"].to_numpy()

    mood_counts = df["mood_k7"].value_counts().sort_index().to_dict()
    quadrant_counts = df["quadrant_k4"].value_counts().sort_index().to_dict()

    mood_dist = {
        GEMS_LABELS[k]: {
            "count": int(v),
            "ratio": round(v / len(df), 5),
        }
        for k, v in mood_counts.items()
    }
    # Fill zero classes explicitly
    for k, label in enumerate(GEMS_LABELS):
        if label not in mood_dist:
            mood_dist[label] = {"count": 0, "ratio": 0.0}

    quadrant_dist = {
        QUADRANT_LABELS[k]: {
            "count": int(v),
            "ratio": round(v / len(df), 5),
        }
        for k, v in quadrant_counts.items()
    }
    for k, label in enumerate(QUADRANT_LABELS):
        if label not in quadrant_dist:
            quadrant_dist[label] = {"count": 0, "ratio": 0.0}

    return {
        "n_clips": int(len(df)),
        "split_film_counts": integrity["films_per_split"],
        "split_clip_counts": integrity["clips_per_split"],
        "va_raw_range": {
            "v_raw_min": round(float(df["v_raw"].min()), 4),
            "v_raw_max": round(float(df["v_raw"].max()), 4),
            "a_raw_min": round(float(df["a_raw"].min()), 4),
            "a_raw_max": round(float(df["a_raw"].max()), 4),
        },
        "va_norm_range": {
            "v_norm_min": round(float(v_vals.min()), 4),
            "v_norm_max": round(float(v_vals.max()), 4),
            "v_norm_mean": round(float(v_vals.mean()), 4),
            "a_norm_min": round(float(a_vals.min()), 4),
            "a_norm_max": round(float(a_vals.max()), 4),
            "a_norm_mean": round(float(a_vals.mean()), 4),
        },
        "variance": {
            "v_var_p50": round(float(np.percentile(v_var, 50)), 4),
            "v_var_p75": round(float(np.percentile(v_var, 75)), 4),
            "v_var_max": round(float(v_var.max()), 4),
            "a_var_p50": round(float(np.percentile(a_var, 50)), 4),
            "a_var_p75": round(float(np.percentile(a_var, 75)), 4),
            "a_var_max": round(float(a_var.max()), 4),
        },
        "mood_k7_distribution": mood_dist,
        "quadrant_k4_distribution": quadrant_dist,
    }


def gate_k7_strict(mood_dist: dict, min_ratio: float = 0.01) -> dict:
    """V5-FINAL §8 — strict data sufficiency gate.

    Fails if any one of the 7 GEMS classes has ratio < 1%.
    """
    per_class = {
        label: {
            "ratio": info["ratio"],
            "passes": info["ratio"] >= min_ratio,
        }
        for label, info in mood_dist.items()
    }
    fails = [label for label, info in per_class.items() if not info["passes"]]
    passed = len(fails) == 0
    return {
        "threshold_ratio": min_ratio,
        "per_class": per_class,
        "fails": fails,
        "n_fail": len(fails),
        "n_pass": 7 - len(fails),
        "verdict": "PASS (K=7 admissible)" if passed else "FAIL → K=4 fallback",
        "passed": passed,
    }


def run(
    liris_root: Path,
    output_csv: Path,
    report_dir: Path,
) -> dict:
    ann_dir = liris_root / "annotations" / "annotations"
    ranking_path = ann_dir / "ACCEDEranking.txt"
    sets_path = ann_dir / "ACCEDEsets.txt"
    desc_path = liris_root / "data" / "ACCEDEdescription.xml"

    for p in (ranking_path, sets_path, desc_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    print(f"[phase1] parse ranking   : {ranking_path}")
    ranking = parse_ranking(ranking_path)
    print(f"[phase1] parse sets      : {sets_path}")
    sets = parse_sets(sets_path)
    print(f"[phase1] parse description.xml")
    desc = parse_description_xml(desc_path)

    print(f"[phase1] build metadata ({len(ranking)} ranking × {len(sets)} sets × {len(desc)} desc)")
    df = build_metadata(ranking, sets, desc)

    print("[phase1] integrity asserts")
    integrity = assert_integrity(df)

    print(f"[phase1] write metadata : {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("[phase1] distribution report")
    report = distribution_report(df, integrity)

    print("[phase1] §8 K=7 strict gate")
    gate = gate_k7_strict(report["mood_k7_distribution"])

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "distribution_report.json").write_text(json.dumps(report, indent=2))
    (report_dir / "gate_k7.json").write_text(json.dumps(gate, indent=2))

    print("\n[phase1] === SUMMARY ===")
    print(f"  clips         : {report['n_clips']}")
    print(f"  films/split   : {integrity['films_per_split']}")
    print(f"  clips/split   : {integrity['clips_per_split']}")
    print(f"  v_norm range  : {report['va_norm_range']['v_norm_min']}..{report['va_norm_range']['v_norm_max']}")
    print(f"  a_norm range  : {report['va_norm_range']['a_norm_min']}..{report['va_norm_range']['a_norm_max']}")
    print(f"  v_var p75     : {report['variance']['v_var_p75']}")
    print(f"  a_var p75     : {report['variance']['a_var_p75']}")
    print("  mood K=7 dist :")
    for label in GEMS_LABELS:
        info = report["mood_k7_distribution"][label]
        print(f"    {label:>18s}: {info['count']:>5d} ({info['ratio']*100:>5.2f}%)")
    print("  quadrant K=4  :")
    for label in QUADRANT_LABELS:
        info = report["quadrant_k4_distribution"][label]
        print(f"    {label}: {info['count']:>5d} ({info['ratio']*100:>5.2f}%)")
    print(f"\n  §8 K=7 gate   : {gate['verdict']}")
    if gate["fails"]:
        print(f"    failing classes: {gate['fails']}")
    return {"metadata": df, "report": report, "gate": gate}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--liris-root",
        type=Path,
        default=Path("dataset/autoEQ/liris"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset/autoEQ/liris/liris_metadata.csv"),
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("runs/phase1_preprocess"),
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="V5-FINAL §14-1: verify split integrity, V/A round-trip, "
             "variance threshold fire rate. Exits non-zero on failure.",
    )
    return p.parse_args()


def verify(output_csv: Path) -> int:
    """§14-1 invariants. Returns 0 on pass, non-zero on failure."""
    if not output_csv.is_file():
        print(f"[verify] metadata missing: {output_csv}", file=sys.stderr)
        return 2
    df = pd.read_csv(output_csv)
    errors: list[str] = []

    # 1. column integrity
    needed = {"id", "name", "film_id", "split", "v_raw", "a_raw",
              "v_var", "a_var", "v_norm", "a_norm",
              "mood_k7", "quadrant_k4", "start_frame", "end_frame"}
    missing = needed - set(df.columns)
    if missing:
        errors.append(f"missing columns: {missing}")

    # 2. Split cardinality
    n_tr = int((df.split == "train").sum())
    n_va = int((df.split == "val").sum())
    n_te = int((df.split == "test").sum())
    if (n_tr, n_va, n_te) != (2450, 2450, 4900):
        errors.append(f"split clip count mismatch: train={n_tr} val={n_va} test={n_te} (want 2450/2450/4900)")
    for sp, want in [("train", 40), ("val", 40), ("test", 80)]:
        got = df[df.split == sp].film_id.nunique()
        if got != want:
            errors.append(f"{sp} film count: got {got} want {want}")

    # 3. film overlap (zero expected)
    a = set(df[df.split == "train"].film_id)
    b = set(df[df.split == "val"].film_id)
    c = set(df[df.split == "test"].film_id)
    if a & b: errors.append(f"train/val film overlap: {len(a&b)}")
    if a & c: errors.append(f"train/test film overlap: {len(a&c)}")
    if b & c: errors.append(f"val/test film overlap: {len(b&c)}")

    # 4. V/A round-trip (v_raw - 3) / 2 == v_norm
    v_err = (((df.v_raw - 3) / 2) - df.v_norm).abs().max()
    a_err = (((df.a_raw - 3) / 2) - df.a_norm).abs().max()
    if v_err > 1e-6: errors.append(f"v_norm round-trip error: {v_err}")
    if a_err > 1e-6: errors.append(f"a_norm round-trip error: {a_err}")

    # 5. Variance threshold fire rate (§2-2 expect 6-10% for AND)
    v_thr, a_thr = 0.117, 0.164
    fire_and = float(((df.v_var > v_thr) & (df.a_var > a_thr)).mean())
    if not (0.03 <= fire_and <= 0.15):
        errors.append(f"variance AND fire rate {fire_and:.3f} outside §2-2 expected 3-15%")

    # 6. §2-5: JA (mood_k7 == 3) count must be 0
    ja_n = int((df.mood_k7 == 3).sum())
    if ja_n != 0:
        errors.append(f"§2-5: JoyfulActivation count {ja_n} != 0 (V/A range insufficient)")

    # 7. Report
    print(f"[verify] {output_csv}")
    print(f"  n_clips       = {len(df)}")
    print(f"  splits        = train={n_tr} val={n_va} test={n_te}")
    print(f"  films         = train={df[df.split=='train'].film_id.nunique()}"
          f" val={df[df.split=='val'].film_id.nunique()}"
          f" test={df[df.split=='test'].film_id.nunique()}")
    print(f"  V/A round-trip max err: v={v_err:.2e}  a={a_err:.2e}")
    print(f"  variance AND fire rate: {fire_and*100:.2f}%")
    print(f"  K=7 class 3 (JA) count: {ja_n}")
    if errors:
        print(f"\n[verify] FAIL — {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n[verify] PASS (§14-1 invariants satisfied)")
    return 0


def main() -> int:
    args = parse_args()
    if args.verify:
        return verify(args.output_csv)
    run(args.liris_root, args.output_csv, args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
