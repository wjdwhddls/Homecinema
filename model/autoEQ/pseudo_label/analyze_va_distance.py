"""V/A 분포·거리 재분석 (CogniMuse/LIRIS 기존 percentile 재사용 금지 원칙 근거).

입력: layer1_aggregate.csv (ensemble_v, ensemble_a per window)
산출:
  - va_scatter.png      : 전체 V/A scatter (film별 색상)
  - va_distance_hist.png: pairwise Euclidean distance 히스토그램
  - va_distance.json    : 25/50/75/90 percentile, 4분면 분포, class balance

CogniMuse 기본 percentile(참고용, 재사용 금지):
  25=0.35, 50=0.55, 75=0.80, 90=1.05  (train_pseudo/negative_sampler.py의 관행치)

Usage:
  python -m model.autoEQ.pseudo_label.analyze_va_distance \\
    --aggregate_csv dataset/autoEQ/CCMovies/labels/layer1_aggregate.csv \\
    --output_dir    dataset/autoEQ/CCMovies/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PERCENTILES = [25, 50, 75, 90]
COGNIMUSE_REFERENCE = {"p25": 0.35, "p50": 0.55, "p75": 0.80, "p90": 1.05}


def pairwise_euclidean_sample(v: np.ndarray, a: np.ndarray, max_pairs: int = 5_000_000,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    n = len(v)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        iu = np.triu_indices(n, k=1)
        dv = v[iu[0]] - v[iu[1]]
        da = a[iu[0]] - a[iu[1]]
        return np.sqrt(dv * dv + da * da)

    rng = rng or np.random.default_rng(42)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    dv = v[i] - v[j]
    da = a[i] - a[j]
    return np.sqrt(dv * dv + da * da)


def quadrant_counts(v: np.ndarray, a: np.ndarray) -> dict:
    q = {
        "HVHA": int(((v >= 0) & (a >= 0)).sum()),
        "HVLA": int(((v >= 0) & (a < 0)).sum()),
        "LVHA": int(((v < 0) & (a >= 0)).sum()),
        "LVLA": int(((v < 0) & (a < 0)).sum()),
    }
    total = sum(q.values())
    q_pct = {k: v / total * 100 for k, v in q.items()}
    return {"counts": q, "percent": q_pct,
            "min_mood_class_pct": min(q_pct.values()) if q_pct else 0.0}


def analyze(aggregate_csv: Path, output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(aggregate_csv)
    df = df.dropna(subset=["ensemble_v", "ensemble_a"])
    v = df["ensemble_v"].to_numpy()
    a = df["ensemble_a"].to_numpy()
    print(f"[info] {len(df)} windows after NaN drop")

    distances = pairwise_euclidean_sample(v, a)
    percentiles_new = {f"p{p}": float(np.percentile(distances, p)) for p in PERCENTILES}
    print(f"[info] pairwise distance percentiles: {percentiles_new}")

    quadrants = quadrant_counts(v, a)
    print(f"[info] quadrant %: {quadrants['percent']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    for film, grp in df.groupby("film_id"):
        ax.scatter(grp["ensemble_v"], grp["ensemble_a"], s=6, alpha=0.4, label=film)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Valence (calibrated)")
    ax.set_ylabel("Arousal (calibrated)")
    ax.set_title(f"Ensemble V/A scatter (n={len(df)})")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "va_scatter.png", dpi=120)
    plt.close(fig)

    # distance hist
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(distances, bins=80, alpha=0.85)
    for p, val in percentiles_new.items():
        ax.axvline(val, linestyle="--", label=f"{p}={val:.3f}")
    for p, val in COGNIMUSE_REFERENCE.items():
        ax.axvline(val, linestyle=":", color="red", alpha=0.6,
                   label=f"CogniMuse {p}={val:.2f}")
    ax.set_xlabel("pairwise Euclidean distance in V/A space")
    ax.set_ylabel("count")
    ax.set_title("V/A pairwise distance — CC films vs CogniMuse reference")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "va_distance_hist.png", dpi=120)
    plt.close(fig)

    report = {
        "n_windows": int(len(df)),
        "ensemble_v_stats": {
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
        },
        "ensemble_a_stats": {
            "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
        },
        "pairwise_distance_percentiles": percentiles_new,
        "cognimuse_reference_percentiles": COGNIMUSE_REFERENCE,
        "percentile_diff_vs_cognimuse": {
            k: percentiles_new[k] - COGNIMUSE_REFERENCE[k] for k in COGNIMUSE_REFERENCE
        },
        "quadrants": quadrants,
        "film_count": int(df["film_id"].nunique()),
        "films": sorted(df["film_id"].unique().tolist()),
    }
    (output_dir / "va_distance.json").write_text(json.dumps(report, indent=2))
    print(f"[done] report → {output_dir / 'va_distance.json'}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()
    analyze(args.aggregate_csv, args.output_dir)


if __name__ == "__main__":
    main()
