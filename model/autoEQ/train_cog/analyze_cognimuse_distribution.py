"""Phase 0 distribution gate.

Loads cognimuse_metadata.pt and reports:
  - Window counts per movie
  - V/A 2D quadrant percentages (HVHA/HVLA/LVHA/LVLA)
  - K-class mood percentage (K=7 default, K=4 quadrant via --quadrant)
  - σ (valence_std / arousal_std) distribution (p50/p90/p99)
  - Per-movie V/A mean

Exits with code 1 when any mood class has < GATE_THRESHOLD share — the caller
(shell / CI) treats this as a hard stop before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ..train.dataset import va_to_mood
from .dataset import COGNIMUSE_MOVIES, va_to_quadrant


GATE_THRESHOLD = 0.01  # 1 %


def _quadrant_name(v: float, a: float) -> str:
    return ("H" if v >= 0 else "L") + "V" + ("H" if a >= 0 else "L") + "A"


def analyze_distribution(
    metadata: dict[str, dict],
    num_mood_classes: int = 7,
) -> dict:
    n = len(metadata)
    movie_counts: dict[str, int] = {c: 0 for c in COGNIMUSE_MOVIES}
    quadrant_counts: dict[str, int] = {"HVHA": 0, "HVLA": 0, "LVHA": 0, "LVLA": 0}
    mood_counts: dict[int, int] = {k: 0 for k in range(num_mood_classes)}
    sigma_v: list[float] = []
    sigma_a: list[float] = []
    movie_va_sum: dict[str, tuple[float, float, int]] = {
        c: (0.0, 0.0, 0) for c in COGNIMUSE_MOVIES
    }

    for m in metadata.values():
        v = float(m["valence"])
        a = float(m["arousal"])
        code = m["movie_code"]
        movie_counts[code] = movie_counts.get(code, 0) + 1
        quadrant_counts[_quadrant_name(v, a)] += 1
        if num_mood_classes == 4:
            mood = va_to_quadrant(v, a)
        else:
            mood = va_to_mood(v, a)
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
        sigma_v.append(float(m.get("valence_std", 0.0)))
        sigma_a.append(float(m.get("arousal_std", 0.0)))
        s_v, s_a, n_m = movie_va_sum[code]
        movie_va_sum[code] = (s_v + v, s_a + a, n_m + 1)

    sv_arr = np.array(sigma_v) if sigma_v else np.array([0.0])
    sa_arr = np.array(sigma_a) if sigma_a else np.array([0.0])

    mood_pct = {
        str(k): (count / n if n else 0.0) for k, count in sorted(mood_counts.items())
    }
    min_mood_class_pct = min(mood_pct.values()) if mood_pct else 0.0

    report = {
        "n_windows": n,
        "K": num_mood_classes,
        "movie_counts": movie_counts,
        "va_quadrant_pct": {
            k: (v / n if n else 0.0) for k, v in quadrant_counts.items()
        },
        "mood_class_pct": mood_pct,
        "sigma_valence": {
            "p50": float(np.percentile(sv_arr, 50)),
            "p90": float(np.percentile(sv_arr, 90)),
            "p99": float(np.percentile(sv_arr, 99)),
        },
        "sigma_arousal": {
            "p50": float(np.percentile(sa_arr, 50)),
            "p90": float(np.percentile(sa_arr, 90)),
            "p99": float(np.percentile(sa_arr, 99)),
        },
        "movie_va_mean": {
            c: ([s_v / n_m, s_a / n_m] if n_m else [0.0, 0.0])
            for c, (s_v, s_a, n_m) in movie_va_sum.items()
        },
        "min_mood_class_pct": min_mood_class_pct,
        "gate_threshold": GATE_THRESHOLD,
        "gate_passed": min_mood_class_pct >= GATE_THRESHOLD,
    }
    return report


def _maybe_save_plots(
    metadata: dict[str, dict], output_dir: Path, num_mood_classes: int
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping plots")
        return

    plots_dir = output_dir / "distribution_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    valences = np.array([float(m["valence"]) for m in metadata.values()])
    arousals = np.array([float(m["arousal"]) for m in metadata.values()])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(valences, arousals, s=4, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("CogniMuse V/A distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "va_scatter.png", dpi=120)
    plt.close(fig)

    # σ histograms
    for arr, label in (
        ([float(m.get("valence_std", 0.0)) for m in metadata.values()], "valence_std"),
        ([float(m.get("arousal_std", 0.0)) for m in metadata.values()], "arousal_std"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(arr, bins=40)
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{label}_hist.png", dpi=120)
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 0 distribution gate")
    p.add_argument("--feature_dir", type=Path, required=True)
    p.add_argument("--split_name", type=str, default="cognimuse")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_mood_classes", type=int, default=7, choices=[4, 7])
    p.add_argument("--no_plots", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    meta_path = args.feature_dir / f"{args.split_name}_metadata.pt"
    metadata = torch.load(meta_path, weights_only=False)

    report = analyze_distribution(metadata, num_mood_classes=args.num_mood_classes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "distribution_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if not args.no_plots:
        _maybe_save_plots(metadata, args.output_dir, args.num_mood_classes)

    if not report["gate_passed"]:
        print(
            f"[PHASE 0 FAIL] min mood class pct = "
            f"{report['min_mood_class_pct']:.3%} < {GATE_THRESHOLD:.0%}"
        )
        print(
            "Response options: (a) lower lambda_mood, (b) reduce to "
            "num_mood_classes=4 (quadrant), (c) remove mood head entirely."
        )
        return 1

    print(
        f"[PHASE 0 PASS] K={report['K']} · n={report['n_windows']} · "
        f"min mood class = {report['min_mood_class_pct']:.3%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
