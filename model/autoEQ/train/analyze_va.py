"""V/A distribution analysis for MoodEQ (spec [1-5], [3-1], [3-3]).

Summarizes:
  - Mood 7-class sample counts + imbalance ratio
  - Pairwise V/A distance distribution + percentiles (25/50/75/90)
  - Per-quadrant film counts (HVHA/HVLA/LVHA/LVLA)
  - Recommended class weights + negative-sampler thresholds

Runs against a SyntheticAutoEQDataset (for smoke tests) or against
metadata.pt produced by precompute.py (for real data after arrival).
Writes distribution_report.json + 4 PNG plots to --output_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from .dataset import (  # noqa: E402
    MOOD_CENTERS,
    _va_quadrant,
    compute_movie_va,
    va_to_mood,
)

MOOD_LABELS = [
    "Tension",
    "Sadness",
    "Peacefulness",
    "Joyful Activation",
    "Tenderness",
    "Power",
    "Wonder",
]


# -------- Core analysis --------


def compute_mood_distribution(valences, arousals) -> dict:
    """Count samples per mood label + imbalance diagnostic."""
    counts = [0] * 7
    for v, a in zip(valences, arousals):
        counts[va_to_mood(float(v), float(a))] += 1
    total = sum(counts)
    non_zero = [c for c in counts if c > 0]
    imbalance = (max(non_zero) / min(non_zero)) if non_zero else 0.0
    return {
        "labels": MOOD_LABELS,
        "counts": counts,
        "total": total,
        "imbalance_ratio": imbalance,
        "per_label_ratio": [c / total for c in counts] if total else [0.0] * 7,
    }


def compute_va_pairwise_distances(
    valences, arousals, max_pairs: int = 100_000, seed: int = 42
) -> dict:
    """Pairwise Euclidean V/A distances + 25/50/75/90 percentiles.

    Subsamples random pairs once total > max_pairs to stay memory-bounded.
    """
    v = torch.tensor(list(valences), dtype=torch.float32)
    a = torch.tensor(list(arousals), dtype=torch.float32)
    coords = torch.stack([v, a], dim=-1)
    n = coords.size(0)

    total_pairs = n * (n - 1) // 2
    if n < 2:
        dists = torch.zeros(0)
    elif total_pairs <= max_pairs:
        dists = torch.pdist(coords)
    else:
        rng = torch.Generator().manual_seed(seed)
        i = torch.randint(0, n, (max_pairs,), generator=rng)
        j = torch.randint(0, n, (max_pairs,), generator=rng)
        mask = i != j
        i, j = i[mask], j[mask]
        dists = (coords[i] - coords[j]).norm(dim=-1)

    if dists.numel() == 0:
        percentiles = {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    else:
        percentiles = {
            f"p{int(q * 100)}": torch.quantile(dists, q).item()
            for q in (0.25, 0.50, 0.75, 0.90)
        }
    return {
        "distances": dists,
        "percentiles": percentiles,
        "num_pairs": int(dists.numel()),
    }


def compute_quadrant_film_counts(movie_va: dict[int, tuple[float, float]]) -> dict:
    counts = {"HVHA": 0, "HVLA": 0, "LVHA": 0, "LVLA": 0}
    for _mid, (v, a) in movie_va.items():
        counts[_va_quadrant(v, a)] += 1
    return counts


def recommend_class_weights(
    counts: list[int], imbalance_threshold: float = 2.0
) -> list[float] | None:
    """Inverse-frequency weights (normalized to mean=1).

    Returns None when min*threshold >= max (dataset considered balanced enough).
    Empty classes are treated as count=1 to avoid div-by-zero in the weight
    vector; downstream CE loss should still skip them or accept the weight.
    """
    safe_counts = [max(c, 1) for c in counts]
    if min(safe_counts) * imbalance_threshold >= max(safe_counts):
        return None
    inv = [1.0 / c for c in safe_counts]
    mean_inv = sum(inv) / len(inv)
    return [w / mean_inv for w in inv]


def recommend_negative_sampler_thresholds(percentiles: dict) -> dict:
    """Map V/A-distance percentiles to slight/strong incongruent bands.

    Matches spec: slight = 25–50 percentile, strong >= 75 percentile.
    """
    return {
        "slight_lower": percentiles["p25"],
        "slight_upper": percentiles["p50"],
        "strong_lower": percentiles["p75"],
    }


def analyze_distribution(
    movie_ids, valences, arousals, max_pairs: int = 100_000, seed: int = 42
) -> dict:
    """Top-level entry. Returns a dict ready for JSON + plotting."""
    mood_dist = compute_mood_distribution(valences, arousals)
    va_dist = compute_va_pairwise_distances(
        valences, arousals, max_pairs=max_pairs, seed=seed
    )
    movie_va = compute_movie_va(movie_ids, valences, arousals)
    quad_counts = compute_quadrant_film_counts(movie_va)
    return {
        "mood_distribution": mood_dist,
        "va_distances": va_dist,
        "quadrant_films": quad_counts,
        "num_films": len(movie_va),
        "num_samples": len(valences),
        "recommendations": {
            "class_weights": recommend_class_weights(mood_dist["counts"]),
            "negative_sampler_thresholds": recommend_negative_sampler_thresholds(
                va_dist["percentiles"]
            ),
        },
    }


# -------- Plotting --------


def plot_va_scatter(valences, arousals, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    mood_labels = [va_to_mood(float(v), float(a)) for v, a in zip(valences, arousals)]
    for i in range(7):
        xs = [valences[j] for j, m in enumerate(mood_labels) if m == i]
        ys = [arousals[j] for j, m in enumerate(mood_labels) if m == i]
        if xs:
            ax.scatter(xs, ys, c=[cmap(i)], label=MOOD_LABELS[i], alpha=0.6, s=18)

    for i, (cv, ca) in enumerate(MOOD_CENTERS.tolist()):
        ax.scatter(cv, ca, c="black", marker="X", s=180, edgecolors="white",
                   zorder=5, label="_center" if i else "Mood centers")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("V/A scatter (colored by Mood)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_mood_counts(counts: list[int], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MOOD_LABELS))
    ax.bar(x, counts, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(MOOD_LABELS, rotation=30, ha="right")
    ax.set_ylabel("Sample count")
    ax.set_title("Mood category distribution")
    for xi, c in zip(x, counts):
        ax.text(xi, c, str(c), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_va_distance_histogram(
    distances: Tensor, percentiles: dict, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = distances.numpy() if isinstance(distances, Tensor) else np.asarray(distances)
    if vals.size == 0:
        ax.text(0.5, 0.5, "no pairs", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(vals, bins=60, color="#55A868", alpha=0.85)
        colors = {"p25": "#C44E52", "p50": "#8172B3", "p75": "#CCB974", "p90": "#64B5CD"}
        for name, v in percentiles.items():
            ax.axvline(v, color=colors.get(name, "black"), linestyle="--",
                       label=f"{name}={v:.3f}")
        ax.legend()
    ax.set_xlabel("V/A Euclidean distance")
    ax.set_ylabel("Pair count")
    ax.set_title("Pairwise V/A distance distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_quadrant_films(quadrant_counts: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = ["HVHA", "HVLA", "LVHA", "LVLA"]
    values = [quadrant_counts.get(k, 0) for k in keys]
    ax.bar(keys, values, color="#DD8452")
    for k, v in zip(keys, values):
        ax.text(k, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Film count")
    ax.set_title("Per-quadrant film counts")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# -------- Data loading --------


def _load_synthetic(n_clips: int, n_films: int, seed: int):
    from .config import TrainConfig
    from .dataset import SyntheticAutoEQDataset

    ds = SyntheticAutoEQDataset(
        num_clips=n_clips, num_films=n_films, config=TrainConfig(), seed=seed
    )
    return ds.movie_ids, ds.valences.tolist(), ds.arousals.tolist()


def _load_from_feature_dir(feature_dir: str, split_name: str):
    path = Path(feature_dir) / f"{split_name}_metadata.pt"
    metadata = torch.load(path, weights_only=False)
    movie_ids: list[int] = []
    valences: list[float] = []
    arousals: list[float] = []
    for wid in sorted(metadata.keys()):
        m = metadata[wid]
        movie_ids.append(int(m["movie_id"]))
        valences.append(float(m["valence"]))
        arousals.append(float(m["arousal"]))
    return movie_ids, valences, arousals


# -------- JSON sanitization --------


def _report_for_json(report: dict) -> dict:
    """Strip the raw distances tensor; keep everything else."""
    va = report["va_distances"]
    return {
        "num_samples": report["num_samples"],
        "num_films": report["num_films"],
        "mood_distribution": report["mood_distribution"],
        "va_distances": {
            "percentiles": va["percentiles"],
            "num_pairs": va["num_pairs"],
        },
        "quadrant_films": report["quadrant_films"],
        "recommendations": report["recommendations"],
    }


# -------- CLI --------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="V/A distribution analysis")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", choices=["synthetic"],
                     help="use the synthetic dataset (smoke/testing)")
    src.add_argument("--feature_dir", type=str,
                     help="directory containing {split_name}_metadata.pt")
    p.add_argument("--split_name", type=str, default="liris_accede")
    p.add_argument("--synthetic_num_clips", type=int, default=240)
    p.add_argument("--synthetic_num_films", type=int, default=12)
    p.add_argument("--max_pairs", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="analysis/latest")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    if args.source == "synthetic":
        movie_ids, valences, arousals = _load_synthetic(
            args.synthetic_num_clips, args.synthetic_num_films, args.seed
        )
    else:
        movie_ids, valences, arousals = _load_from_feature_dir(
            args.feature_dir, args.split_name
        )

    report = analyze_distribution(
        movie_ids, valences, arousals, max_pairs=args.max_pairs, seed=args.seed
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "distribution_report.json").open("w") as f:
        json.dump(_report_for_json(report), f, indent=2)

    plot_va_scatter(valences, arousals, out / "va_scatter.png")
    plot_mood_counts(report["mood_distribution"]["counts"], out / "mood_counts.png")
    plot_va_distance_histogram(
        report["va_distances"]["distances"],
        report["va_distances"]["percentiles"],
        out / "va_distance_hist.png",
    )
    plot_quadrant_films(report["quadrant_films"], out / "quadrant_films.png")

    md = report["mood_distribution"]
    va = report["va_distances"]
    print(f"Samples: {report['num_samples']}  Films: {report['num_films']}")
    print(f"Mood counts: {dict(zip(MOOD_LABELS, md['counts']))}")
    print(f"Imbalance ratio (max/min nonzero): {md['imbalance_ratio']:.2f}")
    print(f"V/A distance percentiles: {va['percentiles']}")
    print(f"Quadrant films: {report['quadrant_films']}")
    rec = report["recommendations"]
    if rec["class_weights"] is not None:
        print("Recommended class weights:",
              [f"{w:.2f}" for w in rec["class_weights"]])
    else:
        print("Mood distribution is reasonably balanced; no class weights needed.")
    print(f"Negative-sampler thresholds: {rec['negative_sampler_thresholds']}")
    print(f"Artifacts written to: {out}")
    return report


if __name__ == "__main__":
    main()
