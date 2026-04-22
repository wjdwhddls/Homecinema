"""V5-FINAL §6-2 / §6-3 산출 — V/A scatter + variance histogram + centroid dist.

Produces:
  runs/phase0_sanity/va_scatter.png          (§6-2)
  runs/phase0_sanity/gt_centroid_dist.json   (§6-2)
  runs/phase0_sanity/variance_dist.json      (§6-3)
  runs/phase0_sanity/variance_dist.png       (§6-3)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
META = REPO / "dataset" / "autoEQ" / "liris" / "liris_metadata.csv"
OUT = REPO / "runs" / "phase0_sanity"

# Index order must match model/autoEQ/train/dataset.py::MOOD_CENTERS
MOOD_CENTROIDS = {
    "Tension":            (-0.6, +0.7),   # 0
    "Sadness":            (-0.6, -0.4),   # 1
    "Peacefulness":       (+0.5, -0.5),   # 2
    "JoyfulActivation":   (+0.7, +0.6),   # 3
    "Tenderness":         (+0.4, -0.2),   # 4
    "Power":              (+0.2, +0.8),   # 5
    "Wonder":             (+0.5, +0.3),   # 6
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(META)
    print(f"[load] {META}  n={len(df)}")

    # ── 1. VA scatter (§6-2) ────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df.v_norm, df.a_norm, s=2, alpha=0.3, c="steelblue", label="LIRIS clip")
    for name, (v, a) in MOOD_CENTROIDS.items():
        ax.scatter([v], [a], marker="X", s=160, c="crimson", edgecolor="black", zorder=5)
        ax.annotate(name, (v, a), fontsize=9, ha="center",
                    xytext=(0, 10), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("valence (norm)")
    ax.set_ylabel("arousal (norm)")
    ax.set_title(f"LIRIS-ACCEDE V/A distribution (N={len(df)}) + GEMS K=7 centroids")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / "va_scatter.png", dpi=110)
    plt.close(fig)
    print(f"[saved] {OUT / 'va_scatter.png'}")

    # ── 2. GT centroid distribution (§6-2 / §2-5) ────────────────────────
    ratio = df.mood_k7.value_counts(normalize=True).sort_index()
    centroid_dist = {}
    # Map mood_k7 integer → name. Default deterministic order:
    mood_order = list(MOOD_CENTROIDS.keys())
    for i, name in enumerate(mood_order):
        centroid_dist[name] = {
            "index": i,
            "centroid_va": MOOD_CENTROIDS[name],
            "count": int((df.mood_k7 == i).sum()),
            "ratio": float(ratio.get(i, 0.0)),
        }
    (OUT / "gt_centroid_dist.json").write_text(json.dumps(centroid_dist, indent=2))
    print(f"[saved] {OUT / 'gt_centroid_dist.json'}")

    # ── 3. Variance distribution (§6-3) ──────────────────────────────────
    v_p = {p: float(np.percentile(df.v_var, p)) for p in (25, 50, 75, 90, 95, 99, 100)}
    a_p = {p: float(np.percentile(df.a_var, p)) for p in (25, 50, 75, 90, 95, 99, 100)}
    v_thr, a_thr = 0.117, 0.164
    v_high = (df.v_var > v_thr).mean()
    a_high = (df.a_var > a_thr).mean()
    both_high = ((df.v_var > v_thr) & (df.a_var > a_thr)).mean()
    variance_report = {
        "v_var_percentiles": v_p,
        "a_var_percentiles": a_p,
        "thresholds": {"v_p75": v_thr, "a_p75": a_thr},
        "fire_rate": {
            "v_only_above_p75": float(v_high),
            "a_only_above_p75": float(a_high),
            "both_above_p75_AND": float(both_high),
        },
    }
    (OUT / "variance_dist.json").write_text(json.dumps(variance_report, indent=2))
    print(f"[saved] {OUT / 'variance_dist.json'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df.v_var, bins=40, color="steelblue", alpha=0.85)
    axes[0].axvline(v_thr, color="crimson", ls="--", label=f"p75={v_thr}")
    axes[0].set_title(f"valence variance  (fire rate > p75 = {v_high*100:.1f}%)")
    axes[0].set_xlabel("v_var")
    axes[0].legend()
    axes[1].hist(df.a_var, bins=40, color="darkorange", alpha=0.85)
    axes[1].axvline(a_thr, color="crimson", ls="--", label=f"p75={a_thr}")
    axes[1].set_title(f"arousal variance  (fire rate > p75 = {a_high*100:.1f}%)")
    axes[1].set_xlabel("a_var")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(OUT / "variance_dist.png", dpi=110)
    plt.close(fig)
    print(f"[saved] {OUT / 'variance_dist.png'}")

    print("\n=== Summary ===")
    print(f"VA range: v=[{df.v_norm.min():.3f},{df.v_norm.max():.3f}]  "
          f"a=[{df.a_norm.min():.3f},{df.a_norm.max():.3f}]")
    print(f"Variance p75: v={v_p[75]:.4f}  a={a_p[75]:.4f}")
    print(f"AND fire rate (v>p75 & a>p75): {both_high*100:.2f}%")
    for name, info in centroid_dist.items():
        mark = "✅" if info["ratio"] >= 0.01 else "❌ FAIL"
        print(f"  {name:<18}  count={info['count']:>4d}  ratio={info['ratio']*100:>6.2f}%  {mark}")


if __name__ == "__main__":
    main()
