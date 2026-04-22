"""Comprehensive split/label/feature diagnostics for Phase 2a ceiling investigation.

Answers:
  1. Are film_ids leaked across splits?
  2. Do v_norm, a_norm, v_var, a_var, mood_k7, quadrant_k4 have the same
     distribution across train/val/test (KS test + means/stds)?
  3. Per-film clip count & V/A spread — are val films more "boring" than train?
  4. Intra-film V/A variance — label noise ceiling estimate.
  5. Naive baselines: (a) predict global mean; (b) predict film mean (impossible
     at inference but useful as an oracle upper bound); (c) Ridge linear probe
     on concat(X-CLIP, PANNs) features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
META = ROOT / "dataset" / "autoEQ" / "liris" / "liris_metadata.csv"
FEATS = ROOT / "data" / "features" / "liris_panns" / "features.pt"
OUT = ROOT / "runs" / "phase2a" / "split_diagnostics.json"


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return 0.0
    return float(2 * cov / denom)


def mean_ccc(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    cv = ccc(pred[:, 0], true[:, 0])
    ca = ccc(pred[:, 1], true[:, 1])
    return (cv + ca) / 2, cv, ca


def main():
    df = pd.read_csv(META)
    print(f"[load] metadata: {len(df)} rows, cols={list(df.columns)}")

    report: dict = {}

    # ── 1. Film leakage check ───────────────────────────────────────────
    leakage = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        a_films = set(df[df.split == a].film_id.unique())
        b_films = set(df[df.split == b].film_id.unique())
        overlap = a_films & b_films
        leakage[f"{a}∩{b}"] = {"n_overlap_films": len(overlap), "films": sorted(overlap)[:10]}
    report["film_leakage"] = leakage
    print("\n[1] film leakage:")
    for k, v in leakage.items():
        print(f"  {k}: {v['n_overlap_films']} films")

    # ── 2. Per-split distributions ──────────────────────────────────────
    split_stats: dict = {}
    for sp in ["train", "val", "test"]:
        sub = df[df.split == sp]
        split_stats[sp] = {
            "n_clips": int(len(sub)),
            "n_films": int(sub.film_id.nunique()),
            "v_norm_mean": float(sub.v_norm.mean()),
            "v_norm_std": float(sub.v_norm.std()),
            "a_norm_mean": float(sub.a_norm.mean()),
            "a_norm_std": float(sub.a_norm.std()),
            "v_var_mean": float(sub.v_var.mean()),
            "v_var_p75": float(sub.v_var.quantile(0.75)),
            "a_var_mean": float(sub.a_var.mean()),
            "a_var_p75": float(sub.a_var.quantile(0.75)),
            "quadrant_k4_ratio": (sub.quadrant_k4.value_counts(normalize=True).round(4)).to_dict(),
            "mood_k7_ratio": (sub.mood_k7.value_counts(normalize=True).round(4)).to_dict(),
        }
    report["split_stats"] = split_stats
    print("\n[2] Per-split basic stats:")
    for sp, s in split_stats.items():
        print(f"  {sp:5s}  clips={s['n_clips']:>4d}  films={s['n_films']:>2d}  "
              f"v_norm={s['v_norm_mean']:+.3f}±{s['v_norm_std']:.3f}  "
              f"a_norm={s['a_norm_mean']:+.3f}±{s['a_norm_std']:.3f}  "
              f"v_var={s['v_var_mean']:.3f} a_var={s['a_var_mean']:.3f}")

    print("\n[2b] quadrant_k4 ratios per split:")
    q_df = pd.DataFrame({sp: split_stats[sp]["quadrant_k4_ratio"] for sp in ["train", "val", "test"]}).fillna(0)
    print(q_df.round(4).to_string())

    # ── 3. KS test per continuous var ───────────────────────────────────
    ks_results = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        pair: dict = {}
        for col in ["v_norm", "a_norm", "v_var", "a_var"]:
            x = df[df.split == a][col].values
            y = df[df.split == b][col].values
            ks = stats.ks_2samp(x, y)
            pair[col] = {"statistic": round(float(ks.statistic), 4), "pvalue": round(float(ks.pvalue), 4)}
        ks_results[f"{a}_vs_{b}"] = pair
    report["ks_tests"] = ks_results
    print("\n[3] KS tests (null: same distribution; p<0.05 = distributions differ):")
    for pair_name, cols in ks_results.items():
        flagged = [f"{c}(D={v['statistic']:.3f},p={v['pvalue']:.3f})" for c, v in cols.items() if v["pvalue"] < 0.05]
        if flagged:
            print(f"  {pair_name}: ⚠️  {', '.join(flagged)}")
        else:
            print(f"  {pair_name}: all cols p≥0.05 (no evidence of mismatch)")

    # ── 4. Per-film V/A spread (diversity) ──────────────────────────────
    print("\n[4] Per-film V/A spread (mean clip count, within-film std of v_norm/a_norm):")
    per_film: dict = {}
    for sp in ["train", "val", "test"]:
        sub = df[df.split == sp]
        films = sub.groupby("film_id").agg(
            n_clips=("id", "count"),
            v_mean=("v_norm", "mean"),
            v_std=("v_norm", "std"),
            a_mean=("a_norm", "mean"),
            a_std=("a_norm", "std"),
        )
        per_film[sp] = {
            "film_clip_count_mean": round(float(films.n_clips.mean()), 2),
            "film_clip_count_std": round(float(films.n_clips.std()), 2),
            "film_clip_count_min": int(films.n_clips.min()),
            "film_clip_count_max": int(films.n_clips.max()),
            "within_film_v_std_mean": round(float(films.v_std.mean()), 4),
            "within_film_a_std_mean": round(float(films.a_std.mean()), 4),
            "between_film_v_std": round(float(films.v_mean.std()), 4),
            "between_film_a_std": round(float(films.a_mean.std()), 4),
        }
        s = per_film[sp]
        print(f"  {sp:5s}  clips/film={s['film_clip_count_mean']:.1f}±{s['film_clip_count_std']:.1f}  "
              f"within_v_std={s['within_film_v_std_mean']:.3f} within_a_std={s['within_film_a_std_mean']:.3f}  "
              f"between_v_std={s['between_film_v_std']:.3f} between_a_std={s['between_film_a_std']:.3f}")
    report["per_film_spread"] = per_film

    # ── 5. Naive baselines (CCC on val) ─────────────────────────────────
    tr, va = df[df.split == "train"], df[df.split == "val"]
    va_true = va[["v_norm", "a_norm"]].values

    # (5a) predict global train mean
    glob_pred = np.tile([tr.v_norm.mean(), tr.a_norm.mean()], (len(va), 1))
    m_g, v_g, a_g = mean_ccc(glob_pred, va_true)

    # (5b) predict film mean — ORACLE (uses val film_id, infeasible at inference, but informative)
    va_film_means = va.groupby("film_id")[["v_norm", "a_norm"]].transform("mean").values
    m_f, v_f, a_f = mean_ccc(va_film_means, va_true)

    # (5c) predict train-film mean centroid for val's film (also infeasible unless film shared — which it isn't!)
    # skip

    baselines = {
        "predict_global_mean": {"mean_ccc": round(m_g, 4), "ccc_v": round(v_g, 4), "ccc_a": round(a_g, 4)},
        "predict_film_mean_ORACLE": {"mean_ccc": round(m_f, 4), "ccc_v": round(v_f, 4), "ccc_a": round(a_f, 4)},
    }
    report["naive_baselines"] = baselines
    print("\n[5] Naive baselines on val:")
    print(f"  global_mean:                  mean_CCC={m_g:.4f}  (v={v_g:.4f}, a={a_g:.4f})")
    print(f"  film_mean ORACLE (film_id→mean)  mean_CCC={m_f:.4f}  (v={v_f:.4f}, a={a_f:.4f})")
    print(f"  [compare] V2 model avg:       mean_CCC=0.3564  (v=0.3220, a=0.3909)")

    # ── 6. Intra-film V/A variance → label noise ceiling ────────────────
    # If clip labels within a film have std σ, then even a perfect film-level predictor
    # has residual σ per clip. This bounds achievable CCC.
    print("\n[6] Intra-film label noise (std of clip labels within each film):")
    for sp in ["train", "val"]:
        sub = df[df.split == sp]
        stds_v = sub.groupby("film_id").v_norm.std().dropna()
        stds_a = sub.groupby("film_id").a_norm.std().dropna()
        print(f"  {sp}: within-film v_norm std  median={stds_v.median():.4f}  mean={stds_v.mean():.4f}")
        print(f"        within-film a_norm std  median={stds_a.median():.4f}  mean={stds_a.mean():.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
