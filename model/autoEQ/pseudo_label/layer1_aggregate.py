"""Layer 1 집계 — Essentia(3) + Visual(3) 6 모델 calibration + ensemble V/A.

입력:
  - layer1_essentia.csv: film_id, window_id, deam_v, deam_a, emomusic_v, emomusic_a, muse_v, muse_a
  - layer1_visual.csv  : film_id, window_id, emonet_v, emonet_a, emonet_detected, veatic_v, veatic_a, clip_v, clip_a

단계:
  1. 두 CSV를 (film_id, window_id) 기준 inner join.
  2. 각 모델 V, A 축을 **robust scale**: (x - median) / (MAD * k) * target_std
     MAD 계수 k=1.4826 (normal consistency), target_std=0.3 → Essentia·CLIP·VEATIC 동일 범위로 정렬.
     EmoNet은 detected=1 행만으로 median/MAD 산출. detected=0 행은 calibrated NaN 유지.
  3. Per-row ensemble: V/A 각각 사용 가능한 모델들 mean, std.
     EmoNet NaN인 row는 5 모델 ensemble, 그 외는 6 모델.

출력:
  film_id, window_id,
  ensemble_v, ensemble_a, ensemble_std_v, ensemble_std_a, n_models,
  deam_v_cal, deam_a_cal, emomusic_v_cal, emomusic_a_cal, muse_v_cal, muse_a_cal,
  emonet_v_cal, emonet_a_cal, emonet_detected,
  veatic_v_cal, veatic_a_cal, clip_v_cal, clip_a_cal

Usage:
  python -m model.autoEQ.pseudo_label.layer1_aggregate \\
    --essentia_csv dataset/autoEQ/CCMovies/labels/layer1_essentia.csv \\
    --visual_csv   dataset/autoEQ/CCMovies/labels/layer1_visual.csv \\
    --output_csv   dataset/autoEQ/CCMovies/labels/layer1_aggregate.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["deam", "emomusic", "muse", "emonet", "veatic", "clip"]
AXES = ["v", "a"]
MAD_CONSISTENCY = 1.4826
TARGET_STD = 0.3


def robust_scale(x: np.ndarray, target_std: float = TARGET_STD) -> tuple[np.ndarray, float, float]:
    """Return (scaled, median, mad_scaled)."""
    mask = ~np.isnan(x)
    if mask.sum() == 0:
        return np.full_like(x, np.nan, dtype=float), float("nan"), float("nan")
    med = float(np.median(x[mask]))
    mad = float(np.median(np.abs(x[mask] - med))) * MAD_CONSISTENCY
    if mad < 1e-6:
        mad = 1.0
    out = np.full_like(x, np.nan, dtype=float)
    out[mask] = (x[mask] - med) / mad * target_std
    return out, med, mad


def aggregate(essentia_csv: Path, visual_csv: Path, output_csv: Path) -> dict:
    df_e = pd.read_csv(essentia_csv)
    df_v = pd.read_csv(visual_csv)

    print(f"[info] essentia rows: {len(df_e)}, visual rows: {len(df_v)}")
    df = df_e.merge(df_v, on=["film_id", "window_id"], how="inner")
    print(f"[info] joined rows: {len(df)}")

    calibration_stats = {}
    for model in MODELS:
        for axis in AXES:
            col = f"{model}_{axis}"
            if col not in df.columns:
                raise RuntimeError(f"missing column: {col}")
            raw = pd.to_numeric(df[col], errors="coerce").to_numpy()
            scaled, med, mad = robust_scale(raw)
            df[f"{col}_cal"] = scaled
            n_valid = int((~np.isnan(raw)).sum())
            calibration_stats[col] = {
                "median": med, "mad_scaled": mad, "n_valid": n_valid
            }

    cal_v_cols = [f"{m}_v_cal" for m in MODELS]
    cal_a_cols = [f"{m}_a_cal" for m in MODELS]
    cal_v_mat = df[cal_v_cols].to_numpy()
    cal_a_mat = df[cal_a_cols].to_numpy()

    df["ensemble_v"] = np.nanmean(cal_v_mat, axis=1)
    df["ensemble_a"] = np.nanmean(cal_a_mat, axis=1)
    df["ensemble_std_v"] = np.nanstd(cal_v_mat, axis=1, ddof=0)
    df["ensemble_std_a"] = np.nanstd(cal_a_mat, axis=1, ddof=0)
    df["n_models"] = (~np.isnan(cal_v_mat)).sum(axis=1)

    output_cols = [
        "film_id", "window_id",
        "ensemble_v", "ensemble_a", "ensemble_std_v", "ensemble_std_a", "n_models",
        *[f"{m}_{a}_cal" for m in MODELS for a in AXES],
        "emonet_detected",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"[done] wrote {len(df_out)} rows → {output_csv}")

    summary = {
        "joined_rows": len(df),
        "calibration": calibration_stats,
        "ensemble_v_mean": float(df["ensemble_v"].mean()),
        "ensemble_a_mean": float(df["ensemble_a"].mean()),
        "ensemble_std_v_mean": float(df["ensemble_std_v"].mean()),
        "ensemble_std_a_mean": float(df["ensemble_std_a"].mean()),
        "n_models_mean": float(df["n_models"].mean()),
        "rows_with_emonet": int((df["n_models"] == 6).sum()),
        "rows_without_emonet": int((df["n_models"] == 5).sum()),
    }
    stats_path = output_csv.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(summary, indent=2))
    print(f"[info] stats → {stats_path}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--essentia_csv", type=Path, required=True)
    p.add_argument("--visual_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    args = p.parse_args()
    aggregate(args.essentia_csv, args.visual_csv, args.output_csv)


if __name__ == "__main__":
    main()
