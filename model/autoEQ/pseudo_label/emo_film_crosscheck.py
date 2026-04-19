"""Emo-FilM 1Hz V/A 주석과 우리 라벨(Layer 1 ensemble / Layer 2 Gemini) 간 correlation QA.

Emo-FilM consensus(ds004872)에는 3편(big_buck_bunny, sintel, tears_of_steel)에 대해
`t_sec, valence, arousal` 1Hz 시계열이 있다 (V/A ∈ [-1, 1]).

단계:
  1. window metadata.csv에서 (film_id, window_id, t0, t1) 로드
  2. 매칭 영화별 va_<film>.csv 로드 (+ optional --offset_sec 적용)
  3. 각 window의 [t0, t1] 구간에 해당하는 1Hz 샘플 평균 → window_emofilm_v/a
  4. Layer 1 ensemble V/A, Layer 2 Gemini V/A 와 Pearson r, CCC 계산
  5. scatter PNG + JSON 리포트

Usage:
  python -m model.autoEQ.pseudo_label.emo_film_crosscheck \\
    --aggregate_csv dataset/autoEQ/CCMovies/labels/layer1_aggregate.csv \\
    --gemini_csv    dataset/autoEQ/CCMovies/labels/layer2_gemini.csv \\
    --emofilm_dir   dataset/autoEQ/CCMovies/emofilm_annotations \\
    --metadata_csv  dataset/autoEQ/CCMovies/windows/metadata.csv \\
    --output_dir    dataset/autoEQ/CCMovies/reports \\
    [--offset_sec 0.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EMOFILM_MAP = {
    "big_buck_bunny": "va_bigbuckbunny.csv",
    "sintel": "va_sintel.csv",
    "tears_of_steel": "va_tearsofsteel.csv",
}


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    xv, yv = x[mask], y[mask]
    xd = xv - xv.mean()
    yd = yv - yv.mean()
    denom = float(np.sqrt((xd ** 2).sum() * (yd ** 2).sum()))
    if denom < 1e-9:
        return 0.0
    return float((xd * yd).sum() / denom)


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan")
    xv, yv = x[mask], y[mask]
    mx, my = xv.mean(), yv.mean()
    vx, vy = xv.var(), yv.var()
    cov = ((xv - mx) * (yv - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    if denom < 1e-9:
        return 0.0
    return float(2.0 * cov / denom)


def find_emofilm_csv(emofilm_dir: Path, film_id: str) -> Path | None:
    """정확한 매핑 + fallback fuzzy match."""
    target = EMOFILM_MAP.get(film_id)
    if target and (emofilm_dir / target).is_file():
        return emofilm_dir / target
    # fallback: any va_*.csv whose stem substring contains film_id tokens
    film_tokens = film_id.replace("_", "").lower()
    for p in emofilm_dir.glob("va_*.csv"):
        stem = p.stem.lower().replace("va_", "").replace("_", "")
        if stem == film_tokens or stem in film_tokens or film_tokens in stem:
            return p
    return None


def aggregate_1hz_to_window(
    df_1hz: pd.DataFrame, t0: float, t1: float, offset_sec: float = 0.0
) -> tuple[float, float, int]:
    """[t0+offset, t1+offset] 구간 1Hz 샘플의 V/A 평균."""
    lo = t0 + offset_sec
    hi = t1 + offset_sec
    mask = (df_1hz["t_sec"] >= lo) & (df_1hz["t_sec"] < hi)
    sub = df_1hz[mask]
    if len(sub) == 0:
        return float("nan"), float("nan"), 0
    return float(sub["valence"].mean()), float(sub["arousal"].mean()), int(len(sub))


def crosscheck(
    aggregate_csv: Path,
    gemini_csv: Path | None,
    emofilm_dir: Path,
    metadata_csv: Path,
    output_dir: Path,
    offset_sec: float = 0.0,
) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = pd.read_csv(aggregate_csv)
    meta = pd.read_csv(metadata_csv)[["film_id", "window_id", "t0", "t1"]]
    df = agg.merge(meta, on=["film_id", "window_id"], how="inner")

    gem = None
    if gemini_csv and gemini_csv.is_file():
        gem = pd.read_csv(gemini_csv)
        df = df.merge(
            gem[["film_id", "window_id", "gemini_v", "gemini_a"]],
            on=["film_id", "window_id"], how="left",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"offset_sec": offset_sec, "films": {}}

    films = sorted(df["film_id"].unique())
    for film in films:
        emofilm_path = find_emofilm_csv(emofilm_dir, film)
        if emofilm_path is None:
            continue
        df_1hz = pd.read_csv(emofilm_path)
        if {"t_sec", "valence", "arousal"} - set(df_1hz.columns):
            print(f"[warn] {emofilm_path.name} missing required columns — skip")
            continue

        sub = df[df["film_id"] == film].copy()
        v_ef, a_ef, n_samples = [], [], []
        for _, r in sub.iterrows():
            v, a, n = aggregate_1hz_to_window(df_1hz, r["t0"], r["t1"], offset_sec)
            v_ef.append(v); a_ef.append(a); n_samples.append(n)
        sub["emofilm_v"] = v_ef
        sub["emofilm_a"] = a_ef
        sub["emofilm_n_samples"] = n_samples

        layer1_v = sub["ensemble_v"].to_numpy()
        layer1_a = sub["ensemble_a"].to_numpy()
        ef_v = sub["emofilm_v"].to_numpy()
        ef_a = sub["emofilm_a"].to_numpy()

        film_report = {
            "emofilm_source": emofilm_path.name,
            "n_windows_matched": int((~np.isnan(ef_v)).sum()),
            "n_windows_total": int(len(sub)),
            "layer1_vs_emofilm": {
                "pearson_v": pearson_r(layer1_v, ef_v),
                "pearson_a": pearson_r(layer1_a, ef_a),
                "ccc_v": ccc(layer1_v, ef_v),
                "ccc_a": ccc(layer1_a, ef_a),
            },
        }

        if gem is not None and "gemini_v" in sub.columns:
            g_v = pd.to_numeric(sub["gemini_v"], errors="coerce").to_numpy()
            g_a = pd.to_numeric(sub["gemini_a"], errors="coerce").to_numpy()
            film_report["gemini_vs_emofilm"] = {
                "pearson_v": pearson_r(g_v, ef_v),
                "pearson_a": pearson_r(g_a, ef_a),
                "ccc_v": ccc(g_v, ef_v),
                "ccc_a": ccc(g_a, ef_a),
            }

        report["films"][film] = film_report
        print(f"[info] {film}: {film_report}")

        # scatter per film
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(ef_v, layer1_v, s=8, alpha=0.5, label="Layer1")
        if gem is not None:
            axes[0].scatter(ef_v, g_v, s=8, alpha=0.5, label="Gemini", color="tab:orange")
        axes[0].plot([-1, 1], [-1, 1], "k--", linewidth=0.5)
        axes[0].set_xlabel("Emo-FilM V")
        axes[0].set_ylabel("Predicted V")
        axes[0].set_title(f"{film} — Valence")
        axes[0].legend()

        axes[1].scatter(ef_a, layer1_a, s=8, alpha=0.5, label="Layer1")
        if gem is not None:
            axes[1].scatter(ef_a, g_a, s=8, alpha=0.5, label="Gemini", color="tab:orange")
        axes[1].plot([-1, 1], [-1, 1], "k--", linewidth=0.5)
        axes[1].set_xlabel("Emo-FilM A")
        axes[1].set_ylabel("Predicted A")
        axes[1].set_title(f"{film} — Arousal")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"emofilm_scatter_{film}.png", dpi=110)
        plt.close(fig)

    # aggregate across matched films
    all_layer1 = {"pearson_v": [], "pearson_a": [], "ccc_v": [], "ccc_a": []}
    all_gemini = {"pearson_v": [], "pearson_a": [], "ccc_v": [], "ccc_a": []}
    for fr in report["films"].values():
        for k, v in fr["layer1_vs_emofilm"].items():
            if not np.isnan(v):
                all_layer1[k].append(v)
        if "gemini_vs_emofilm" in fr:
            for k, v in fr["gemini_vs_emofilm"].items():
                if not np.isnan(v):
                    all_gemini[k].append(v)
    report["aggregate_mean"] = {
        "layer1": {k: (float(np.mean(vs)) if vs else None) for k, vs in all_layer1.items()},
        "gemini": {k: (float(np.mean(vs)) if vs else None) for k, vs in all_gemini.items()},
    }

    (output_dir / "emofilm_crosscheck.json").write_text(json.dumps(report, indent=2))
    print(f"[done] → {output_dir / 'emofilm_crosscheck.json'}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_csv", type=Path, required=True)
    p.add_argument("--gemini_csv", type=Path, default=None)
    p.add_argument("--emofilm_dir", type=Path, required=True)
    p.add_argument("--metadata_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--offset_sec", type=float, default=0.0)
    args = p.parse_args()
    crosscheck(
        args.aggregate_csv, args.gemini_csv, args.emofilm_dir,
        args.metadata_csv, args.output_dir, args.offset_sec,
    )


if __name__ == "__main__":
    main()
