"""Layer 3 — Gemini-primary adjudication (knowledge distillation 패턴).

입력: layer1_aggregate.csv + layer2_gemini.csv (inner join).

설계 철학 변경 (2026-04-19):
- Gemini 2.5 Pro는 multimodal foundation model로서 Layer 1 audio-only ensemble 대비
  Emo-FilM crosscheck에서 Pearson 최대 8배 개선 확인됨.
- 이전 "둘 다 동의해야 학습 사용" 규칙은 Gemini 단독 정답인 샘플을 배제 → 학습 pool 과소.
- 새 규칙: Gemini를 primary teacher로 쓰고, Layer 1 합치를 "품질 보너스"로만 사용.

규칙:
  excluded     — gemini_confidence < conf_threshold OR parse_ok == 0
                 (Gemini 자체가 확신 못하거나 응답 실패)
  auto_agreement — Gemini OK AND |Δv|<agreement_threshold AND |Δa|<agreement_threshold
                   AND ensemble_std_v<std_threshold AND ensemble_std_a<std_threshold
                   (Layer 1이 교차검증해준 고품질 샘플)
                   final = 0.4 * ensemble + 0.6 * gemini_cal (가중평균)
                   weight = 1.0
  gemini_only   — Gemini OK, 위 조건 미충족 (Layer 1 반대 또는 내부 noise)
                   final = gemini_cal (Gemini 단독)
                   weight = gemini_confidence (일반적으로 0.5-0.9)

출력:
  film_id, window_id, final_v, final_a, confidence, weight, source,
  delta_v, delta_a,
  ensemble_v, ensemble_a, ensemble_std_v, ensemble_std_a,
  gemini_v, gemini_a, gemini_confidence, gemini_parse_ok

Usage:
  python -m model.autoEQ.pseudo_label.layer3_adjudicate \\
    --aggregate_csv dataset/autoEQ/CCMovies/labels/layer1_aggregate.csv \\
    --gemini_csv    dataset/autoEQ/CCMovies/labels/layer2_gemini.csv \\
    --output_csv    dataset/autoEQ/CCMovies/labels/layer3_adjudicated.csv \\
    [--agreement_threshold 0.2 --std_threshold 0.3 --conf_threshold 0.5 \\
     --ensemble_weight 0.4 --gemini_weight 0.6]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def adjudicate(
    aggregate_csv: Path,
    gemini_csv: Path,
    output_csv: Path,
    agreement_threshold: float = 0.2,
    std_threshold: float = 0.3,
    conf_threshold: float = 0.5,
    ensemble_weight: float = 0.4,
    gemini_weight: float = 0.6,
) -> dict:
    agg = pd.read_csv(aggregate_csv)
    gem = pd.read_csv(gemini_csv)

    df = agg.merge(
        gem[["film_id", "window_id", "gemini_v", "gemini_a",
             "gemini_confidence", "parse_ok"]].rename(
            columns={"parse_ok": "gemini_parse_ok"}
        ),
        on=["film_id", "window_id"],
        how="inner",
    )
    print(f"[info] joined rows: {len(df)} (aggregate={len(agg)}, gemini={len(gem)})")

    # numeric coercion (gemini rows with parse_ok=0 may have empty cells)
    for col in ["gemini_v", "gemini_a", "gemini_confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Gemini V/A는 native [-1, 1] scale (std ~0.65), Layer 1 ensemble은 robust-calibrated
    # (std ~0.18). 직접 비교하면 scale 차이로 agreement가 과소 추정됨. Gemini도 Layer 1과
    # 동일하게 robust scale(median→0, MAD×1.4826→target_std=0.3)로 정규화한 뒤 비교한다.
    def _robust_scale(s: pd.Series, target_std: float = 0.3) -> tuple[pd.Series, float, float]:
        mask = ~s.isna()
        if mask.sum() == 0:
            return s, float("nan"), float("nan")
        med = float(s[mask].median())
        mad = float((s[mask] - med).abs().median()) * 1.4826
        if mad < 1e-6:
            mad = 1.0
        return (s - med) / mad * target_std, med, mad

    df["gemini_v_cal"], gv_med, gv_mad = _robust_scale(df["gemini_v"])
    df["gemini_a_cal"], ga_med, ga_mad = _robust_scale(df["gemini_a"])
    print(f"[info] gemini_v calibration: median={gv_med:.3f}, mad_scaled={gv_mad:.3f}")
    print(f"[info] gemini_a calibration: median={ga_med:.3f}, mad_scaled={ga_mad:.3f}")

    df["delta_v"] = (df["ensemble_v"] - df["gemini_v_cal"]).abs()
    df["delta_a"] = (df["ensemble_a"] - df["gemini_a_cal"]).abs()

    # Gemini 신뢰 가능 여부 (parse_ok + confidence)
    gemini_ok = (
        (df["gemini_parse_ok"].fillna(0) == 1)
        & (df["gemini_confidence"] >= conf_threshold)
    )
    # Layer 1 교차검증 (delta 작음 + 내부 std 작음)
    layer1_confirms = (
        (df["delta_v"] < agreement_threshold)
        & (df["delta_a"] < agreement_threshold)
        & (df["ensemble_std_v"] < std_threshold)
        & (df["ensemble_std_a"] < std_threshold)
    )
    excluded = ~gemini_ok
    auto_agreement = gemini_ok & layer1_confirms
    gemini_only = gemini_ok & ~layer1_confirms

    df["source"] = "excluded"
    df.loc[auto_agreement, "source"] = "auto_agreement"
    df.loc[gemini_only, "source"] = "gemini_only"

    df["final_v"] = np.nan
    df["final_a"] = np.nan
    df["confidence"] = 0.0
    df["weight"] = 0.0

    w_sum = ensemble_weight + gemini_weight
    ew = ensemble_weight / w_sum
    gw = gemini_weight / w_sum

    # auto_agreement: Layer 1과 Gemini 가중 평균 (cross-validated → weight=1.0)
    df.loc[auto_agreement, "final_v"] = (
        ew * df.loc[auto_agreement, "ensemble_v"]
        + gw * df.loc[auto_agreement, "gemini_v_cal"]
    )
    df.loc[auto_agreement, "final_a"] = (
        ew * df.loc[auto_agreement, "ensemble_a"]
        + gw * df.loc[auto_agreement, "gemini_a_cal"]
    )
    df.loc[auto_agreement, "confidence"] = df.loc[auto_agreement, "gemini_confidence"]
    df.loc[auto_agreement, "weight"] = 1.0

    # gemini_only: Gemini 단독 (Layer 1 반대/노이즈 → weight=gemini_confidence)
    df.loc[gemini_only, "final_v"] = df.loc[gemini_only, "gemini_v_cal"]
    df.loc[gemini_only, "final_a"] = df.loc[gemini_only, "gemini_a_cal"]
    df.loc[gemini_only, "confidence"] = df.loc[gemini_only, "gemini_confidence"]
    df.loc[gemini_only, "weight"] = df.loc[gemini_only, "gemini_confidence"]

    out_cols = [
        "film_id", "window_id",
        "final_v", "final_a", "confidence", "weight", "source",
        "delta_v", "delta_a",
        "ensemble_v", "ensemble_a", "ensemble_std_v", "ensemble_std_a",
        "gemini_v", "gemini_a", "gemini_v_cal", "gemini_a_cal",
        "gemini_confidence", "gemini_parse_ok",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(output_csv, index=False, float_format="%.6f")

    learnable = auto_agreement | gemini_only
    summary = {
        "joined_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "agreement_rate": float(auto_agreement.sum() / len(df)),
        "gemini_only_rate": float(gemini_only.sum() / len(df)),
        "excluded_rate": float(excluded.sum() / len(df)),
        "learnable_rate": float(learnable.sum() / len(df)),
        "mean_weight_learnable": float(df.loc[learnable, "weight"].mean()) if learnable.sum() > 0 else 0.0,
        "thresholds": {
            "agreement": agreement_threshold,
            "std": std_threshold,
            "confidence": conf_threshold,
        },
        "weights": {"ensemble": ensemble_weight, "gemini": gemini_weight},
        "strategy": "gemini_primary_knowledge_distillation",
    }
    stats_path = output_csv.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(df)} rows → {output_csv}")
    print(f"[info] sources: {summary['source_counts']}")
    print(f"[info] agreement={summary['agreement_rate']:.2%}, "
          f"gemini_only={summary['gemini_only_rate']:.2%}, "
          f"excluded={summary['excluded_rate']:.2%} | "
          f"learnable={summary['learnable_rate']:.2%} "
          f"(mean weight={summary['mean_weight_learnable']:.2f})")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_csv", type=Path, required=True)
    p.add_argument("--gemini_csv", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--agreement_threshold", type=float, default=0.2)
    p.add_argument("--std_threshold", type=float, default=0.3)
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--ensemble_weight", type=float, default=0.4)
    p.add_argument("--gemini_weight", type=float, default=0.6)
    args = p.parse_args()
    adjudicate(
        args.aggregate_csv, args.gemini_csv, args.output_csv,
        args.agreement_threshold, args.std_threshold, args.conf_threshold,
        args.ensemble_weight, args.gemini_weight,
    )


if __name__ == "__main__":
    main()
