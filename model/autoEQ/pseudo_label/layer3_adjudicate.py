"""Layer 3 — agreement/disagreement/excluded 분기.

입력: layer1_aggregate.csv + layer2_gemini.csv (inner join).

규칙:
  excluded    — ensemble_std_v>0.3 OR ensemble_std_a>0.3
                OR gemini_confidence<0.5 OR parse_ok==0
  agreement   — not excluded AND |Δv|<0.2 AND |Δa|<0.2
                final = 0.4 * ensemble + 0.6 * gemini
  disagreement — 나머지 (human adjudication 대기)

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

    df["delta_v"] = (df["ensemble_v"] - df["gemini_v"]).abs()
    df["delta_a"] = (df["ensemble_a"] - df["gemini_a"]).abs()

    excluded = (
        (df["ensemble_std_v"] > std_threshold)
        | (df["ensemble_std_a"] > std_threshold)
        | (df["gemini_confidence"] < conf_threshold)
        | (df["gemini_parse_ok"].fillna(0) == 0)
    )
    agreement = (
        (~excluded)
        & (df["delta_v"] < agreement_threshold)
        & (df["delta_a"] < agreement_threshold)
    )
    disagreement = (~excluded) & (~agreement)

    df["source"] = "excluded"
    df.loc[agreement, "source"] = "auto_agreement"
    df.loc[disagreement, "source"] = "disagreement"

    df["final_v"] = np.nan
    df["final_a"] = np.nan
    df["confidence"] = 0.0
    df["weight"] = 0.0

    w_sum = ensemble_weight + gemini_weight
    ew = ensemble_weight / w_sum
    gw = gemini_weight / w_sum

    df.loc[agreement, "final_v"] = ew * df.loc[agreement, "ensemble_v"] + gw * df.loc[agreement, "gemini_v"]
    df.loc[agreement, "final_a"] = ew * df.loc[agreement, "ensemble_a"] + gw * df.loc[agreement, "gemini_a"]
    df.loc[agreement, "confidence"] = df.loc[agreement, "gemini_confidence"]
    df.loc[agreement, "weight"] = 1.0

    out_cols = [
        "film_id", "window_id",
        "final_v", "final_a", "confidence", "weight", "source",
        "delta_v", "delta_a",
        "ensemble_v", "ensemble_a", "ensemble_std_v", "ensemble_std_a",
        "gemini_v", "gemini_a", "gemini_confidence", "gemini_parse_ok",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(output_csv, index=False, float_format="%.6f")

    summary = {
        "joined_rows": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "agreement_rate": float(agreement.sum() / len(df)),
        "excluded_rate": float(excluded.sum() / len(df)),
        "disagreement_rate": float(disagreement.sum() / len(df)),
        "thresholds": {
            "agreement": agreement_threshold,
            "std": std_threshold,
            "confidence": conf_threshold,
        },
        "weights": {"ensemble": ensemble_weight, "gemini": gemini_weight},
    }
    stats_path = output_csv.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(df)} rows → {output_csv}")
    print(f"[info] sources: {summary['source_counts']}")
    print(f"[info] agreement={summary['agreement_rate']:.2%}, "
          f"disagreement={summary['disagreement_rate']:.2%}, "
          f"excluded={summary['excluded_rate']:.2%}")
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
