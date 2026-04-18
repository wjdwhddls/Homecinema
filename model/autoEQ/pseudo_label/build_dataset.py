"""최종 dataset 구성: split 부여 + test_gold/disagreement queue 추첨 + summary.

입력:
  - layer3_adjudicated.csv (final_v/a, source, etc)
  - film_split.json (train/val/test 영화 리스트)
  - windows/metadata.csv (t0, t1)

출력:
  - final_labels.csv           — 전체 1,109 rows, split 컬럼 포함
  - test_gold_queue.csv        — test films 내부 4분면 stratified ~200 clips (human annotate 대기)
  - disagreement_queue.csv     — train+val films의 source=disagreement 샘플 ~200 clips
  - dataset_summary.json       — split/source/사분면 분포

Usage:
  python -m model.autoEQ.pseudo_label.build_dataset \\
    --adjudicated_csv dataset/autoEQ/CCMovies/labels/layer3_adjudicated.csv \\
    --split_json      dataset/autoEQ/CCMovies/splits/film_split.json \\
    --metadata_csv    dataset/autoEQ/CCMovies/windows/metadata.csv \\
    --output_dir      dataset/autoEQ/CCMovies/labels \\
    [--gold_per_quadrant 50 --disagreement_sample 200 --seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def quadrant(v: float, a: float) -> str:
    if pd.isna(v) or pd.isna(a):
        return "UNK"
    if v >= 0 and a >= 0:
        return "HVHA"
    if v >= 0 and a < 0:
        return "HVLA"
    if v < 0 and a >= 0:
        return "LVHA"
    return "LVLA"


def build(
    adjudicated_csv: Path,
    split_json: Path,
    metadata_csv: Path,
    output_dir: Path,
    gold_per_quadrant: int = 50,
    disagreement_sample: int = 200,
    seed: int = 42,
) -> dict:
    adj = pd.read_csv(adjudicated_csv)
    meta = pd.read_csv(metadata_csv)[["film_id", "window_id", "t0", "t1"]]
    split = json.loads(split_json.read_text())

    train_films = set(split["train"])
    val_films = set(split["val"])
    test_films = set(split["test"])

    df = adj.merge(meta, on=["film_id", "window_id"], how="left")
    df["split"] = "unknown"
    df.loc[df["film_id"].isin(train_films), "split"] = "train"
    df.loc[df["film_id"].isin(val_films), "split"] = "val"
    df.loc[df["film_id"].isin(test_films), "split"] = "test_pseudo"

    assert (df["split"] == "unknown").sum() == 0, \
        f"unknown split for {(df['split']=='unknown').sum()} rows"

    # quadrant (ensemble 기반 — final_v/a는 disagreement에서 NaN이라 적절치 않음)
    df["quadrant"] = [quadrant(v, a) for v, a in zip(df["ensemble_v"], df["ensemble_a"])]

    rng = np.random.default_rng(seed)

    # test gold queue — test films 안에서, excluded 제외, 4분면 stratified
    test_pool = df[df["split"] == "test_pseudo"]
    test_eligible = test_pool[test_pool["source"] != "excluded"]
    gold_rows = []
    for q in ["HVHA", "HVLA", "LVHA", "LVLA"]:
        sub = test_eligible[test_eligible["quadrant"] == q]
        if len(sub) == 0:
            continue
        n = min(gold_per_quadrant, len(sub))
        pick_idx = rng.choice(sub.index, size=n, replace=False)
        gold_rows.extend(pick_idx.tolist())
    # 대신 'test_gold'로 split 재지정
    df.loc[gold_rows, "split"] = "test_gold"

    # disagreement queue — train+val films의 source=disagreement 에서 샘플
    dis_pool = df[
        (df["source"] == "disagreement") & df["split"].isin(["train", "val"])
    ]
    n_dis = min(disagreement_sample, len(dis_pool))
    dis_idx = rng.choice(dis_pool.index, size=n_dis, replace=False) if n_dis > 0 else []

    output_dir.mkdir(parents=True, exist_ok=True)

    # final_labels.csv
    final_cols = [
        "film_id", "window_id", "t0", "t1", "split", "source", "quadrant",
        "final_v", "final_a", "confidence", "weight",
        "ensemble_v", "ensemble_a", "ensemble_std_v", "ensemble_std_a",
        "gemini_v", "gemini_a", "gemini_confidence",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df[final_cols].to_csv(output_dir / "final_labels.csv", index=False, float_format="%.6f")

    # test gold queue CSV — pending human input
    gold_cols = [
        "film_id", "window_id", "t0", "t1", "quadrant",
        "ensemble_v", "ensemble_a", "gemini_v", "gemini_a", "source",
    ]
    gold_cols = [c for c in gold_cols if c in df.columns]
    df.loc[gold_rows, gold_cols].to_csv(
        output_dir / "test_gold_queue.csv", index=False, float_format="%.6f"
    )

    # disagreement queue
    dis_cols = [
        "film_id", "window_id", "t0", "t1", "quadrant",
        "delta_v", "delta_a",
        "ensemble_v", "ensemble_a", "gemini_v", "gemini_a",
    ]
    dis_cols = [c for c in dis_cols if c in df.columns]
    if len(dis_idx):
        df.loc[dis_idx, dis_cols].to_csv(
            output_dir / "disagreement_queue.csv", index=False, float_format="%.6f"
        )
    else:
        pd.DataFrame(columns=dis_cols).to_csv(
            output_dir / "disagreement_queue.csv", index=False
        )

    # summary
    split_counts = df["split"].value_counts().to_dict()
    source_counts = df["source"].value_counts().to_dict()
    quad_counts_by_split = {
        s: df[df["split"] == s]["quadrant"].value_counts().to_dict()
        for s in df["split"].unique()
    }
    summary = {
        "total_rows": int(len(df)),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "quadrants_by_split": quad_counts_by_split,
        "test_gold_queue_size": len(gold_rows),
        "disagreement_queue_size": int(n_dis),
        "seed": seed,
        "config": {
            "gold_per_quadrant": gold_per_quadrant,
            "disagreement_sample": disagreement_sample,
        },
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] final_labels.csv ({len(df)} rows), "
          f"test_gold={len(gold_rows)}, disagreement={n_dis}")
    print(f"[info] split: {split_counts}")
    print(f"[info] source: {source_counts}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adjudicated_csv", type=Path, required=True)
    p.add_argument("--split_json", type=Path, required=True)
    p.add_argument("--metadata_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--gold_per_quadrant", type=int, default=50)
    p.add_argument("--disagreement_sample", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build(
        args.adjudicated_csv, args.split_json, args.metadata_csv,
        args.output_dir, args.gold_per_quadrant, args.disagreement_sample, args.seed,
    )


if __name__ == "__main__":
    main()
