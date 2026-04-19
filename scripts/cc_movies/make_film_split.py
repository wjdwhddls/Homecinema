"""Film-level train/val/test split 생성.

설계:
  - 8편(현재) 코퍼스는 70/15/15 비율이 비현실적 → 실제 **5/1/2** (train/val/test).
  - Stratify 기준: 각 영화의 ensemble (v, a) centroid 4분면.
  - Greedy: test에 우선 사분면 다양성 확보(서로 다른 사분면 2편), 나머지는 train/val 배정.
  - seed 고정 (기본 42).

Usage:
  python scripts/cc_movies/make_film_split.py \\
    --aggregate_csv dataset/autoEQ/CCMovies/labels/layer1_aggregate.csv \\
    --output_json   dataset/autoEQ/CCMovies/splits/film_split.json \\
    [--n_train 5 --n_val 1 --n_test 2 --seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def quadrant(v: float, a: float) -> str:
    if v >= 0 and a >= 0:
        return "HVHA"
    if v >= 0 and a < 0:
        return "HVLA"
    if v < 0 and a >= 0:
        return "LVHA"
    return "LVLA"


def compute_centroids(aggregate_csv: Path) -> dict:
    df = pd.read_csv(aggregate_csv)
    result = {}
    for film, grp in df.groupby("film_id"):
        v = float(grp["ensemble_v"].mean())
        a = float(grp["ensemble_a"].mean())
        result[film] = {
            "v": round(v, 4),
            "a": round(a, 4),
            "quadrant": quadrant(v, a),
            "n_windows": int(len(grp)),
        }
    return result


def make_split(
    centroids: dict,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> dict:
    total = n_train + n_val + n_test
    films = sorted(centroids.keys())
    if len(films) < total:
        raise RuntimeError(
            f"Films ({len(films)}) < requested total ({total}). "
            f"Adjust --n_train/--n_val/--n_test."
        )

    rng = random.Random(seed)

    # test: greedy over quadrants — maximize quadrant diversity
    by_quad: dict[str, list[str]] = {}
    for f in films:
        by_quad.setdefault(centroids[f]["quadrant"], []).append(f)
    # randomize within each quadrant bucket for reproducibility
    for q in by_quad:
        rng.shuffle(by_quad[q])

    test = []
    quads_used = []
    # greedily pick from different quadrants first
    for _ in range(n_test):
        # pick the quadrant with most remaining (or first unused)
        candidates = sorted(
            [q for q in by_quad if by_quad[q] and q not in quads_used],
            key=lambda q: -len(by_quad[q]),
        )
        if not candidates:
            candidates = sorted(
                [q for q in by_quad if by_quad[q]],
                key=lambda q: -len(by_quad[q]),
            )
        if not candidates:
            break
        q = candidates[0]
        test.append(by_quad[q].pop(0))
        quads_used.append(q)

    # remaining films → pool, shuffle, allocate val then train
    pool = [f for fs in by_quad.values() for f in fs]
    rng.shuffle(pool)
    val = pool[:n_val]
    train = pool[n_val:n_val + n_train]

    return {
        "seed": seed,
        "strategy": "4quadrant_film_centroid_greedy",
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "per_film_centroid": centroids,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_csv", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--n_train", type=int, default=5)
    p.add_argument("--n_val", type=int, default=1)
    p.add_argument("--n_test", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    centroids = compute_centroids(args.aggregate_csv)
    print(f"[info] {len(centroids)} films — centroids per quadrant:")
    quad_count: dict[str, int] = {}
    for film, c in centroids.items():
        quad_count[c["quadrant"]] = quad_count.get(c["quadrant"], 0) + 1
        print(f"  {film:25s}  v={c['v']:+.3f}  a={c['a']:+.3f}  ({c['quadrant']})")
    print(f"[info] quadrant distribution: {quad_count}")

    split = make_split(
        centroids, args.n_train, args.n_val, args.n_test, args.seed
    )
    print(f"[info] train({len(split['train'])}): {split['train']}")
    print(f"[info] val  ({len(split['val'])}): {split['val']}")
    print(f"[info] test ({len(split['test'])}): {split['test']}")

    # Verify no film appears in multiple splits
    all_films = split["train"] + split["val"] + split["test"]
    assert len(set(all_films)) == len(all_films), "duplicate film across splits"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(split, indent=2))
    print(f"[done] → {args.output_json}")


if __name__ == "__main__":
    main()
