"""Aggregate Phase 2a-2: Mood head K=4 vs K=7 across both normalization strategies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

RUNS = Path(__file__).resolve().parents[3] / "runs" / "phase2a"
SEEDS = [42, 123, 2024]


def load_best(tag: str, s: int) -> dict:
    return json.loads((RUNS / f"{tag}_s{s}" / "summary.json").read_text())["best_val"]


def ms(vals):
    return mean(vals), stdev(vals) if len(vals) > 1 else 0.0


def paired_t(x, y):
    d = [a - b for a, b in zip(x, y)]
    if all(abs(v - d[0]) < 1e-9 for v in d):
        return {"mean_diff": mean(d), "t": None}
    t = mean(d) / (stdev(d) / math.sqrt(len(d)))
    return {"mean_diff": round(mean(d), 4), "std_diff": round(stdev(d), 4),
            "t": round(t, 3), "diffs": [round(v, 4) for v in d]}


def main():
    cells = {
        ("A", 4): [load_best("2a1_A", s) for s in SEEDS],      # reuse from 2a-1
        ("A", 7): [load_best("2a2_A_K7", s) for s in SEEDS],
        ("B", 4): [load_best("2a1_B", s) for s in SEEDS],      # reuse
        ("B", 7): [load_best("2a2_B_K7", s) for s in SEEDS],
    }

    print(f"\n{'='*75}")
    print(" Phase 2a-2 — Mood Head K=4 vs K=7  ×  V/A norm Strategy A vs B")
    print(f"{'='*75}\n")

    # Per-seed matrix
    print(f"{'cell':<10} {'seed':>5} {'mean_CCC':>10} {'ccc_v':>8} {'ccc_a':>8} {'mean_P':>8}")
    print("-"*55)
    for (strat, k), rows in cells.items():
        for s, r in zip(SEEDS, rows):
            print(f"{strat}+K={k}   {s:>5} {r['mean_ccc']:>10.4f} {r['ccc_v']:>8.4f} "
                  f"{r['ccc_a']:>8.4f} {r['mean_pearson']:>8.4f}")

    # Aggregate
    print("\n=== Aggregate (mean ± std, 3-seed) ===")
    print(f"{'cell':<12} {'mean_CCC':>18} {'ccc_v':>17} {'ccc_a':>17} {'mean_P':>17}")
    agg = {}
    for (strat, k), rows in cells.items():
        mc = ms([r["mean_ccc"] for r in rows])
        cv = ms([r["ccc_v"] for r in rows])
        ca = ms([r["ccc_a"] for r in rows])
        mp = ms([r["mean_pearson"] for r in rows])
        agg[(strat, k)] = {"mean_ccc": mc, "ccc_v": cv, "ccc_a": ca, "mean_p": mp}
        label = f"{strat}+K={k}"
        print(f"{label:<12}  {mc[0]:.4f} ± {mc[1]:.4f}  {cv[0]:.4f} ± {cv[1]:.4f}  "
              f"{ca[0]:.4f} ± {ca[1]:.4f}  {mp[0]:.4f} ± {mp[1]:.4f}")

    # Winner
    best_key = max(agg.keys(), key=lambda k: agg[k]["mean_ccc"][0])
    print(f"\n🏆 Winner: {best_key[0]}+K={best_key[1]} "
          f"→ CCC = {agg[best_key]['mean_ccc'][0]:.4f} ± {agg[best_key]['mean_ccc'][1]:.4f}")

    # paired t tests
    print("\n=== Paired t-tests (n=3) ===")
    comparisons = [
        ("A+K=7 vs A+K=4", ("A", 7), ("A", 4)),
        ("B+K=7 vs B+K=4", ("B", 7), ("B", 4)),
        ("A+K=7 vs B+K=7", ("A", 7), ("B", 7)),
        ("A+K=4 vs B+K=4", ("A", 4), ("B", 4)),
        ("Best vs Base (A+K=4)", best_key, ("A", 4)),
    ]
    for label, a_key, b_key in comparisons:
        a_cccs = [r["mean_ccc"] for r in cells[a_key]]
        b_cccs = [r["mean_ccc"] for r in cells[b_key]]
        t = paired_t(a_cccs, b_cccs)
        sig = ""
        if t.get("t") is not None:
            abs_t = abs(t["t"])
            if abs_t > 4.30: sig = "  ⭐ p<0.05"
            elif abs_t > 2.92: sig = "  · p<0.10"
        print(f"  {label:<25}  {t}{sig}")

    # §8 K=7 Gate Status reminder
    print("\n=== §8 K=7 Gate Status ===")
    print("  Strategy A: JA=0/9800 FAIL → K=7 구조적으로 JA 학습 불가 (그러나 실측 CCC는 개선?!)")
    print("  Strategy B: JA=649/9800 PASS → K=7 정상 학습 가능")

    (RUNS / "2a2_summary.json").write_text(json.dumps({
        "cells": {f"{k[0]}_K{k[1]}": [r for r in v] for k, v in cells.items()},
        "aggregate": {f"{k[0]}_K{k[1]}": {kk: list(vv) for kk, vv in v.items()}
                      for k, v in agg.items()},
        "winner": f"{best_key[0]}+K={best_key[1]}",
        "winner_ccc": agg[best_key]["mean_ccc"][0],
    }, indent=2, default=str))
    print(f"\n[saved] {RUNS/'2a2_summary.json'}")


if __name__ == "__main__":
    main()
