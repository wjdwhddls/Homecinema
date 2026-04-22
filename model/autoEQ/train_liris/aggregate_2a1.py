"""Aggregate Phase 2a-1: V/A normalization A vs B × 3-seed."""

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
    a = [load_best("2a1_A", s) for s in SEEDS]
    b = [load_best("2a1_B", s) for s in SEEDS]

    print(f"\n{'='*70}")
    print(" Phase 2a-1 — V/A Normalization Strategy A vs B")
    print(f"{'='*70}\n")

    print(f"{'strategy':<15} {'seed':>5} {'mean_CCC':>10} {'ccc_v':>8} {'ccc_a':>8} "
          f"{'mean_P':>8} {'mean_MAE':>10}")
    print("-"*80)
    for tag, rows in [("A (spec)", a), ("B (min-max)", b)]:
        for s, r in zip(SEEDS, rows):
            print(f"{tag:<15} {s:>5} {r['mean_ccc']:>10.4f} {r['ccc_v']:>8.4f} "
                  f"{r['ccc_a']:>8.4f} {r['mean_pearson']:>8.4f} {r['mean_mae']:>10.4f}")

    print()
    print("=== Aggregate (mean ± std, 3-seed) ===")
    print(f"{'strategy':<15} {'mean_CCC':>17} {'ccc_v':>17} {'ccc_a':>17} {'mean_P':>17}")
    for tag, rows in [("A (spec)", a), ("B (min-max)", b)]:
        mc = ms([r["mean_ccc"] for r in rows])
        cv = ms([r["ccc_v"] for r in rows])
        ca = ms([r["ccc_a"] for r in rows])
        mp = ms([r["mean_pearson"] for r in rows])
        print(f"{tag:<15} {mc[0]:.4f} ± {mc[1]:.4f}  {cv[0]:.4f} ± {cv[1]:.4f}  "
              f"{ca[0]:.4f} ± {ca[1]:.4f}  {mp[0]:.4f} ± {mp[1]:.4f}")

    # paired t-test
    a_cccs = [r["mean_ccc"] for r in a]
    b_cccs = [r["mean_ccc"] for r in b]
    t = paired_t(b_cccs, a_cccs)
    print(f"\npaired t-test B vs A (n=3):  {t}")
    print("  df=2 critical: |t|>2.92 → p<0.10, |t|>4.30 → p<0.05")

    # Winner determination
    winner = "A" if mean(a_cccs) > mean(b_cccs) else "B"
    delta = mean(b_cccs) - mean(a_cccs)
    print(f"\n=== Winner by val CCC: {winner}  (Δ = {delta:+.4f}) ===")

    # §8 K=7 gate consideration
    print("\n=== §8 K=7 Gate Status ===")
    print("  Strategy A: JA count = 0 / 9800 → gate FAIL (K=4 강제)")
    print("  Strategy B: JA count = 649 / 9800 (6.62%) → gate PASS (K=7 viable)")

    print("\n=== 의사결정 경로 (V5-FINAL §8 + §21) ===")
    if winner == "A":
        print("  Strict interpretation: A 승 → K=4 확정 → 2a-2는 K=7 학술 참고만")
        print("  Pragmatic interpretation: B는 -0.014 미미한 차이 + K=7 viable → B baseline 재고 가치")
        print("  → 2a-2 방안: B 기준선으로 K=4 vs K=7 의미있는 비교")
    else:
        print("  B 승 → §8 gate 자동 PASS → 2a-2에서 K=7 본격 비교")

    (RUNS / "2a1_summary.json").write_text(json.dumps({
        "strategy_A_per_seed": a,
        "strategy_B_per_seed": b,
        "paired_t": t,
        "winner_by_ccc": winner,
        "delta": delta,
        "k7_gate_A_PASS": False,
        "k7_gate_B_PASS": True,
        "k7_JA_count_A": 0,
        "k7_JA_count_B": 649,
    }, indent=2, default=str))
    print(f"\n[saved] {RUNS/'2a1_summary.json'}")


if __name__ == "__main__":
    main()
