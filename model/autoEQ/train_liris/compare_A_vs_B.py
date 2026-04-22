"""Compare official (A) vs full-learning (B) splits."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

RUNS = Path(__file__).resolve().parents[3] / "runs" / "phase2a"
SEEDS = [42, 123, 2024]


def load_best(tag: str, s: int) -> dict:
    return json.loads((RUNS / f"{tag}_s{s}" / "summary.json").read_text())["best_val"]


def load_hist(tag: str, s: int) -> list:
    return json.loads((RUNS / f"{tag}_s{s}" / "history.json").read_text())


def main():
    a_rows = [load_best("spec_baseline", s) for s in SEEDS]
    b_rows = [load_best("spec_baseline_B", s) for s in SEEDS]

    print(f"{'tag':<4} {'seed':>5} {'mean_CCC':>9} {'ccc_v':>7} {'ccc_a':>7} {'best_ep':>8} {'total_ep':>8}")
    for tag, rows, prefix in [("A", a_rows, "spec_baseline"), ("B", b_rows, "spec_baseline_B")]:
        for s, r in zip(SEEDS, rows):
            h = load_hist(prefix, s)
            be = max(range(len(h)), key=lambda i: h[i]["val"]["mean_ccc"])
            print(f"{tag:<4} {s:>5} {r['mean_ccc']:>9.4f} {r['ccc_v']:>7.3f} {r['ccc_a']:>7.3f} {be:>8} {len(h):>8}")

    def stats(vals):
        return f"{mean(vals):.4f} ± {stdev(vals):.4f}"

    for tag, rows in [("A (40/40/80)", a_rows), ("B (64/16/80)", b_rows)]:
        print(f"\n{tag}:")
        for k in ["mean_ccc", "ccc_v", "ccc_a"]:
            print(f"  {k:15s}: {stats([r[k] for r in rows])}")

    a_cccs = [r["mean_ccc"] for r in a_rows]
    b_cccs = [r["mean_ccc"] for r in b_rows]
    d = [b - a for a, b in zip(a_cccs, b_cccs)]
    t = mean(d) / (stdev(d) / math.sqrt(3))
    print(f"\npaired t-test B vs A:  mean_diff={mean(d):+.4f}  std={stdev(d):.4f}  t={t:.3f}")
    print(f"diffs per seed: {[round(x, 4) for x in d]}")
    print(f"df=2 critical: |t|>2.92 → p<0.10, |t|>4.30 → p<0.05")

    # overfit summary
    def gap_trajectory(prefix):
        ks = []
        for s in SEEDS:
            h = load_hist(prefix, s)
            best_i = max(range(len(h)), key=lambda i: h[i]["val"]["mean_ccc"])
            onset = next((i for i, r in enumerate(h) if r["overfit_gap"] > 0.10), None)
            ks.append({"best_ep": best_i, "best_val_ccc": h[best_i]["val"]["mean_ccc"],
                       "train_at_best": h[best_i]["train"]["mean_ccc"],
                       "gap_at_best": h[best_i]["overfit_gap"],
                       "overfit_onset_ep": onset,
                       "final_gap": h[-1]["overfit_gap"],
                       "final_val": h[-1]["val"]["mean_ccc"]})
        return ks

    print("\n=== Overfit dynamics ===")
    for tag, prefix in [("A", "spec_baseline"), ("B", "spec_baseline_B")]:
        ks = gap_trajectory(prefix)
        print(f"\n{tag}:")
        for s, k in zip(SEEDS, ks):
            print(f"  seed={s}  best_ep={k['best_ep']:<2}  val={k['best_val_ccc']:.4f}  "
                  f"train={k['train_at_best']:.4f}  gap@best={k['gap_at_best']:.3f}  "
                  f"overfit_onset=ep{k['overfit_onset_ep']}  final_gap={k['final_gap']:.3f}  "
                  f"final_val={k['final_val']:.4f}")


if __name__ == "__main__":
    main()
