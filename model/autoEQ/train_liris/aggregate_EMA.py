"""Aggregate EMA sweep results vs V2 baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "phase2a"
SEEDS = [42, 123, 2024]


def load_best(dir_name: str) -> dict:
    return json.loads((RUNS / dir_name / "summary.json").read_text())["best_val"]


def history_stats(dir_name: str) -> dict:
    h = json.loads((RUNS / dir_name / "history.json").read_text())
    best_i = max(range(len(h)), key=lambda i: h[i]["val"]["mean_ccc"])
    return {"best_ep": best_i, "total_ep": len(h), "gap_at_best": h[best_i]["overfit_gap"]}


def paired_t(x: list[float], y: list[float]) -> dict:
    d = [a - b for a, b in zip(x, y)]
    if all(abs(v - d[0]) < 1e-9 for v in d):
        return {"mean_diff": mean(d), "t": float("inf" if d[0] > 0 else "-inf" if d[0] < 0 else 0)}
    t = mean(d) / (stdev(d) / math.sqrt(len(d)))
    return {"mean_diff": round(mean(d), 4), "std_diff": round(stdev(d), 4),
            "t": round(t, 3), "diffs": [round(v, 4) for v in d]}


def main():
    # V2 baseline
    v2 = [{"seed": s, **load_best(f"baseline_v2_s{s}"), **history_stats(f"baseline_v2_s{s}")}
          for s in SEEDS]
    # EMA decay=0.99 3-seed
    ema = [{"seed": s, **load_best(f"EMA_d99_s{s}"), **history_stats(f"EMA_d99_s{s}")}
           for s in SEEDS]

    # decay scan on seed 42
    scan = {}
    for tag, d in [("0.99", "EMA_d99_s42"), ("0.995", "EMA_d995_s42"), ("0.998", "EMA_d998_s42")]:
        scan[tag] = {**load_best(d), **history_stats(d)}

    # Pretty
    print("=== EMA decay scan (seed 42) ===")
    print(f"{'decay':<8} {'best_ep':>8} {'total_ep':>9} {'mean_CCC':>10} {'ccc_v':>8} {'ccc_a':>8}")
    print(f"{'(V2)':<8} {11:>8} {22:>9} {0.3450:>10.4f} {0.3077:>8.4f} {0.3823:>8.4f}")
    for tag, r in scan.items():
        print(f"{tag:<8} {r['best_ep']:>8d} {r['total_ep']:>9d} "
              f"{r['mean_ccc']:>10.4f} {r['ccc_v']:>8.4f} {r['ccc_a']:>8.4f}")

    print("\n=== 3-seed comparison: EMA (decay=0.99) vs V2 ===")
    print(f"{'seed':<6} {'V2 CCC':>10} {'EMA CCC':>10} {'Δ':>10} {'V2 best_ep':>12} {'EMA best_ep':>12}")
    for a, b in zip(v2, ema):
        print(f"{a['seed']:<6d} {a['mean_ccc']:>10.4f} {b['mean_ccc']:>10.4f} "
              f"{b['mean_ccc']-a['mean_ccc']:>+10.4f} {a['best_ep']:>12d} {b['best_ep']:>12d}")

    v2_cccs = [r["mean_ccc"] for r in v2]
    ema_cccs = [r["mean_ccc"] for r in ema]
    print(f"\n{'mean':<6} {mean(v2_cccs):>10.4f} {mean(ema_cccs):>10.4f} {mean(ema_cccs)-mean(v2_cccs):>+10.4f}")
    print(f"{'std':<6}  {stdev(v2_cccs):>9.4f}  {stdev(ema_cccs):>9.4f}")

    t = paired_t(ema_cccs, v2_cccs)
    print(f"\npaired t-test (EMA - V2, n=3): {t}")
    print("  df=2: |t|>2.92 → p<0.10  |t|>4.30 → p<0.05")

    out = {
        "v2_baseline": v2, "ema_d099": ema, "decay_scan_s42": scan,
        "paired_test": t,
    }
    (RUNS / "EMA_sweep_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {RUNS / 'EMA_sweep_summary.json'}")


if __name__ == "__main__":
    main()
