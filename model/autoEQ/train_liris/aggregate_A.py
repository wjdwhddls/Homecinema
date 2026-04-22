"""Aggregate Phase 2a-A sweep results + paired comparison vs V2 baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs" / "phase2a"
SEEDS = [42, 123, 2024]
CONFIGS = {
    "V2": {"head_dropout": 0.3, "weight_decay": 1e-4, "dir": "baseline_v2_s{seed}"},
    "a1": {"head_dropout": 0.5, "weight_decay": 1e-4, "dir": "A_a1_s{seed}"},
    "a2": {"head_dropout": 0.3, "weight_decay": 1e-3, "dir": "A_a2_s{seed}"},
    "a3": {"head_dropout": 0.5, "weight_decay": 1e-3, "dir": "A_a3_s{seed}"},
}
KEYS = ["mean_ccc", "ccc_v", "ccc_a", "mean_pearson", "mean_mae"]


def load_summary(cfg_name: str, seed: int) -> dict:
    dir_tpl = CONFIGS[cfg_name]["dir"]
    path = RUNS / dir_tpl.format(seed=seed) / "summary.json"
    data = json.loads(path.read_text())
    # V2 summary has a different shape (aggregated): look at per-seed if possible
    if "best_val" in data:
        return data["best_val"]
    raise ValueError(f"unexpected summary shape: {path}")


def load_v2_per_seed(seed: int) -> dict:
    """V2 was summarized in one aggregate JSON; also each run has its own summary.json."""
    # Check per-run summary first
    per = RUNS / f"baseline_v2_s{seed}" / "summary.json"
    return json.loads(per.read_text())["best_val"]


def load_history_best_epoch(cfg_name: str, seed: int) -> int:
    """Find the epoch with max val mean_CCC."""
    dir_tpl = CONFIGS[cfg_name]["dir"]
    hist = json.loads((RUNS / dir_tpl.format(seed=seed) / "history.json").read_text())
    best_ep = max(range(len(hist)), key=lambda i: hist[i]["val"]["mean_ccc"])
    return best_ep


def load_history_stats(cfg_name: str, seed: int) -> dict:
    """Overfit diagnostics: max gap, ep of best val, train CCC at best ep."""
    dir_tpl = CONFIGS[cfg_name]["dir"]
    hist = json.loads((RUNS / dir_tpl.format(seed=seed) / "history.json").read_text())
    best_i = max(range(len(hist)), key=lambda i: hist[i]["val"]["mean_ccc"])
    return {
        "best_ep": best_i,
        "best_val_ccc": hist[best_i]["val"]["mean_ccc"],
        "train_at_best": hist[best_i]["train"]["mean_ccc"],
        "gap_at_best": hist[best_i]["overfit_gap"],
        "final_gap": hist[-1]["overfit_gap"],
        "total_epochs": len(hist),
    }


def paired_ttest(x: list[float], y: list[float]) -> dict:
    """Simple paired t-test (n=3). Returns t-statistic and 2-tailed p via crude table.
    For n=3, df=2, critical values: |t|>2.920 ≈ p<0.10, |t|>4.303 ≈ p<0.05."""
    diffs = [a - b for a, b in zip(x, y)]
    d_mean = mean(diffs)
    if len(diffs) < 2 or all(d == d_mean for d in diffs):
        return {"t": 0.0, "diffs": diffs, "mean_diff": d_mean, "note": "no variance"}
    d_std = stdev(diffs)
    n = len(diffs)
    t = d_mean / (d_std / math.sqrt(n))
    return {
        "t": round(t, 3),
        "mean_diff": round(d_mean, 4),
        "std_diff": round(d_std, 4),
        "diffs": [round(x, 4) for x in diffs],
        "rough_p_note": "|t|>2.92→p<0.10, |t|>4.30→p<0.05 (df=2)",
    }


def main():
    results: dict[str, dict] = {}
    for cfg_name in CONFIGS:
        per_seed = []
        hist_stats = []
        for s in SEEDS:
            m = load_v2_per_seed(s) if cfg_name == "V2" else load_summary(cfg_name, s)
            per_seed.append({"seed": s, **{k: m[k] for k in KEYS}})
            hist_stats.append({"seed": s, **load_history_stats(cfg_name, s)})
        agg = {}
        for k in KEYS:
            vals = [p[k] for p in per_seed]
            agg[k] = {
                "mean": round(mean(vals), 4),
                "std": round(stdev(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }
        results[cfg_name] = {
            "hyper": {
                "head_dropout": CONFIGS[cfg_name]["head_dropout"],
                "weight_decay": CONFIGS[cfg_name]["weight_decay"],
            },
            "per_seed": per_seed,
            "aggregate": agg,
            "history": hist_stats,
        }

    # Paired comparison: each candidate vs V2 baseline
    v2_mean_ccc = [p["mean_ccc"] for p in results["V2"]["per_seed"]]
    for cand in ["a1", "a2", "a3"]:
        cand_mean_ccc = [p["mean_ccc"] for p in results[cand]["per_seed"]]
        results[cand]["paired_vs_V2"] = paired_ttest(cand_mean_ccc, v2_mean_ccc)

    out = RUNS / "A_sweep_summary.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {out}")

    # Pretty print
    print("\n=== A sweep aggregate (3 seeds each) ===")
    header = f"{'config':<8}  {'head':>4}  {'wd':>5}  {'mean_ccc':>17}  {'ccc_v':>17}  {'ccc_a':>17}  {'best_ep':>8}  {'gap@best':>9}  {'final_gap':>9}"
    print(header)
    print("-" * len(header))
    for cfg_name, r in results.items():
        mc = r["aggregate"]["mean_ccc"]
        cv = r["aggregate"]["ccc_v"]
        ca = r["aggregate"]["ccc_a"]
        best_eps = [h["best_ep"] for h in r["history"]]
        gaps_at_best = [h["gap_at_best"] for h in r["history"]]
        final_gaps = [h["final_gap"] for h in r["history"]]
        print(
            f"{cfg_name:<8}  "
            f"{r['hyper']['head_dropout']:>4}  "
            f"{r['hyper']['weight_decay']:>5}  "
            f"{mc['mean']:.4f}±{mc['std']:.4f}  "
            f"{cv['mean']:.4f}±{cv['std']:.4f}  "
            f"{ca['mean']:.4f}±{ca['std']:.4f}  "
            f"{mean(best_eps):>8.1f}  "
            f"{mean(gaps_at_best):>9.3f}  "
            f"{mean(final_gaps):>9.3f}"
        )

    print("\n=== Paired t-test vs V2 baseline (mean_ccc, n=3) ===")
    for cand in ["a1", "a2", "a3"]:
        p = results[cand]["paired_vs_V2"]
        print(f"{cand}: Δmean={p['mean_diff']:+.4f}  diffs={p['diffs']}  t={p.get('t', 'NA')}")


if __name__ == "__main__":
    main()
