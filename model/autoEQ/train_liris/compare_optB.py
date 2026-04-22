"""Compare original spec_baseline_B vs Option-B-enhanced arch."""

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


def paired_t(x: list[float], y: list[float]) -> dict:
    d = [a - b for a, b in zip(x, y)]
    if all(abs(v - d[0]) < 1e-9 for v in d):
        return {"mean_diff": mean(d), "t": None}
    t = mean(d) / (stdev(d) / math.sqrt(len(d)))
    return {"mean_diff": round(mean(d), 4), "std_diff": round(stdev(d), 4),
            "t": round(t, 3), "diffs": [round(v, 4) for v in d]}


def summary(tag: str) -> dict:
    rows = [load_best(tag, s) for s in SEEDS]
    hists = [load_hist(tag, s) for s in SEEDS]
    best_eps = [max(range(len(h)), key=lambda i: h[i]["val"]["mean_ccc"]) for h in hists]
    gaps_at_best = [h[be]["overfit_gap"] for h, be in zip(hists, best_eps)]
    trains_at_best = [h[be]["train"]["mean_ccc"] for h, be in zip(hists, best_eps)]
    onsets = []
    for h in hists:
        o = next((i for i, r in enumerate(h) if r["overfit_gap"] > 0.10), None)
        onsets.append(o if o is not None else -1)

    def ms(vals):
        return (mean(vals), stdev(vals) if len(vals) > 1 else 0.0)

    return {
        "tag": tag,
        "mean_ccc": ms([r["mean_ccc"] for r in rows]),
        "ccc_v": ms([r["ccc_v"] for r in rows]),
        "ccc_a": ms([r["ccc_a"] for r in rows]),
        "best_ep": ms(best_eps),
        "gap_at_best": ms(gaps_at_best),
        "train_at_best": ms(trains_at_best),
        "overfit_onset": onsets,
        "total_ep": ms([len(h) for h in hists]),
        "per_seed_ccc": [r["mean_ccc"] for r in rows],
    }


def fmt(x):
    m, s = x
    return f"{m:.4f} ± {s:.4f}"


def main():
    groups = [
        ("spec_baseline", "A: official split + spec arch"),
        ("spec_baseline_B", "B-split: full learning + spec arch"),
        ("spec_baseline_optB", "B-split + Enhanced arch (Option B)"),
    ]

    results = {tag: summary(tag) for tag, _ in groups}

    print(f"\n{'='*80}")
    print(f"{'config':<40}  {'val mean_CCC':>18}  {'ccc_v':>16}  {'ccc_a':>16}")
    print(f"{'='*80}")
    for tag, label in groups:
        r = results[tag]
        print(f"{label:<40}  {fmt(r['mean_ccc']):>18}  {fmt(r['ccc_v']):>16}  {fmt(r['ccc_a']):>16}")

    print(f"\n{'='*80}")
    print(f"Training dynamics")
    print(f"{'='*80}")
    print(f"{'config':<40}  {'best_ep':>10}  {'train@best':>12}  {'gap@best':>12}  {'onset':>12}")
    for tag, label in groups:
        r = results[tag]
        print(f"{label:<40}  {fmt(r['best_ep']):>10}  {fmt(r['train_at_best']):>12}  {fmt(r['gap_at_best']):>12}  {str(r['overfit_onset']):>12}")

    # paired t-test optB vs spec_baseline_B
    a = results["spec_baseline_B"]["per_seed_ccc"]
    b = results["spec_baseline_optB"]["per_seed_ccc"]
    t = paired_t(b, a)
    print(f"\npaired t-test Enhanced vs Original (n=3):  {t}")
    print(f"  df=2 critical: |t|>2.92 → p<0.10, |t|>4.30 → p<0.05")

    # Also vs spec_baseline (A)
    a2 = results["spec_baseline"]["per_seed_ccc"]
    t2 = paired_t(b, a2)
    print(f"\npaired t-test Enhanced vs A official-split (n=3):  {t2}")

    (RUNS / "optB_comparison.json").write_text(json.dumps({
        "results": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                        for kk, vv in v.items()} for k, v in results.items()},
        "paired_t_optB_vs_B": t,
        "paired_t_optB_vs_A": t2,
    }, indent=2, default=str))
    print(f"\n[saved] {RUNS/'optB_comparison.json'}")


if __name__ == "__main__":
    main()
