"""Full evaluation metrics for the FROZEN BASE MODEL (2026-04-21).

Computes CCC, Pearson, MAE × {mean, valence, arousal} across 3 seeds
from spec_baseline_optB_s{42,123,2024}/best.pt checkpoints.

Output: runs/phase2a/base_model_final_metrics.json + console table.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

RUNS = Path(__file__).resolve().parents[3] / "runs" / "phase2a"
SEEDS = [42, 123, 2024]
# BASE 2026-04-21 (Phase 2a-2 winner): A + K=7
PREFIX = "2a2_A_K7"


def load_best(seed: int) -> dict:
    return json.loads((RUNS / f"{PREFIX}_s{seed}" / "summary.json").read_text())["best_val"]


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"


def main():
    rows = [load_best(s) for s in SEEDS]

    # Extract 9 metrics × 3 seeds
    keys = {
        # CCC
        "mean_ccc":       ("CCC (mean)",     "ccc"),
        "ccc_v":          ("CCC — Valence",  "ccc"),
        "ccc_a":          ("CCC — Arousal",  "ccc"),
        # Pearson
        "mean_pearson":   ("Pearson (mean)", "pearson"),
        "pearson_valence":("Pearson — V",    "pearson"),
        "pearson_arousal":("Pearson — A",    "pearson"),
        # MAE
        "mean_mae":       ("MAE (mean)",     "mae"),
        "mae_valence":    ("MAE — Valence",  "mae"),
        "mae_arousal":    ("MAE — Arousal",  "mae"),
    }

    out = {"per_seed": {s: r for s, r in zip(SEEDS, rows)}, "metrics": {}}

    print("\n" + "=" * 78)
    print(" Base Model Final Evaluation — 3-seed (42 / 123 / 2024)")
    print("         FROZEN 2026-04-21 · LIRIS val (16 films, 585 clips)")
    print("=" * 78)

    for group_label, filter_type in [("CCC", "ccc"), ("Pearson", "pearson"), ("MAE", "mae")]:
        print(f"\n─── {group_label} ───")
        print(f"  {'metric':<22} {'seed42':>8} {'seed123':>8} {'seed2024':>8}   {'mean ± std':>20}")
        for k, (label, typ) in keys.items():
            if typ != filter_type:
                continue
            vals = [r[k] for r in rows]
            m, s = mean(vals), stdev(vals)
            out["metrics"][k] = {"label": label, "mean": m, "std": s, "per_seed": vals}
            print(f"  {label:<22} "
                  f"{vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}   "
                  f"{fmt(m, s):>20}")

    # Summary markdown-style
    print("\n" + "=" * 78)
    print(" Markdown summary (for thesis / report)")
    print("=" * 78)
    print()
    print("| Metric | mean | Valence (V) | Arousal (A) |")
    print("|--------|:-:|:-:|:-:|")
    for group_label in ["CCC", "Pearson", "MAE"]:
        if group_label == "CCC":
            mk, vk, ak = "mean_ccc", "ccc_v", "ccc_a"
        elif group_label == "Pearson":
            mk, vk, ak = "mean_pearson", "pearson_valence", "pearson_arousal"
        else:
            mk, vk, ak = "mean_mae", "mae_valence", "mae_arousal"
        line = (
            f"| **{group_label}** | "
            f"{mean([r[mk] for r in rows]):.4f} ± {stdev([r[mk] for r in rows]):.4f} | "
            f"{mean([r[vk] for r in rows]):.4f} ± {stdev([r[vk] for r in rows]):.4f} | "
            f"{mean([r[ak] for r in rows]):.4f} ± {stdev([r[ak] for r in rows]):.4f} |"
        )
        print(line)

    (RUNS / "base_model_final_metrics.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(f"\n[saved] {RUNS / 'base_model_final_metrics.json'}")


if __name__ == "__main__":
    main()
