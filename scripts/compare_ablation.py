#!/usr/bin/env python3
"""Pair two LOMO 9-fold reports by test_movie_code and print a comparison.

Primary hypothesis: does a given variant (e.g. AST audio encoder) improve
mean CCC — and specifically arousal CCC — over the PANNs baseline on the same
film split, seed, lr, and epoch budget?

Δ mean CCC > 0.03 is flagged as a practically meaningful improvement.
A paired t-test across the 9 folds reports statistical significance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scipy import stats


METRIC_KEYS = [
    "mean_ccc",
    "ccc_valence",
    "ccc_arousal",
    "mean_mae",
    "mae_valence",
    "mae_arousal",
    "mood_accuracy",
    "mood_f1_macro",
]


def load_report(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def index_by_movie(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fold in report["per_fold"]:
        if fold.get("status") != "ok":
            continue
        out[fold["test_movie_code"]] = fold["best_val"]
    return out


def paired_delta(base: list[float], var: list[float]) -> dict[str, float]:
    diffs = [v - b for b, v in zip(base, var)]
    t_stat, p_val = stats.ttest_rel(var, base)
    mean_delta = sum(diffs) / len(diffs)
    std = (sum((d - mean_delta) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    return {
        "mean_delta": mean_delta,
        "std_delta": std,
        "t": float(t_stat),
        "p": float(p_val),
        "n": len(diffs),
    }


def fmt_delta(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("runs/ccmovies_lomo9/lomo_report.json"),
        help="Baseline lomo_report.json",
    )
    ap.add_argument(
        "--variant",
        type=Path,
        default=Path("runs/ablation_ast_lomo9/lomo_report.json"),
        help="Ablation variant lomo_report.json",
    )
    ap.add_argument(
        "--meaningful_threshold",
        type=float,
        default=0.03,
        help="Δ mean CCC threshold flagged as meaningful improvement",
    )
    ap.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to write the comparison as JSON",
    )
    args = ap.parse_args()

    base = load_report(args.baseline)
    var = load_report(args.variant)

    base_by_movie = index_by_movie(base)
    var_by_movie = index_by_movie(var)

    common = sorted(set(base_by_movie) & set(var_by_movie))
    if not common:
        raise SystemExit("No overlapping test_movie_code between the two reports.")

    base_only = sorted(set(base_by_movie) - set(var_by_movie))
    var_only = sorted(set(var_by_movie) - set(base_by_movie))

    print(f"Baseline : {args.baseline}  (run_id={base.get('run_id')})")
    print(f"Variant  : {args.variant}  (run_id={var.get('run_id')})")
    print(f"Paired folds (by test_movie_code): {len(common)}")
    if base_only:
        print(f"  baseline-only movies: {base_only}")
    if var_only:
        print(f"  variant-only movies : {var_only}")
    print()

    header = f"{'movie':<22} {'ΔmeanCCC':>10} {'ΔCCC_V':>10} {'ΔCCC_A':>10} {'ΔMAE':>10} {'ΔmoodAcc':>10}"
    print(header)
    print("-" * len(header))
    per_fold_rows = []
    for movie in common:
        b = base_by_movie[movie]
        v = var_by_movie[movie]
        row = {
            "movie": movie,
            "delta_mean_ccc": v["mean_ccc"] - b["mean_ccc"],
            "delta_ccc_valence": v["ccc_valence"] - b["ccc_valence"],
            "delta_ccc_arousal": v["ccc_arousal"] - b["ccc_arousal"],
            "delta_mean_mae": v["mean_mae"] - b["mean_mae"],
            "delta_mood_accuracy": v["mood_accuracy"] - b["mood_accuracy"],
        }
        per_fold_rows.append(row)
        print(
            f"{movie:<22} "
            f"{fmt_delta(row['delta_mean_ccc']):>10} "
            f"{fmt_delta(row['delta_ccc_valence']):>10} "
            f"{fmt_delta(row['delta_ccc_arousal']):>10} "
            f"{fmt_delta(row['delta_mean_mae']):>10} "
            f"{fmt_delta(row['delta_mood_accuracy']):>10}"
        )

    print()
    print("Paired t-test (variant − baseline, n=%d):" % len(common))
    summary: dict[str, Any] = {"per_fold": per_fold_rows, "paired": {}}
    for key in METRIC_KEYS:
        b = [base_by_movie[m][key] for m in common]
        v = [var_by_movie[m][key] for m in common]
        pd = paired_delta(b, v)
        summary["paired"][key] = {
            "baseline_mean": sum(b) / len(b),
            "variant_mean": sum(v) / len(v),
            **pd,
        }
        print(
            f"  {key:<18} base={sum(b)/len(b):+.4f}  var={sum(v)/len(v):+.4f}  "
            f"Δ={fmt_delta(pd['mean_delta'])} ± {pd['std_delta']:.4f}  "
            f"t={pd['t']:+.3f}  p={pd['p']:.4f}"
        )

    print()
    delta_cc = summary["paired"]["mean_ccc"]["mean_delta"]
    p_cc = summary["paired"]["mean_ccc"]["p"]
    verdict_lines = []
    if delta_cc > args.meaningful_threshold and p_cc < 0.05:
        verdict_lines.append(
            f"VARIANT WINS: Δ mean CCC = {delta_cc:+.4f} "
            f"(> {args.meaningful_threshold}, p={p_cc:.4f})"
        )
    elif delta_cc > args.meaningful_threshold:
        verdict_lines.append(
            f"variant leads by Δ mean CCC = {delta_cc:+.4f} "
            f"(> {args.meaningful_threshold}) but p={p_cc:.4f} is not significant."
        )
    elif delta_cc < -args.meaningful_threshold:
        verdict_lines.append(
            f"BASELINE WINS: Δ mean CCC = {delta_cc:+.4f} "
            f"(< −{args.meaningful_threshold}, p={p_cc:.4f})"
        )
    else:
        verdict_lines.append(
            f"EQUIVALENT: |Δ mean CCC| = {abs(delta_cc):.4f} ≤ {args.meaningful_threshold} "
            f"(p={p_cc:.4f}); no practically meaningful difference."
        )

    delta_aro = summary["paired"]["ccc_arousal"]["mean_delta"]
    p_aro = summary["paired"]["ccc_arousal"]["p"]
    verdict_lines.append(
        f"Arousal-specific: Δ CCC_A = {delta_aro:+.4f} (p={p_aro:.4f})"
    )
    summary["verdict"] = verdict_lines

    print("Verdict:")
    for line in verdict_lines:
        print(f"  - {line}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote JSON summary to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
