"""7-fold LOMO orchestrator.

Runs `run_train.main()` seven times, one per movie as test, and aggregates
the best metrics from each fold into lomo_report.{json,md}.

Execution rules:
  - Sequential (GPU memory).
  - Per-fold seed = base_seed + fold_idx.
  - A fold failure (exception) is captured in logs/fold_<k>_error.log but
    does NOT abort the remaining folds.
  - Final report: mean ± std of (mean_ccc, mean_mae, mean_rmse,
    mood_f1_macro, mood_kappa) across successful folds + per-fold table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

import torch

from . import run_train
from .dataset import COGNIMUSE_MOVIES


_REPORT_METRICS = (
    "mean_ccc",
    "ccc_valence",
    "ccc_arousal",
    "mean_mae",
    "mae_valence",
    "mae_arousal",
    "mean_rmse",
    "rmse_valence",
    "rmse_arousal",
    "mood_f1_macro",
    "mood_kappa",
    "mood_accuracy",
)


def _extract_best_val_metrics(history: list[dict]) -> dict[str, float]:
    """Pick the epoch with the highest (mean_ccc, -mean_mae) tuple from history."""
    best = None
    best_key = (-float("inf"), -float("inf"))
    for rec in history:
        val = rec.get("val", {})
        key = (float(val.get("mean_ccc", 0.0)), -float(val.get("mean_mae", 1.0)))
        if key > best_key:
            best_key = key
            best = val
    out: dict[str, float] = {}
    if best is None:
        return out
    for k in _REPORT_METRICS:
        v = best.get(k)
        if v is not None and not isinstance(v, dict):
            out[k] = float(v)
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CogniMuse 7-fold LOMO orchestrator")
    p.add_argument("--feature_dir", type=str, required=True)
    p.add_argument("--split_name", type=str, default="cognimuse")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--val_tail_ratio", type=float, default=None)
    p.add_argument("--val_gap_windows", type=int, default=None)
    p.add_argument("--sigma_filter_threshold", type=float, default=None)
    p.add_argument("--modality_dropout_p", type=float, default=None)
    p.add_argument("--num_mood_classes", type=int, default=None, choices=[4, 7])
    p.add_argument("--lambda_mood", type=float, default=None)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--output_dir", type=str, default="runs/cog_lomo")
    p.add_argument("--folds", type=str, default="0,1,2,3,4,5,6",
                   help="comma-separated fold indices to run")
    return p


def _compose_fold_argv(args: argparse.Namespace, fold: int, fold_output: Path) -> list[str]:
    argv = [
        "--feature_dir", args.feature_dir,
        "--split_name", args.split_name,
        "--lomo_fold", str(fold),
        "--epochs", str(args.epochs),
        "--seed", str(args.base_seed + fold),
        "--device", args.device,
        "--output_dir", str(fold_output),
    ]
    if args.batch_size is not None:
        argv += ["--batch_size", str(args.batch_size)]
    if args.lr is not None:
        argv += ["--lr", str(args.lr)]
    if args.val_tail_ratio is not None:
        argv += ["--val_tail_ratio", str(args.val_tail_ratio)]
    if args.val_gap_windows is not None:
        argv += ["--val_gap_windows", str(args.val_gap_windows)]
    if args.sigma_filter_threshold is not None:
        argv += ["--sigma_filter_threshold", str(args.sigma_filter_threshold)]
    if args.modality_dropout_p is not None:
        argv += ["--modality_dropout_p", str(args.modality_dropout_p)]
    if args.num_mood_classes is not None:
        argv += ["--num_mood_classes", str(args.num_mood_classes)]
    if args.lambda_mood is not None:
        argv += ["--lambda_mood", str(args.lambda_mood)]
    if args.use_wandb:
        argv.append("--use_wandb")
    if args.wandb_project:
        argv += ["--wandb_project", args.wandb_project]
    return argv


def _evaluate_pass(per_fold: list[dict]) -> dict:
    """Apply pass/fail gates (V/A Primary + Safety) from PLAN.md §합격 기준."""
    ccc_values = [f["best_val"].get("mean_ccc", 0.0) for f in per_fold if f["status"] == "ok"]
    mae_values = [f["best_val"].get("mean_mae", 1.0) for f in per_fold if f["status"] == "ok"]

    if not ccc_values:
        return {"primary_ccc_pass": False, "primary_mae_pass": False,
                "safety_ccc_pass": False, "safety_mae_pass": False}

    mean_ccc = statistics.fmean(ccc_values)
    mean_mae = statistics.fmean(mae_values)

    # Primary thresholds from PLAN
    primary_ccc_pass = mean_ccc >= 0.20
    primary_mae_pass = mean_mae <= 0.25

    # Safety: ≥ 6/7 folds satisfy condition
    def _per_fold(key_cond_fn):
        return sum(1 for f in per_fold if f["status"] == "ok" and key_cond_fn(f["best_val"]))

    safety_ccc_ok = _per_fold(
        lambda v: min(v.get("ccc_valence", 0.0), v.get("ccc_arousal", 0.0)) > 0.0
    )
    safety_mae_ok = _per_fold(
        lambda v: max(v.get("mae_valence", 1.0), v.get("mae_arousal", 1.0)) <= 0.30
    )
    safety_ccc_pass = safety_ccc_ok >= 6
    safety_mae_pass = safety_mae_ok >= 6

    return {
        "primary_ccc_pass": primary_ccc_pass,
        "primary_mae_pass": primary_mae_pass,
        "safety_ccc_pass": safety_ccc_pass,
        "safety_mae_pass": safety_mae_pass,
        "overall_pass": (
            primary_ccc_pass and primary_mae_pass and safety_ccc_pass and safety_mae_pass
        ),
        "mean_ccc_across_folds": mean_ccc,
        "mean_mae_across_folds": mean_mae,
        "folds_passing_ccc_safety": safety_ccc_ok,
        "folds_passing_mae_safety": safety_mae_ok,
    }


def _write_markdown_report(report: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# LOMO 7-fold Report\n")
    lines.append(f"- Run: {report['run_id']}")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Epochs (per fold): {report['epochs']}")
    lines.append(f"- Num mood classes: {report['num_mood_classes']}\n")

    lines.append("## Per-fold best val metrics\n")
    header = "| fold | movie | ccc_v | ccc_a | mean_ccc | mae_v | mae_a | mean_mae | rmse_v | rmse_a | mean_rmse | mood_f1_macro | mood_kappa | status |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for f in report["per_fold"]:
        bv = f.get("best_val", {})
        status = f["status"]
        lines.append(
            "| {fold} | {movie} | {ccc_v:.3f} | {ccc_a:.3f} | {m_ccc:.3f} | "
            "{mae_v:.3f} | {mae_a:.3f} | {m_mae:.3f} | {rmse_v:.3f} | {rmse_a:.3f} | "
            "{m_rmse:.3f} | {f1:.3f} | {kap:.3f} | {st} |".format(
                fold=f["fold"], movie=f["test_movie_code"],
                ccc_v=bv.get("ccc_valence", 0.0),
                ccc_a=bv.get("ccc_arousal", 0.0),
                m_ccc=bv.get("mean_ccc", 0.0),
                mae_v=bv.get("mae_valence", 0.0),
                mae_a=bv.get("mae_arousal", 0.0),
                m_mae=bv.get("mean_mae", 0.0),
                rmse_v=bv.get("rmse_valence", 0.0),
                rmse_a=bv.get("rmse_arousal", 0.0),
                m_rmse=bv.get("mean_rmse", 0.0),
                f1=bv.get("mood_f1_macro", 0.0),
                kap=bv.get("mood_kappa", 0.0),
                st=status,
            )
        )

    lines.append("\n## Aggregates (successful folds)\n")
    agg = report["aggregates"]
    for k, (mean, std) in agg.items():
        lines.append(f"- **{k}**: {mean:.4f} ± {std:.4f}")

    gate = report["gate"]
    lines.append("\n## Gate (PLAN.md §합격 기준)\n")
    lines.append(f"- V/A Primary (CCC ≥ 0.20): **{'PASS' if gate['primary_ccc_pass'] else 'FAIL'}** "
                 f"(mean_ccc = {gate['mean_ccc_across_folds']:.4f})")
    lines.append(f"- V/A Primary (MAE ≤ 0.25): **{'PASS' if gate['primary_mae_pass'] else 'FAIL'}** "
                 f"(mean_mae = {gate['mean_mae_across_folds']:.4f})")
    lines.append(f"- V/A Safety (≥6/7 folds min(ccc_v,ccc_a) > 0): **"
                 f"{'PASS' if gate['safety_ccc_pass'] else 'FAIL'}** "
                 f"({gate['folds_passing_ccc_safety']}/7)")
    lines.append(f"- V/A Safety (≥6/7 folds max(mae_v,mae_a) ≤ 0.30): **"
                 f"{'PASS' if gate['safety_mae_pass'] else 'FAIL'}** "
                 f"({gate['folds_passing_mae_safety']}/7)")
    lines.append(f"- **Overall**: **{'PASS' if gate['overall_pass'] else 'FAIL'}**")

    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    run_id = root.name
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    per_fold: list[dict] = []
    for k in folds:
        fold_output = root / f"fold_{k}_{COGNIMUSE_MOVIES[k]}"
        fold_argv = _compose_fold_argv(args, k, fold_output)
        fold_record: dict = {
            "fold": k,
            "test_movie_code": COGNIMUSE_MOVIES[k],
            "output_dir": str(fold_output),
        }
        try:
            result = run_train.main(fold_argv)
            fold_record["status"] = "ok"
            fold_record["best_mean_ccc"] = result["best_mean_ccc"]
            fold_record["best_mean_mae"] = result["best_mean_mae"]
            history_path = fold_output / "history.json"
            if history_path.is_file():
                with open(history_path) as f:
                    history = json.load(f)
                fold_record["best_val"] = _extract_best_val_metrics(history)
            else:
                fold_record["best_val"] = {}
        except Exception as e:
            fold_record["status"] = "error"
            fold_record["error"] = str(e)
            (logs_dir / f"fold_{k}_error.log").write_text(
                traceback.format_exc()
            )
            fold_record["best_val"] = {}
        per_fold.append(fold_record)
        # free memory between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregates
    aggregates: dict[str, tuple[float, float]] = {}
    for key in _REPORT_METRICS:
        values = [f["best_val"].get(key) for f in per_fold if f["status"] == "ok"]
        values = [v for v in values if isinstance(v, (int, float))]
        aggregates[key] = _mean_std(values)

    gate = _evaluate_pass(per_fold)

    report = {
        "run_id": run_id,
        "timestamp": timestamp,
        "epochs": args.epochs,
        "num_mood_classes": args.num_mood_classes,
        "per_fold": per_fold,
        "aggregates": {k: {"mean": m, "std": s} for k, (m, s) in aggregates.items()},
        "gate": gate,
    }

    with open(root / "lomo_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Markdown report needs tuple-form aggregates
    md_report = dict(report)
    md_report["aggregates"] = aggregates
    _write_markdown_report(md_report, root / "lomo_report.md")

    return report


if __name__ == "__main__":
    report = main()
    print(json.dumps(
        {
            "run_id": report["run_id"],
            "gate": report["gate"],
        },
        indent=2,
    ))
