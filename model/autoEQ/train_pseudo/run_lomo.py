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

import math

from . import run_train
from .ccmovies_preprocess import CCMOVIES
from .dataset import COGNIMUSE_MOVIES


MOVIE_SETS = {
    "cognimuse": COGNIMUSE_MOVIES,
    "ccmovies": CCMOVIES,
}


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
    p = argparse.ArgumentParser(description="LOMO orchestrator (CogniMuse 7 or CCMovies 9)")
    p.add_argument("--feature_dir", type=str, required=True)
    p.add_argument("--split_name", type=str, default="cognimuse")
    p.add_argument("--movie_set", choices=list(MOVIE_SETS), default="cognimuse")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--val_tail_ratio", type=float, default=None)
    p.add_argument("--val_gap_windows", type=int, default=None)
    p.add_argument("--sigma_filter_threshold", type=float, default=None)
    p.add_argument("--modality_dropout_p", type=float, default=None)
    p.add_argument("--num_mood_classes", type=int, default=None, choices=[4, 7])
    p.add_argument("--lambda_mood", type=float, default=None)
    # Augmentation
    p.add_argument("--feature_noise_std", type=float, default=None)
    p.add_argument("--mixup_prob", type=float, default=None)
    p.add_argument("--mixup_alpha", type=float, default=None)
    p.add_argument("--label_smooth_eps", type=float, default=None)
    p.add_argument("--label_smooth_sigma_threshold", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--output_dir", type=str, default="runs/lomo_latest")
    p.add_argument("--folds", type=str, default=None,
                   help="comma-separated fold indices (default: all folds of movie_set)")
    return p


def _compose_fold_argv(args: argparse.Namespace, fold: int, fold_output: Path) -> list[str]:
    argv = [
        "--feature_dir", args.feature_dir,
        "--split_name", args.split_name,
        "--movie_set", args.movie_set,
        "--lomo_fold", str(fold),
        "--epochs", str(args.epochs),
        "--seed", str(args.base_seed + fold),
        "--device", args.device,
        "--output_dir", str(fold_output),
    ]
    optional = [
        ("batch_size", "--batch_size"),
        ("lr", "--lr"),
        ("weight_decay", "--weight_decay"),
        ("warmup_steps", "--warmup_steps"),
        ("val_tail_ratio", "--val_tail_ratio"),
        ("val_gap_windows", "--val_gap_windows"),
        ("sigma_filter_threshold", "--sigma_filter_threshold"),
        ("modality_dropout_p", "--modality_dropout_p"),
        ("num_mood_classes", "--num_mood_classes"),
        ("lambda_mood", "--lambda_mood"),
        ("feature_noise_std", "--feature_noise_std"),
        ("mixup_prob", "--mixup_prob"),
        ("mixup_alpha", "--mixup_alpha"),
        ("label_smooth_eps", "--label_smooth_eps"),
        ("label_smooth_sigma_threshold", "--label_smooth_sigma_threshold"),
        ("early_stop_patience", "--early_stop_patience"),
    ]
    for attr, flag in optional:
        v = getattr(args, attr, None)
        if v is not None:
            argv += [flag, str(v)]
    if args.use_wandb:
        argv.append("--use_wandb")
    if args.wandb_project:
        argv += ["--wandb_project", args.wandb_project]
    return argv


def _evaluate_pass(per_fold: list[dict], total_folds: int) -> dict:
    """Apply pass/fail gates (V/A Primary + Safety) from PLAN.md §합격 기준.

    Safety threshold generalizes the original `>= 6/7` to
    `>= ceil(total_folds * 6/7)` so CCMovies 9-fold requires ≥ 8/9, CogniMuse
    7-fold requires ≥ 6/7 (identical).
    """
    ccc_values = [f["best_val"].get("mean_ccc", 0.0) for f in per_fold if f["status"] == "ok"]
    mae_values = [f["best_val"].get("mean_mae", 1.0) for f in per_fold if f["status"] == "ok"]

    safety_threshold = math.ceil(total_folds * 6 / 7)

    if not ccc_values:
        return {"primary_ccc_pass": False, "primary_mae_pass": False,
                "safety_ccc_pass": False, "safety_mae_pass": False,
                "safety_threshold": safety_threshold, "total_folds": total_folds}

    mean_ccc = statistics.fmean(ccc_values)
    mean_mae = statistics.fmean(mae_values)

    primary_ccc_pass = mean_ccc >= 0.20
    primary_mae_pass = mean_mae <= 0.25

    def _per_fold(key_cond_fn):
        return sum(1 for f in per_fold if f["status"] == "ok" and key_cond_fn(f["best_val"]))

    safety_ccc_ok = _per_fold(
        lambda v: min(v.get("ccc_valence", 0.0), v.get("ccc_arousal", 0.0)) > 0.0
    )
    safety_mae_ok = _per_fold(
        lambda v: max(v.get("mae_valence", 1.0), v.get("mae_arousal", 1.0)) <= 0.30
    )
    safety_ccc_pass = safety_ccc_ok >= safety_threshold
    safety_mae_pass = safety_mae_ok >= safety_threshold

    return {
        "primary_ccc_pass": primary_ccc_pass,
        "primary_mae_pass": primary_mae_pass,
        "safety_ccc_pass": safety_ccc_pass,
        "safety_mae_pass": safety_mae_pass,
        "safety_threshold": safety_threshold,
        "total_folds": total_folds,
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
    n_folds = report["gate"].get("total_folds", 7)
    lines.append(f"# LOMO {n_folds}-fold Report\n")
    lines.append(f"- Run: {report['run_id']}")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Movie set: {report.get('movie_set', 'cognimuse')}")
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
    thr = gate.get("safety_threshold", 6)
    total = gate.get("total_folds", 7)
    lines.append("\n## Gate (PLAN.md §합격 기준)\n")
    lines.append(f"- V/A Primary (CCC ≥ 0.20): **{'PASS' if gate['primary_ccc_pass'] else 'FAIL'}** "
                 f"(mean_ccc = {gate['mean_ccc_across_folds']:.4f})")
    lines.append(f"- V/A Primary (MAE ≤ 0.25): **{'PASS' if gate['primary_mae_pass'] else 'FAIL'}** "
                 f"(mean_mae = {gate['mean_mae_across_folds']:.4f})")
    lines.append(f"- V/A Safety (≥{thr}/{total} folds min(ccc_v,ccc_a) > 0): **"
                 f"{'PASS' if gate['safety_ccc_pass'] else 'FAIL'}** "
                 f"({gate['folds_passing_ccc_safety']}/{total})")
    lines.append(f"- V/A Safety (≥{thr}/{total} folds max(mae_v,mae_a) ≤ 0.30): **"
                 f"{'PASS' if gate['safety_mae_pass'] else 'FAIL'}** "
                 f"({gate['folds_passing_mae_safety']}/{total})")
    lines.append(f"- **Overall**: **{'PASS' if gate['overall_pass'] else 'FAIL'}**")

    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    movie_list = MOVIE_SETS[args.movie_set]
    if args.folds:
        folds = [int(x) for x in args.folds.split(",") if x.strip()]
    else:
        folds = list(range(len(movie_list)))

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    run_id = root.name
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    per_fold: list[dict] = []
    for k in folds:
        fold_output = root / f"fold_{k}_{movie_list[k]}"
        fold_argv = _compose_fold_argv(args, k, fold_output)
        fold_record: dict = {
            "fold": k,
            "test_movie_code": movie_list[k],
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

    gate = _evaluate_pass(per_fold, total_folds=len(movie_list))

    report = {
        "run_id": run_id,
        "timestamp": timestamp,
        "movie_set": args.movie_set,
        "movie_list": movie_list,
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
