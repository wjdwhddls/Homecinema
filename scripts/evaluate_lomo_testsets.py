"""Evaluate each fold's best_model.pt on its held-out test movie.

The LOMO trainer (model_base/run_train.py) computes metrics on the
**val split of the training films** (15% tail) and saves `best_model.pt`
based on that. The hold-out test movie is excluded from training/eval
loaders — its windows are merely flagged and never scored. This script
closes that gap: for every fold_* directory under --run_dir, it
re-materialises the test window list using the exact same
``lomo_splits_with_time_val`` logic, loads that fold's ``best_model.pt``,
runs it over the held-out movie windows, and aggregates CCC/MAE/RMSE
across folds.

Result written to ``<run_dir>/lomo_test_report.json`` and printed.

Usage:
    python scripts/evaluate_lomo_testsets.py \\
        --run_dir runs/final_gmu_vaonly_lomo9_sigma_off \\
        --feature_dir data/features/ccmovies \\
        --variant gmu
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.autoEQ.train_pseudo.dataset import (
    PrecomputedCogDataset,
    lomo_splits_with_time_val,
)
from model.autoEQ.train.utils import (
    compute_mean_ccc,
    compute_mood_metrics,
    compute_va_regression_metrics,
)
from model.autoEQ.train_pseudo.dataset import va_to_quadrant


# variant key → (module, model_cls_name, config_cls_name)
VARIANTS = {
    "base":    ("model_base",           "AutoEQModelCog",           "TrainCogConfig"),
    "gmu":     ("model_gmu",            "AutoEQModelGMU",           "TrainCogConfigGMU"),
    "concat":  ("model_concat",         "AutoEQModelConcat",        "TrainCogConfigConcat"),
    "ast":     ("model_ast",            "AutoEQModelAST",           "TrainCogConfigAST"),
    "clipimg": ("model_clip_framemean", "AutoEQModelClipFrameMean", "TrainCogConfigClipFrameMean"),
    "ast_gmu": ("model_ast_gmu",        "AutoEQModelASTGMU",        "TrainCogConfigASTGMU"),
}


def load_variant(variant_key: str):
    mod_name, model_cls_name, config_cls_name = VARIANTS[variant_key]
    # model_base's config lives at train_pseudo.config (not under model_base/)
    if variant_key == "base":
        cfg_mod = importlib.import_module("model.autoEQ.train_pseudo.config")
    else:
        cfg_mod = importlib.import_module(f"model.autoEQ.train_pseudo.{mod_name}.config")
    model_mod = importlib.import_module(f"model.autoEQ.train_pseudo.{mod_name}.model")
    return getattr(model_mod, model_cls_name), getattr(cfg_mod, config_cls_name)


def build_config_from_fold_mapping(config_cls, fm: dict):
    """Construct a config dataclass matching the fold's recorded hyperparameters."""
    kwargs = {}
    for key in (
        "modality_dropout_p",
        "feature_noise_std",
        "mixup_prob",
        "mixup_alpha",
        "label_smooth_eps",
        "label_smooth_sigma_threshold",
        "sigma_filter_threshold",
        "val_tail_ratio",
        "val_gap_windows",
        "lambda_mood",
    ):
        if key in fm and fm[key] is not None:
            kwargs[key] = fm[key]
    # num_mood_classes: fold_mapping may store None; default to baseline config default
    nmc = fm.get("num_mood_classes")
    if nmc is not None:
        kwargs["num_mood_classes"] = int(nmc)
    return config_cls(**kwargs)


@torch.no_grad()
def evaluate_test_movie(model, dataset, test_ids, device, batch_size) -> dict:
    wid_to_idx = {w: i for i, w in enumerate(dataset.window_ids)}
    test_indices = [wid_to_idx[w] for w in test_ids if w in wid_to_idx]
    if not test_indices:
        return {"error": "no test windows"}
    loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    all_va_pred, all_va_target = [], []
    all_mood_logits, all_mood_target = [], []
    for batch in loader:
        visual = batch["visual_feat"].to(device)
        audio = batch["audio_feat"].to(device)
        va_target = torch.stack(
            [batch["valence"], batch["arousal"]], dim=-1
        ).to(device)
        mood_target = batch["mood"].to(device)
        out = model(visual, audio)
        all_va_pred.append(out["va_pred"])
        all_va_target.append(va_target)
        all_mood_logits.append(out["mood_logits"])
        all_mood_target.append(mood_target)
    va_pred = torch.cat(all_va_pred, dim=0)
    va_target = torch.cat(all_va_target, dim=0)
    mood_logits = torch.cat(all_mood_logits, dim=0)
    mood_target = torch.cat(all_mood_target, dim=0)

    # compute CCC (BUG FIX 2026-04-20: compute_mean_ccc returns (mean, v, a) —
    # previous tuple unpacking was swapped, causing ccc_valence/ccc_arousal to
    # be misreported across all test-set evaluations. See audit log.)
    mean_ccc, ccc_v, ccc_a = compute_mean_ccc(va_pred, va_target)
    # MAE/RMSE/Pearson (per-dim)
    reg = compute_va_regression_metrics(va_pred, va_target)
    # Mood metrics
    mood = compute_mood_metrics(mood_logits, mood_target)
    metrics = {
        "test_n": len(test_indices),
        "mean_ccc": float(mean_ccc.item() if hasattr(mean_ccc, "item") else mean_ccc),
        "ccc_valence": float(ccc_v.item() if hasattr(ccc_v, "item") else ccc_v),
        "ccc_arousal": float(ccc_a.item() if hasattr(ccc_a, "item") else ccc_a),
    }
    for k, v in reg.items():
        metrics[k] = float(v.item() if hasattr(v, "item") else v)
    # Add mean_mae, mean_rmse if not present
    if "mean_mae" not in metrics and "mae_valence" in metrics and "mae_arousal" in metrics:
        metrics["mean_mae"] = 0.5 * (metrics["mae_valence"] + metrics["mae_arousal"])
    if "mean_rmse" not in metrics and "rmse_valence" in metrics and "rmse_arousal" in metrics:
        metrics["mean_rmse"] = 0.5 * (metrics["rmse_valence"] + metrics["rmse_arousal"])
    for k, v in mood.items():
        # compute_mood_metrics already prefixes with "mood_" so no double-prefix.
        if isinstance(v, (int, float)):
            metrics[k] = float(v)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--feature_dir", type=Path, required=True)
    parser.add_argument("--split_name", type=str, default="ccmovies")
    parser.add_argument("--variant", choices=list(VARIANTS), default="gmu")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Device resolution
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    model_cls, config_cls = load_variant(args.variant)

    # Build dataset once (it's the full split, we'll subset per fold).
    # num_mood_classes for dataset: pick 4 for K=4 trained runs (fold_mapping will tell).
    # Use the first fold's mapping to pick; fall back to 4.
    fold_dirs = sorted(args.run_dir.glob("fold_*_*"))
    if not fold_dirs:
        raise SystemExit(f"no fold directories under {args.run_dir}")
    fm0 = json.load(open(fold_dirs[0] / "fold_mapping.json"))
    nmc = int(fm0.get("num_mood_classes") or 4)
    dataset = PrecomputedCogDataset(
        feature_dir=str(args.feature_dir),
        split_name=args.split_name,
        num_mood_classes=nmc,
    )

    print(f"Run dir    : {args.run_dir}")
    print(f"Feature dir: {args.feature_dir}")
    print(f"Variant    : {args.variant}")
    print(f"Device     : {args.device}")
    print(f"Dataset    : {len(dataset)} windows,  num_mood_classes={nmc}")
    print()

    per_fold = []
    for fd in fold_dirs:
        fm = json.load(open(fd / "fold_mapping.json"))
        fold = int(fm["fold"])
        test_movie = fm["test_movie_code"]
        movie_list = fm["movie_list"]

        # Regenerate train/val/test ids using the exact recorded settings.
        train_ids, val_ids, test_ids = lomo_splits_with_time_val(
            dataset.metadata,
            fold=fold,
            val_tail_ratio=float(fm["val_tail_ratio"]),
            gap_windows=int(fm["val_gap_windows"]),
            movie_list=movie_list,
        )

        config = build_config_from_fold_mapping(config_cls, fm)
        model = model_cls(config).to(args.device)
        ckpt_path = fd / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print(f"  [warn] fold {fold} state_dict missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

        metrics = evaluate_test_movie(
            model, dataset, test_ids, args.device, args.batch_size
        )
        metrics["fold"] = fold
        metrics["test_movie_code"] = test_movie
        per_fold.append(metrics)

        print(
            f"fold {fold}  {test_movie:<22} n={metrics['test_n']:>4}  "
            f"CCC={metrics['mean_ccc']:+.4f}  "
            f"CCC_V={metrics['ccc_valence']:+.4f}  CCC_A={metrics['ccc_arousal']:+.4f}  "
            f"MAE={metrics.get('mean_mae', float('nan')):.4f}"
        )

    print()
    print("=== Aggregate (fold-mean ± std) on HOLD-OUT TEST MOVIES ===")
    agg = {}
    for k in (
        "mean_ccc",
        "ccc_valence",
        "ccc_arousal",
        "mean_mae",
        "mae_valence",
        "mae_arousal",
        "mean_rmse",
        "rmse_valence",
        "rmse_arousal",
        "pearson_valence",
        "pearson_arousal",
        "mood_accuracy",
        "mood_f1_macro",
        "mood_f1_weighted",
        "mood_kappa",
    ):
        vals = [f[k] for f in per_fold if k in f]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg[k] = {"mean": m, "std": s}
        print(f"  {k:<18} {m:+.4f} ± {s:.4f}")

    report = {
        "run_dir": str(args.run_dir),
        "feature_dir": str(args.feature_dir),
        "variant": args.variant,
        "per_fold": per_fold,
        "aggregate": agg,
    }
    out_path = args.run_dir / "lomo_test_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
