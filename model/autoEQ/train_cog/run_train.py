"""Single-fold LOMO training entrypoint (CogniMuse).

Usage:
    python -m model.autoEQ.train_cog.run_train \\
        --feature_dir data/features/cognimuse \\
        --split_name cognimuse \\
        --lomo_fold 0 \\
        --epochs 30

Writes:
    <output_dir>/best_model.pt
    <output_dir>/history.json
    <output_dir>/fold_mapping.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import TrainCogConfig
from .dataset import (
    COGNIMUSE_MOVIES,
    PrecomputedCogDataset,
    apply_sigma_filter,
    create_dataloaders_cog,
    lomo_splits_with_time_val,
)
from .model import AutoEQModelCog
from .trainer import TrainerCog


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args: argparse.Namespace) -> TrainCogConfig:
    cfg = TrainCogConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.feature_dir:
        cfg.feature_dir = args.feature_dir
    if args.val_tail_ratio is not None:
        cfg.val_tail_ratio = args.val_tail_ratio
    if args.val_gap_windows is not None:
        cfg.val_gap_windows = args.val_gap_windows
    if args.modality_dropout_p is not None:
        cfg.modality_dropout_p = args.modality_dropout_p
    if args.sigma_filter_threshold is not None:
        cfg.sigma_filter_threshold = args.sigma_filter_threshold
    if args.num_mood_classes is not None:
        cfg.num_mood_classes = args.num_mood_classes
    if args.lambda_mood is not None:
        cfg.lambda_mood = args.lambda_mood
    cfg.use_wandb = args.use_wandb
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.wandb_run_name:
        cfg.wandb_run_name = args.wandb_run_name
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CogniMuse single-fold LOMO training")
    p.add_argument("--feature_dir", type=str, required=True)
    p.add_argument("--split_name", type=str, default="cognimuse")
    p.add_argument("--lomo_fold", type=int, required=True,
                   help="0..6, index into COGNIMUSE_MOVIES")
    p.add_argument("--val_tail_ratio", type=float, default=None)
    p.add_argument("--val_gap_windows", type=int, default=None)
    p.add_argument("--sigma_filter_threshold", type=float, default=None)
    p.add_argument("--modality_dropout_p", type=float, default=None)
    p.add_argument("--num_mood_classes", type=int, default=None, choices=[4, 7])
    p.add_argument("--lambda_mood", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--output_dir", type=str, default="runs/cog_latest")
    return p


def _save_fold_mapping(
    output_dir: Path,
    args: argparse.Namespace,
    config: TrainCogConfig,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    dropped_window_ids: list[str],
    sigma_filtered_out: int,
    preprocess_manifest_sha: dict | None,
) -> None:
    test_code = COGNIMUSE_MOVIES[args.lomo_fold]
    mapping = {
        "run_id": output_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fold": args.lomo_fold,
        "test_movie_code": test_code,
        "test_movie_id": args.lomo_fold,
        "train_movies": [c for c in COGNIMUSE_MOVIES if c != test_code],
        "val_tail_ratio": config.val_tail_ratio,
        "val_gap_windows": config.val_gap_windows,
        "dropped_window_ids": dropped_window_ids,
        "window_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "sigma_filter_threshold": config.sigma_filter_threshold,
        "sigma_filtered_out_count": sigma_filtered_out,
        "modality_dropout_p": config.modality_dropout_p,
        "num_mood_classes": config.num_mood_classes,
        "lambda_mood": config.lambda_mood,
        "seed": args.seed,
        "preprocess_manifest_sha": preprocess_manifest_sha,
        "cognimuse_movies_constant": COGNIMUSE_MOVIES,
    }
    with open(output_dir / "fold_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    set_seed(args.seed)
    device = select_device(args.device)
    config = build_config(args)

    # Dataset + splits
    dataset = PrecomputedCogDataset(
        args.feature_dir,
        args.split_name,
        num_mood_classes=config.num_mood_classes,
    )
    train_ids, val_ids, test_ids = lomo_splits_with_time_val(
        dataset.metadata,
        fold=args.lomo_fold,
        val_tail_ratio=config.val_tail_ratio,
        gap_windows=config.val_gap_windows,
    )
    all_window_ids = set(dataset.window_ids)
    kept = set(train_ids) | set(val_ids) | set(test_ids)
    dropped_window_ids = sorted(all_window_ids - kept)

    # σ filter on train split only
    train_pre_filter = len(train_ids)
    train_ids = apply_sigma_filter(train_ids, dataset.metadata, config.sigma_filter_threshold)
    sigma_filtered_out = train_pre_filter - len(train_ids)

    train_loader, val_loader = create_dataloaders_cog(dataset, train_ids, val_ids, config)

    # Model
    model = AutoEQModelCog(config)
    trainer = TrainerCog(model, train_loader, val_loader, config, device=device)

    # Manifest SHA passthrough (optional)
    manifest_sha = None
    manifest_path = Path(args.feature_dir) / "cognimuse_preprocess_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_sha = manifest.get("file_sha")

    # Output dir + fold_mapping BEFORE training (fail-fast on I/O)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_fold_mapping(
        output_dir,
        args,
        config,
        train_ids,
        val_ids,
        test_ids,
        dropped_window_ids,
        sigma_filtered_out,
        manifest_sha,
    )

    history = trainer.fit()

    ckpt = trainer.best_state_dict or trainer.model.state_dict()
    torch.save(ckpt, output_dir / "best_model.pt")
    with open(output_dir / "history.json", "w") as f:
        json.dump(_jsonify(history), f, indent=2)

    return {
        "fold": args.lomo_fold,
        "test_movie_code": COGNIMUSE_MOVIES[args.lomo_fold],
        "best_mean_ccc": trainer.best_mean_ccc,
        "best_mean_mae": trainer.best_mean_mae,
        "history_len": len(history),
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
