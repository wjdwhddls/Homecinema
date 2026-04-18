"""MoodEQ training entrypoint.

Usage:
    # Synthetic smoke (no data needed)
    python -m model.autoEQ.train.run_train --use_synthetic --epochs 3

    # Real data (after precompute.py has produced feature .pt files)
    python -m model.autoEQ.train.run_train \\
        --feature_dir data/features \\
        --split_name liris_accede \\
        --stratified \\
        --use_wandb

Writes best model checkpoint (`best_model.pt`) and history
(`history.json`) to `--output_dir` (default: `runs/latest`).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig
from .dataset import (
    PrecomputedFeatureDataset,
    SyntheticAutoEQDataset,
    compute_movie_va,
    create_dataloaders,
    film_level_split,
    stratified_film_level_split,
)
from .model import AutoEQModel
from .trainer import Trainer


def set_seed(seed: int) -> None:
    """Fix seeds across random / numpy / torch (+ CUDA if present)."""
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
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.feature_dir:
        cfg.feature_dir = args.feature_dir
    cfg.use_wandb = args.use_wandb
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.wandb_run_name:
        cfg.wandb_run_name = args.wandb_run_name
    return cfg


def build_dataset(args: argparse.Namespace, config: TrainConfig):
    if args.use_synthetic:
        return SyntheticAutoEQDataset(
            num_clips=args.synthetic_num_clips,
            num_films=args.synthetic_num_films,
            config=config,
            seed=args.seed,
        )
    if not args.feature_dir:
        raise SystemExit("either --use_synthetic or --feature_dir must be set")
    return PrecomputedFeatureDataset(args.feature_dir, args.split_name)


def build_splits(args: argparse.Namespace, dataset):
    if args.stratified:
        if hasattr(dataset, "compute_per_movie_va"):
            movie_va = dataset.compute_per_movie_va()
        else:
            movie_va = compute_movie_va(
                dataset.movie_ids,
                dataset.valences.tolist(),
                dataset.arousals.tolist(),
            )
        return stratified_film_level_split(movie_va, seed=args.seed)
    return film_level_split(dataset.movie_ids, seed=args.seed)


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


def save_run_artifacts(trainer: Trainer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = trainer.best_state_dict or trainer.model.state_dict()
    torch.save(ckpt, output_dir / "best_model.pt")
    with (output_dir / "history.json").open("w") as f:
        json.dump(_jsonify(trainer.history), f, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MoodEQ training CLI")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--use_synthetic", action="store_true",
                     help="train on synthetic data (no feature files needed)")
    src.add_argument("--feature_dir", type=str, default="",
                     help="directory with {split_name}_{visual|audio|metadata}.pt files")
    p.add_argument("--split_name", type=str, default="liris_accede")
    p.add_argument("--synthetic_num_clips", type=int, default=120)
    p.add_argument("--synthetic_num_films", type=int, default=10)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratified", action="store_true",
                   help="V/A-quadrant stratified film-level split (spec 2-3)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--output_dir", type=str, default="runs/latest")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    set_seed(args.seed)
    device = select_device(args.device)
    config = build_config(args)

    dataset = build_dataset(args, config)
    train_ids, val_ids, _ = build_splits(args, dataset)
    train_loader, val_loader = create_dataloaders(dataset, train_ids, val_ids, config)

    model = AutoEQModel(config)
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    history = trainer.fit()

    save_run_artifacts(trainer, Path(args.output_dir))
    return {
        "history_len": len(history),
        "best_mean_ccc": trainer.best_mean_ccc,
        "output_dir": args.output_dir,
    }


if __name__ == "__main__":
    main()
