"""CLI — Phase 2a-7 VA-only (no MoodHead) training.

Reuses BASE's PrecomputedLirisDataset / features.pt (liris_panns_v5spec) —
no precompute step. Only mood-head removal + lambda_mood=0 change.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..dataset import (
    MixupTargetShrinkageCollator,
    PrecomputedLirisDataset,
    official_split,
)
from ..trainer import train_model
from .config import TrainLirisConfigVAOnly
from .model import AutoEQModelLirisVAOnly


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_loaders(
    cfg: TrainLirisConfigVAOnly,
    features: dict,
    splits: dict[str, pd.DataFrame],
) -> dict[str, DataLoader]:
    ds_train = PrecomputedLirisDataset(splits["train"], features)
    ds_val = PrecomputedLirisDataset(splits["val"], features)
    ds_test = PrecomputedLirisDataset(splits["test"], features)

    train_collate = MixupTargetShrinkageCollator(cfg, active=True)
    eval_collate = MixupTargetShrinkageCollator(cfg, active=False)

    return {
        "train": DataLoader(
            ds_train, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, collate_fn=train_collate, drop_last=True,
        ),
        "val": DataLoader(
            ds_val, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=eval_collate,
        ),
        "test": DataLoader(
            ds_test, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=eval_collate,
        ),
    }


def load_features(path: Path) -> dict:
    print(f"[load] features: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def override_from_args(
    cfg: TrainLirisConfigVAOnly, args: argparse.Namespace
) -> TrainLirisConfigVAOnly:
    for k, v in vars(args).items():
        if v is None:
            continue
        if hasattr(cfg, k) and k not in {"output_root"}:
            setattr(cfg, k, v)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--feature-file", default=None)
    p.add_argument("--metadata-csv", default=None)
    p.add_argument("--use-full-learning-set", action="store_true", default=None)
    p.add_argument("--modality-dropout-p", type=float, default=None)
    p.add_argument("--feature-noise-std", type=float, default=None)
    p.add_argument("--mixup-prob", type=float, default=None)
    p.add_argument("--target-shrinkage-eps", type=float, default=None)
    # Deliberately NOT exposing --lambda-mood (kept at 0 by design)
    p.add_argument("--lambda-gate-entropy", type=float, default=None)
    p.add_argument("--ccc-loss-weight", type=float, default=None)
    p.add_argument("--head-dropout", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--va-norm-strategy", choices=["A", "B"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    args = p.parse_args()

    cfg = TrainLirisConfigVAOnly()
    cfg = override_from_args(cfg, args)
    if args.run_name is None:
        cfg.run_name = f"2a7_va_only_s{cfg.seed}"
    if args.output_dir is None:
        cfg.output_dir = f"runs/phase2a/{cfg.run_name}"
    # Hard-guard: lambda_mood must remain 0 for VA-only semantics
    assert cfg.lambda_mood == 0.0, (
        f"lambda_mood must be 0 in VA-only variant, got {cfg.lambda_mood}"
    )
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    print(f"[run] device={device} lambda_mood={cfg.lambda_mood} (VA-only)")

    features = load_features(Path(cfg.feature_file))
    splits = official_split(
        Path(cfg.metadata_csv),
        use_full_learning_set=cfg.use_full_learning_set,
        va_norm_strategy=cfg.va_norm_strategy,
    )
    print(
        f"[split] train={len(splits['train'])}  val={len(splits['val'])}  "
        f"test={len(splits['test'])}"
    )

    loaders = build_loaders(cfg, features, splits)
    model = AutoEQModelLirisVAOnly(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_trainable/1e6:.2f}M (VA-only, no MoodHead)")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    results = train_model(model, loaders["train"], loaders["val"], cfg, device, out)
    (out / "summary.json").write_text(
        json.dumps({"best_val": results["best_val"]}, indent=2)
    )
    print(f"\n[done] best val: {results['best_val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
