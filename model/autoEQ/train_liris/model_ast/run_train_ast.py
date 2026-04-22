"""CLI — Phase 2a-3 AST variant training.

Mirrors `model/autoEQ/train_liris/run_train.py` but swaps:
    TrainLirisConfig           → TrainLirisConfigAST
    AutoEQModelLiris           → AutoEQModelLirisAST
    PrecomputedLirisDataset    → PrecomputedLirisDatasetAST

All other pipeline components (losses, trainer, augmentation collator,
official_split, seeding) are imported as-is from `..` — ensuring the only
axis under ablation is the audio encoder / feature path (V5-FINAL §21 OAT).
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

from ..trainer import train_model
from .config import TrainLirisConfigAST
from .dataset import (
    MixupTargetShrinkageCollator,
    PrecomputedLirisDatasetAST,
    official_split,
)
from .model import AutoEQModelLirisAST


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_loaders(
    cfg: TrainLirisConfigAST,
    features: dict,
    splits: dict[str, pd.DataFrame],
) -> dict[str, DataLoader]:
    ds_train = PrecomputedLirisDatasetAST(splits["train"], features)
    ds_val = PrecomputedLirisDatasetAST(splits["val"], features)
    ds_test = PrecomputedLirisDatasetAST(splits["test"], features)

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
    cfg: TrainLirisConfigAST, args: argparse.Namespace
) -> TrainLirisConfigAST:
    for k, v in vars(args).items():
        if v is None:
            continue
        if hasattr(cfg, k) and k not in {"output_root"}:
            setattr(cfg, k, v)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="2a3_ast")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-mood-classes", type=int, default=None, choices=[4, 7])
    p.add_argument("--feature-file", default=None)
    p.add_argument("--metadata-csv", default=None)
    p.add_argument("--use-full-learning-set", action="store_true", default=None)
    p.add_argument("--modality-dropout-p", type=float, default=None)
    p.add_argument("--feature-noise-std", type=float, default=None)
    p.add_argument("--mixup-prob", type=float, default=None)
    p.add_argument("--target-shrinkage-eps", type=float, default=None)
    p.add_argument("--lambda-mood", type=float, default=None)
    p.add_argument("--lambda-gate-entropy", type=float, default=None)
    p.add_argument("--ccc-loss-weight", type=float, default=None)
    p.add_argument("--head-dropout", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--use-ema", action="store_true", default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--va-norm-strategy", choices=["A", "B"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--ast-model-name", type=str, default=None)
    args = p.parse_args()

    cfg = TrainLirisConfigAST()
    cfg = override_from_args(cfg, args)
    if args.output_dir is None:
        cfg.output_dir = f"runs/phase2a/{cfg.run_name}"
    set_seed(cfg.seed)
    device = cfg.resolved_device()
    print(f"[run] device={device} cfg={cfg}")

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
    model = AutoEQModelLirisAST(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_trainable/1e6:.2f}M (AST variant, audio_raw_dim={cfg.audio_raw_dim})")

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
