"""Trainer — Lexicographic early stopping + Overfit auto-monitor + per-term logging.

V5-FINAL §6 (early stop), §10-2 (overfit), §10-3 (per-term), §11 (baseline).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..train.utils import compute_mean_ccc, compute_va_regression_metrics
from .config import TrainLirisConfig
from .losses import combined_loss_liris


# --- EMA ----------------------------------------------------------------------


class ModelEMA:
    """Exponential moving average of model trainable parameters.

    Keeps a shadow copy of params and updates after each optimizer step.
    apply_to/restore temporarily swap live weights with the shadow for eval.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainLirisConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Linear warmup (warmup_steps) then cosine decay to 0 over remaining steps."""
    if not cfg.use_cosine_schedule:
        return None
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, cfg.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- Lexicographic metric -----------------------------------------------------


def lexicographic_better(
    new: dict[str, float], old: dict[str, float] | None
) -> bool:
    """Strict > on (mean_CCC, mean_Pearson, -mean_MAE)."""
    if old is None:
        return True
    if new["mean_ccc"] != old["mean_ccc"]:
        return new["mean_ccc"] > old["mean_ccc"]
    if new["mean_pearson"] != old["mean_pearson"]:
        return new["mean_pearson"] > old["mean_pearson"]
    # lower MAE is better
    return new["mean_mae"] < old["mean_mae"]


# --- Core train/eval loops ----------------------------------------------------


def _move(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    cfg: TrainLirisConfig,
    device: str,
    is_train: bool,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    model.train(is_train)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    all_pred, all_tgt = [], []
    for batch in loader:
        batch = _move(batch, device)
        visual = batch["visual"]
        audio = batch["audio"]
        va_target = batch["va_target"]
        mood_target = batch["mood_k4"] if cfg.num_mood_classes == 4 else batch["mood_k7"]

        with torch.set_grad_enabled(is_train):
            out = model(visual, audio)
            loss, log = combined_loss_liris(out, va_target, mood_target, cfg)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        B = va_target.size(0)
        count += B
        for k, v in log.items():
            totals[k] += v * B
        all_pred.append(out["va_pred"].detach().cpu())
        all_tgt.append(va_target.detach().cpu())

    for k in list(totals):
        totals[k] /= max(count, 1)

    preds = torch.cat(all_pred, dim=0)
    tgts = torch.cat(all_tgt, dim=0)
    mean_ccc, ccc_v, ccc_a = compute_mean_ccc(preds, tgts)
    extra = compute_va_regression_metrics(preds, tgts)

    metrics = dict(totals)
    metrics.update(
        {
            "mean_ccc": float(mean_ccc.item()),
            "ccc_v": float(ccc_v.item()),
            "ccc_a": float(ccc_a.item()),
            "mean_pearson": (extra["pearson_valence"] + extra["pearson_arousal"]) / 2.0,
            "mean_mae": (extra["mae_valence"] + extra["mae_arousal"]) / 2.0,
            **extra,
        }
    )
    return metrics


# --- Main training loop -------------------------------------------------------


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainLirisConfig,
    device: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = make_lr_scheduler(optimizer, cfg, steps_per_epoch)
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    if ema is not None:
        print(f"[ema] enabled  decay={cfg.ema_decay}")
    best_val: dict[str, float] | None = None
    bad_epochs = 0
    history: list[dict] = []

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, cfg, device, True, scheduler, ema=ema)
        if ema is not None:
            ema.apply_to(model)
        val_m = run_epoch(model, val_loader, None, cfg, device, False)
        current_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        gap = train_m["mean_ccc"] - val_m["mean_ccc"]
        overfit = gap > cfg.overfit_gap_threshold

        row = {
            "epoch": epoch,
            "dt_sec": round(dt, 1),
            "train": train_m,
            "val": val_m,
            "overfit_gap": round(gap, 4),
            "overfit_flag": overfit,
        }
        history.append(row)

        if lexicographic_better(val_m, best_val):
            best_val = val_m
            bad_epochs = 0
            # Save weights that achieved this val — EMA if enabled, else live.
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__, "val_metrics": val_m},
                best_path,
            )
            marker = "*"
        else:
            bad_epochs += 1
            marker = " "

        if ema is not None:
            ema.restore(model)

        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__}, last_path)

        msg = (
            f"[ep {epoch:02d}{marker}] "
            f"train CCC={train_m['mean_ccc']:+.4f} (v={train_m['ccc_v']:+.3f} a={train_m['ccc_a']:+.3f})  "
            f"val CCC={val_m['mean_ccc']:+.4f} (v={val_m['ccc_v']:+.3f} a={val_m['ccc_a']:+.3f})  "
            f"gap={gap:+.3f}{' !' if overfit else ''}  "
            f"L={train_m['loss_total']:.3f}"
            f" (va={train_m['loss_va']:.3f} mood={train_m['loss_mood']:.3f} gate={train_m['loss_gate_entropy']:+.3f})  "
            f"lr={current_lr:.2e}  bad={bad_epochs}  {dt:.1f}s"
        )
        print(msg)

        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

        if bad_epochs >= cfg.early_stop_patience:
            print(f"[early stop] patience {cfg.early_stop_patience} exhausted at epoch {epoch}")
            break

    return {"best_val": best_val, "history": history}
