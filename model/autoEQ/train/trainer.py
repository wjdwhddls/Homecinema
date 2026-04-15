import copy
import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from .config import TrainConfig
from .losses import combined_loss
from .model import AutoEQModel
from .negative_sampler import NegativeSampler
from .utils import compute_head_grad_norms, compute_mean_ccc


class Trainer:
    def __init__(
        self,
        model: AutoEQModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Optimizer: only trainable parameters
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduler: linear warmup + cosine annealing
        total_steps = config.epochs * len(train_loader)

        warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, config.warmup_steps)),
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - config.warmup_steps),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_steps],
        )

        self.negative_sampler = NegativeSampler(config)

        # Early stopping state
        self.best_mean_ccc = -float("inf")
        self.patience_counter = 0
        self.best_state_dict: dict | None = None

        # History
        self.history: list[dict] = []

    def train_one_epoch(self) -> dict:
        """Run one training epoch. Returns dict of average losses."""
        self.model.train()
        total_losses = {"va": 0.0, "mood": 0.0, "cong": 0.0, "gate_entropy": 0.0, "total": 0.0}
        grad_norms_accum: dict[str, float] = {"va": 0.0, "mood": 0.0, "cong": 0.0}
        gate_w_v_sum = 0.0
        gate_w_a_sum = 0.0
        num_batches = 0

        for batch in self.train_loader:
            visual = batch["visual_feat"].to(self.device)
            audio = batch["audio_feat"].to(self.device)
            va_target = torch.stack(
                [batch["valence"], batch["arousal"]], dim=-1
            ).to(self.device)
            mood_target = batch["mood"].to(self.device)
            movie_ids = batch["movie_id"]

            # Negative sampling
            audio, cong_target = self.negative_sampler.sample(
                audio, va_target, movie_ids
            )
            cong_target = cong_target.to(self.device)

            # Forward
            outputs = self.model(visual, audio, cong_label=cong_target)

            # Loss
            loss, loss_dict = combined_loss(
                outputs, va_target, mood_target, cong_target, self.config
            )

            # NaN/Inf guard: skip corrupted batches
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient norm measurement (before clipping)
            heads = {
                "va": self.model.va_head,
                "mood": self.model.mood_head,
                "cong": self.model.cong_head,
            }
            batch_grad_norms = compute_head_grad_norms(heads)

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Accumulate
            for k in total_losses:
                total_losses[k] += loss_dict[k]
            for k in grad_norms_accum:
                grad_norms_accum[k] += batch_grad_norms[k]
            gw = outputs["gate_weights"].detach()
            gate_w_v_sum += gw[:, 0].mean().item()
            gate_w_a_sum += gw[:, 1].mean().item()
            num_batches += 1

        # Average
        if num_batches > 0:
            for k in total_losses:
                total_losses[k] /= num_batches
            for k in grad_norms_accum:
                grad_norms_accum[k] /= num_batches

        total_losses["grad_norms"] = grad_norms_accum
        if num_batches > 0:
            total_losses["gate_w_v"] = gate_w_v_sum / num_batches
            total_losses["gate_w_a"] = gate_w_a_sum / num_batches
        return total_losses

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation. Returns dict with losses and CCC metrics."""
        self.model.eval()
        total_losses = {"va": 0.0, "mood": 0.0, "cong": 0.0, "gate_entropy": 0.0, "total": 0.0}
        if self.config.use_ccc_loss:
            total_losses["va_mse"] = 0.0
            total_losses["va_ccc"] = 0.0
        all_va_pred = []
        all_va_target = []
        gate_w_v_sum = 0.0
        gate_w_a_sum = 0.0
        num_batches = 0

        for batch in self.val_loader:
            visual = batch["visual_feat"].to(self.device)
            audio = batch["audio_feat"].to(self.device)
            va_target = torch.stack(
                [batch["valence"], batch["arousal"]], dim=-1
            ).to(self.device)
            mood_target = batch["mood"].to(self.device)
            cong_target = batch["cong_label"].to(self.device)

            outputs = self.model(visual, audio, cong_label=None)

            _, loss_dict = combined_loss(
                outputs, va_target, mood_target, cong_target, self.config
            )

            for k in total_losses:
                if k in loss_dict:
                    total_losses[k] += loss_dict[k]

            gw = outputs["gate_weights"]
            gate_w_v_sum += gw[:, 0].mean().item()
            gate_w_a_sum += gw[:, 1].mean().item()

            all_va_pred.append(outputs["va_pred"])
            all_va_target.append(va_target)
            num_batches += 1

        if num_batches > 0:
            for k in total_losses:
                total_losses[k] /= num_batches
            total_losses["gate_w_v"] = gate_w_v_sum / num_batches
            total_losses["gate_w_a"] = gate_w_a_sum / num_batches

        # CCC
        if all_va_pred:
            va_pred_cat = torch.cat(all_va_pred, dim=0)
            va_target_cat = torch.cat(all_va_target, dim=0)
            mean_ccc, ccc_v, ccc_a = compute_mean_ccc(va_pred_cat, va_target_cat)
            total_losses["mean_ccc"] = mean_ccc.item()
            total_losses["ccc_valence"] = ccc_v.item()
            total_losses["ccc_arousal"] = ccc_a.item()
        else:
            total_losses["mean_ccc"] = 0.0
            total_losses["ccc_valence"] = 0.0
            total_losses["ccc_arousal"] = 0.0

        return total_losses

    def check_early_stopping(self, val_metrics: dict) -> bool:
        """Check early stopping condition. Returns True if should stop."""
        mean_ccc = val_metrics.get("mean_ccc", 0.0)

        if mean_ccc > self.best_mean_ccc:
            self.best_mean_ccc = mean_ccc
            self.patience_counter = 0
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stop_patience

    def fit(self, max_epochs: int | None = None) -> list[dict]:
        """Full training loop.

        Args:
            max_epochs: override config.epochs if provided

        Returns:
            List of per-epoch history dicts.
        """
        epochs = max_epochs or self.config.epochs

        for epoch in range(epochs):
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            self.history.append(record)

            should_stop = self.check_early_stopping(val_metrics)
            if should_stop:
                break

        # Restore best model weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history
