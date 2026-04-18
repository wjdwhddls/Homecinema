import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from ..train.utils import (
    compute_head_grad_norms,
    compute_mean_ccc,
    compute_mood_metrics,
    compute_va_regression_metrics,
)
from .config import TrainCogConfig
from .losses import combined_loss_cog
from .model import AutoEQModelCog


class TrainerCog:
    """CogniMuse-only trainer. Differences from train.Trainer:

    - No NegativeSampler / cong_target / cong logits.
    - forward() called with only (visual, audio).
    - Early stopping: (mean_ccc, -mean_mae) tuple comparison → CCC primary
      with MAE tiebreaker, prevents Pareto regression.
    - Logs RMSE + mean_rmse in validation metrics.
    """

    def __init__(
        self,
        model: AutoEQModelCog,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainCogConfig,
        device: torch.device | None = None,
        wandb_run=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.wandb_run = wandb_run
        if config.use_wandb and self.wandb_run is None:
            self.wandb_run = self._maybe_init_wandb()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        total_steps = config.epochs * max(1, len(train_loader))
        warmup = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, config.warmup_steps)),
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - config.warmup_steps),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_steps],
        )

        # Early stopping state — tuple (ccc, -mae) tracked
        self.best_mean_ccc = -float("inf")
        self.best_mean_mae = float("inf")
        self.patience_counter = 0
        self.best_state_dict: dict | None = None

        self.history: list[dict] = []

    def _maybe_init_wandb(self):
        try:
            import wandb
        except ImportError:
            return None
        return wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name or None,
            config=vars(self.config),
        )

    def _log_to_wandb(self, epoch: int, train_m: dict, val_m: dict) -> None:
        if self.wandb_run is None:
            return
        payload = {
            "epoch": epoch,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        scalar_keys = (
            "va", "mood", "gate_entropy", "total",
            "gate_w_v", "gate_w_a",
            "mean_ccc", "ccc_valence", "ccc_arousal",
            "va_mse", "va_ccc",
            "mae_valence", "mae_arousal", "mean_mae",
            "rmse_valence", "rmse_arousal", "mean_rmse",
            "pearson_valence", "pearson_arousal",
            "mood_accuracy", "mood_f1_macro", "mood_f1_weighted", "mood_kappa",
        )
        for prefix, metrics in (("train", train_m), ("val", val_m)):
            for k in scalar_keys:
                v = metrics.get(k)
                if v is None or isinstance(v, dict):
                    continue
                payload[f"{prefix}/{k}"] = v
            for head_name, g in metrics.get("grad_norms", {}).items():
                payload[f"{prefix}/grad_norm/{head_name}"] = g
        self.wandb_run.log(payload)

    def train_one_epoch(self) -> dict:
        self.model.train()
        total_losses = {"va": 0.0, "mood": 0.0, "gate_entropy": 0.0, "total": 0.0}
        grad_norms_accum: dict[str, float] = {"va": 0.0, "mood": 0.0}
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

            outputs = self.model(visual, audio)
            loss, loss_dict = combined_loss_cog(
                outputs, va_target, mood_target, self.config
            )

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            loss.backward()

            heads = {
                "va": self.model.va_head,
                "mood": self.model.mood_head,
            }
            batch_grad_norms = compute_head_grad_norms(heads)

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            for k in total_losses:
                total_losses[k] += loss_dict[k]
            for k in grad_norms_accum:
                grad_norms_accum[k] += batch_grad_norms[k]
            gw = outputs["gate_weights"].detach()
            gate_w_v_sum += gw[:, 0].mean().item()
            gate_w_a_sum += gw[:, 1].mean().item()
            num_batches += 1

        if num_batches > 0:
            for k in total_losses:
                total_losses[k] /= num_batches
            for k in grad_norms_accum:
                grad_norms_accum[k] /= num_batches
            total_losses["gate_w_v"] = gate_w_v_sum / num_batches
            total_losses["gate_w_a"] = gate_w_a_sum / num_batches
        total_losses["grad_norms"] = grad_norms_accum
        return total_losses

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_losses = {"va": 0.0, "mood": 0.0, "gate_entropy": 0.0, "total": 0.0}
        if self.config.use_ccc_loss:
            total_losses["va_mse"] = 0.0
            total_losses["va_ccc"] = 0.0
        all_va_pred: list[torch.Tensor] = []
        all_va_target: list[torch.Tensor] = []
        all_mood_logits: list[torch.Tensor] = []
        all_mood_target: list[torch.Tensor] = []
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

            outputs = self.model(visual, audio)
            _, loss_dict = combined_loss_cog(
                outputs, va_target, mood_target, self.config
            )

            for k in total_losses:
                if k in loss_dict:
                    total_losses[k] += loss_dict[k]

            gw = outputs["gate_weights"]
            gate_w_v_sum += gw[:, 0].mean().item()
            gate_w_a_sum += gw[:, 1].mean().item()

            all_va_pred.append(outputs["va_pred"])
            all_va_target.append(va_target)
            all_mood_logits.append(outputs["mood_logits"])
            all_mood_target.append(mood_target)
            num_batches += 1

        if num_batches > 0:
            for k in total_losses:
                total_losses[k] /= num_batches
            total_losses["gate_w_v"] = gate_w_v_sum / num_batches
            total_losses["gate_w_a"] = gate_w_a_sum / num_batches

        if all_va_pred:
            va_pred_cat = torch.cat(all_va_pred, dim=0)
            va_target_cat = torch.cat(all_va_target, dim=0)
            mean_ccc, ccc_v, ccc_a = compute_mean_ccc(va_pred_cat, va_target_cat)
            total_losses["mean_ccc"] = mean_ccc.item()
            total_losses["ccc_valence"] = ccc_v.item()
            total_losses["ccc_arousal"] = ccc_a.item()
            total_losses.update(compute_va_regression_metrics(va_pred_cat, va_target_cat))

            # Aggregated scalars for gate + early stopping
            mean_mae = 0.5 * (total_losses["mae_valence"] + total_losses["mae_arousal"])
            mean_rmse = 0.5 * (total_losses["rmse_valence"] + total_losses["rmse_arousal"])
            total_losses["mean_mae"] = mean_mae
            total_losses["mean_rmse"] = mean_rmse

            mood_logits_cat = torch.cat(all_mood_logits, dim=0)
            mood_target_cat = torch.cat(all_mood_target, dim=0)
            total_losses.update(
                compute_mood_metrics(
                    mood_logits_cat, mood_target_cat, self.config.num_mood_classes
                )
            )
        else:
            total_losses["mean_ccc"] = 0.0
            total_losses["ccc_valence"] = 0.0
            total_losses["ccc_arousal"] = 0.0
            total_losses["mean_mae"] = 1.0
            total_losses["mean_rmse"] = 1.0

        return total_losses

    def check_early_stopping(self, val_metrics: dict) -> bool:
        """CCC primary + MAE tiebreaker (Pareto-guard).

        Update best when (ccc, -mae) tuple strictly improves.
        """
        mean_ccc = float(val_metrics.get("mean_ccc", 0.0))
        mean_mae = float(val_metrics.get("mean_mae", 1.0))

        current = (mean_ccc, -mean_mae)
        best = (self.best_mean_ccc, -self.best_mean_mae)
        if current > best:
            self.best_mean_ccc = mean_ccc
            self.best_mean_mae = mean_mae
            self.patience_counter = 0
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stop_patience

    def fit(self, max_epochs: int | None = None) -> list[dict]:
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
            self._log_to_wandb(epoch, train_metrics, val_metrics)
            if self.check_early_stopping(val_metrics):
                break
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass
        return self.history
