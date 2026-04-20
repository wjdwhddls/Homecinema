"""Simple-concat ablation config — subclasses TrainCogConfig.

The gate-entropy loss term is disabled (lambda = 0) because there is no gate
to regularise. Everything else — encoder dims, learning rate, augmentation
hyperparameters, early stopping — is inherited from the baseline so that the
single ablation axis is the fusion strategy.
"""

from dataclasses import dataclass

from ..config import TrainCogConfig


@dataclass
class TrainCogConfigConcat(TrainCogConfig):
    # Gate term has no meaning without a gate — zero it out so ``combined_loss_cog``
    # contributes nothing from the entropy head.
    lambda_gate_entropy: float = 0.0

    # Wandb — distinguish the ablation tracks by project name if enabled.
    wandb_project: str = "moodeq_cog_concat"
