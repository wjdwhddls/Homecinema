"""Config for Phase 2a-7 VA-only (no Mood head) variant.

Option B' approach: Override `lambda_mood = 0.0` so MoodHead-less model's
dummy-zero mood_logits output contributes nothing to the total loss, and
BASE trainer/loss code needs no modification.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainLirisConfig


@dataclass
class TrainLirisConfigVAOnly(TrainLirisConfig):
    """Phase 2a-7 VA-only variant.

    Overrides vs TrainLirisConfig (BASE, Phase 2a-5 rev.):
      lambda_mood     0.3  → 0.0   (disable multi-task auxiliary loss)
      run_name        →  "2a7_va_only"
      output_dir      →  "runs/phase2a/2a7_va_only"

    Architecturally, AutoEQModelLirisVAOnly removes MoodHead entirely
    (264,711 params saved). Forward returns a dummy `mood_logits`
    tensor of zeros so the BASE trainer (which always reads mood_logits)
    is happy without code change.

    All other BASE hyperparameters (K=7 nominal, va_norm="A",
    head_dropout=0.3, wd=1e-4, augmentations, etc.) inherited unchanged.
    """

    lambda_mood: float = 0.0  # disable multi-task aux loss
    run_name: str = "2a7_va_only"
    output_dir: str = "runs/phase2a/2a7_va_only"
