"""Simple-concat ablation model — ``AutoEQModelCog`` with the gate removed.

The baseline tower's gate network is deleted after ``super().__init__`` so it
contributes no parameters / gradients. ``gate_weights`` is returned as a
constant (0.5, 0.5) placeholder so the trainer's logging code (which reads
``outputs["gate_weights"]``) continues to work without modification.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model_base.model import AutoEQModelCog
from .config import TrainCogConfigConcat


class AutoEQModelConcat(AutoEQModelCog):
    """Same architecture as AutoEQModelCog minus the gate network.

    Expected inputs (unchanged from baseline):
        visual_feat: (B, 512)  — frozen X-CLIP video embedding
        audio_feat:  (B, 2048) — frozen PANNs CNN14
    """

    def __init__(self, config: TrainCogConfigConcat):
        super().__init__(config)
        # Remove gate parameters entirely so they don't receive gradients.
        del self.gate_network

    def forward(
        self,
        visual_feat: Tensor,
        audio_feat: Tensor,
    ) -> dict[str, Tensor]:
        a_proj = self.audio_projection(audio_feat)      # (B, 512)
        v, a = self._apply_modality_dropout(visual_feat, a_proj)
        v, a = self._apply_feature_noise(v, a)

        fused = torch.cat([v, a], dim=-1)               # (B, 1024) — no gating

        # Trainer reads gate_weights for logging; emit a constant 0.5/0.5 so
        # the log column remains a valid scalar and entropy = ln 2.
        B = v.size(0)
        gate_weights = torch.full(
            (B, 2), 0.5, device=v.device, dtype=v.dtype
        )

        return {
            "va_pred": self.va_head(fused),
            "mood_logits": self.mood_head(fused),
            "gate_weights": gate_weights,
        }


__all__ = ["AutoEQModelConcat"]
