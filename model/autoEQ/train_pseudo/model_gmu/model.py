"""GMU ablation model — swaps the scalar softmax gate for a GMU fusion block.

The baseline ``AutoEQModelCog`` concatenates scalar-weighted projected
features ``cat(w_v·v, w_a·a)`` into a 1024-dim vector. GMU replaces this with
an element-wise sigmoid gate producing a single 512-dim vector, which feeds
the VA / Mood heads (built with ``fused_dim=512`` in the GMU config).

For logging compatibility, ``gate_weights`` in the forward output is populated
with the batch-mean of ``z`` (visual ratio) in column 0 and ``1 - mean(z)`` in
column 1, so the trainer's (B, 2) expectation is satisfied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..model_base.model import AutoEQModelCog
from .config import TrainCogConfigGMU


class GMUFusion(nn.Module):
    """Arevalo 2017 — element-wise sigmoid gating over two modalities."""

    def __init__(self, config: TrainCogConfigGMU):
        super().__init__()
        d = config.gmu_hidden_dim
        self.Wv = nn.Linear(config.visual_dim, d)
        self.Wa = nn.Linear(config.audio_proj_dim, d)
        self.Wz = nn.Linear(config.visual_dim + config.audio_proj_dim, d)

    def forward(self, v: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        hv = torch.tanh(self.Wv(v))
        ha = torch.tanh(self.Wa(a))
        z = torch.sigmoid(self.Wz(torch.cat([v, a], dim=-1)))
        fused = z * hv + (1.0 - z) * ha
        return fused, z


class AutoEQModelGMU(AutoEQModelCog):
    """Same encoders, heads, dropout/noise pipeline as baseline; GMU fusion.

    Expected inputs (unchanged from baseline):
        visual_feat: (B, 512)  — frozen X-CLIP video embedding
        audio_feat:  (B, 2048) — frozen PANNs CNN14
    """

    def __init__(self, config: TrainCogConfigGMU):
        # ``super().__init__`` builds audio_projection / gate_network / heads
        # using ``fused_dim=512`` (from config), so VAHead/MoodHead already
        # expect a 512-dim input — exactly what GMU produces. We then drop the
        # baseline scalar gate (unused) and install the GMU block.
        super().__init__(config)
        del self.gate_network
        self.gmu = GMUFusion(config)

    def forward(
        self,
        visual_feat: Tensor,
        audio_feat: Tensor,
    ) -> dict[str, Tensor]:
        a_proj = self.audio_projection(audio_feat)        # (B, 512)
        v, a = self._apply_modality_dropout(visual_feat, a_proj)
        v, a = self._apply_feature_noise(v, a)

        fused, z = self.gmu(v, a)                         # (B, 512), (B, 512)

        # Collapse z into a 2-dim gate-weight summary for trainer logging:
        # mean(z) is the overall visual-share; 1 - mean(z) is audio-share.
        z_mean = z.mean(dim=-1, keepdim=True)             # (B, 1)
        gate_weights = torch.cat([z_mean, 1.0 - z_mean], dim=-1)  # (B, 2)

        return {
            "va_pred": self.va_head(fused),
            "mood_logits": self.mood_head(fused),
            "gate_weights": gate_weights,
        }


__all__ = ["AutoEQModelGMU", "GMUFusion"]
