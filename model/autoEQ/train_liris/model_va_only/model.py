"""AutoEQModelLirisVAOnly — BASE minus MoodHead.

BASE architecture with MoodHead completely removed (clean trainable param
count = 3,152,388 vs BASE 3,417,099). Forward pass returns a DUMMY
all-zeros `mood_logits (B, K)` tensor for trainer API compatibility;
`cfg.lambda_mood = 0.0` ensures this dummy has zero loss contribution.

Inherits from AutoEQModelLiris to reuse _modality_dropout / _feature_noise
helpers. __init__ is overridden via nn.Module.__init__ to skip MoodHead
creation.

Output interface preserved (same keys as BASE):
    {va_pred (B, 2), mood_logits (B, K=7) [dummy zeros], gate_weights (B, 2)}
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..model import (
    AudioProjection,
    AutoEQModelLiris,
    GateNetwork,
    VAHead,
    # NOTE: intentionally NOT importing MoodHead
)
from .config import TrainLirisConfigVAOnly


class AutoEQModelLirisVAOnly(AutoEQModelLiris):
    """VA-only variant: MoodHead removed. Dummy zeros returned for mood_logits."""

    def __init__(self, cfg: TrainLirisConfigVAOnly):
        # Skip AutoEQModelLiris.__init__ (which builds MoodHead) — build manually.
        nn.Module.__init__(self)
        self.cfg = cfg
        self.audio_projection = AudioProjection(cfg)
        self.gate_network = GateNetwork(cfg)
        self.va_head = VAHead(cfg)
        # MoodHead intentionally NOT created (BASE Phase 2a-5 / 2a-2 winner
        # arch minus auxiliary task). lambda_mood=0 ensures no loss contribution.

    def forward(self, visual_feat: Tensor, audio_feat: Tensor) -> dict[str, Tensor]:
        a_proj = self.audio_projection(audio_feat)
        v, a = self._modality_dropout(visual_feat, a_proj)
        v, a = self._feature_noise(v, a)

        gate = self.gate_network(v, a)
        w_v = gate[:, 0:1]
        w_a = gate[:, 1:2]
        fused = torch.cat([w_v * v, w_a * a], dim=-1)

        # Dummy all-zeros mood_logits for trainer API compatibility.
        # With cfg.lambda_mood=0.0 the resulting L_mood (≈ ln K = 1.945 constant)
        # contributes nothing to total loss — training is single-task V/A.
        B = fused.size(0)
        mood_logits = torch.zeros(
            B, self.cfg.num_mood_classes,
            dtype=fused.dtype, device=fused.device,
        )

        return {
            "va_pred": self.va_head(fused),
            "mood_logits": mood_logits,
            "gate_weights": gate,
        }
