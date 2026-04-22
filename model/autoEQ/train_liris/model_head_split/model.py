"""AutoEQModelLirisHeadSplit — BASE architecture with SeparateVAHead + configurable fusion.

Inherits from AutoEQModelLiris so that the augmentation utilities
(_modality_dropout, _feature_noise) are reused. We manually call
nn.Module.__init__ to skip the parent's layer construction, then build our
own set of modules with SeparateVAHead replacing the joint VAHead.

Fusion mechanisms (concat / gate / gmu) are imported from the Phase 2a-5
`model_fusion` subpackage — no duplication.

Output interface: {va_pred (B, 2), mood_logits (B, K=7), gate_weights (B, 2)}
is kept identical to BASE so the trainer / loss code remain unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..model import (
    AudioProjection,
    AutoEQModelLiris,
    GateNetwork,
    MoodHead,
)
from ..model_fusion.fusion import GMUFusion, GMUNoTanhFusion, SimpleConcatFusion
from .config import TrainLirisConfigHeadSplit
from .heads import SeparateVAHead


_FUSION_MODES = {"gate", "concat", "gmu", "gmu_notanh"}


class AutoEQModelLirisHeadSplit(AutoEQModelLiris):
    """Head-split architecture: V and A predicted by independent MLPs.

    All other components (AudioProjection, MoodHead, fusion modules, and
    augmentation helpers) are byte-identical to their BASE / Phase 2a-5
    counterparts. State_dict for mood_head / audio_projection / gate_network
    / fusion remains compatible with model_fusion — only va_head differs.
    """

    def __init__(self, cfg: TrainLirisConfigHeadSplit):
        if cfg.fusion_mode not in _FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {_FUSION_MODES}, got {cfg.fusion_mode!r}"
            )
        # Skip AutoEQModelLiris.__init__ (which builds joint VAHead) —
        # build our own layer set manually so unused modules are NOT created.
        nn.Module.__init__(self)
        self.cfg = cfg
        self.audio_projection = AudioProjection(cfg)
        self.va_head = SeparateVAHead(cfg)   # <<< ONLY STRUCTURAL DIFFERENCE vs BASE/Phase 2a-5
        self.mood_head = MoodHead(cfg)

        if cfg.fusion_mode == "gate":
            self.gate_network = GateNetwork(cfg)
            self.fusion = None
        elif cfg.fusion_mode == "concat":
            self.gate_network = None
            self.fusion = SimpleConcatFusion()
        elif cfg.fusion_mode == "gmu":
            self.gate_network = None
            self.fusion = GMUFusion(cfg)
        elif cfg.fusion_mode == "gmu_notanh":
            self.gate_network = None
            self.fusion = GMUNoTanhFusion(cfg)

    def forward(self, visual_feat: Tensor, audio_feat: Tensor) -> dict[str, Tensor]:
        a_proj = self.audio_projection(audio_feat)
        v, a = self._modality_dropout(visual_feat, a_proj)
        v, a = self._feature_noise(v, a)

        if self.cfg.fusion_mode == "gate":
            gate = self.gate_network(v, a)
            w_v = gate[:, 0:1]
            w_a = gate[:, 1:2]
            fused = torch.cat([w_v * v, w_a * a], dim=-1)
        else:
            fused, gate = self.fusion(v, a)

        return {
            "va_pred": self.va_head(fused),          # SeparateVAHead → (B, 2) [V, A]
            "mood_logits": self.mood_head(fused),    # (B, 7)
            "gate_weights": gate,                    # (B, 2)
        }
