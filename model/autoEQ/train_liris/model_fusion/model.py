"""AutoEQModelLirisFusion — Base Model with swappable fusion stage.

Preserves AudioProjection / VAHead / MoodHead byte-identical to BASE.
Only the fusion stage changes based on `cfg.fusion_mode`.

Under `fusion_mode="gate"`, this class is state-dict compatible with
`AutoEQModelLiris(TrainLirisConfig())` — i.e. the same weights can be
loaded interchangeably and produce identical forward outputs. This is
guaranteed by `test_fusion_compliance.py::test_gate_mode_byte_identical`.
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
    VAHead,
)
from .config import TrainLirisConfigFusion
from .fusion import GMUFusion, GMUNoTanhFusion, SimpleConcatFusion


_FUSION_MODES = {"gate", "concat", "gmu", "gmu_notanh"}


class AutoEQModelLirisFusion(AutoEQModelLiris):
    """BASE architecture with configurable fusion stage.

    NOTE: we manually call `nn.Module.__init__` (skipping parent's
    `AutoEQModelLiris.__init__`) so that for non-gate modes the unused
    GateNetwork parameters are never created — keeps the trainable param
    count clean and prevents weight-decay from touching dead weights.
    """

    def __init__(self, cfg: TrainLirisConfigFusion):
        if cfg.fusion_mode not in _FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {_FUSION_MODES}, got {cfg.fusion_mode!r}"
            )
        nn.Module.__init__(self)
        self.cfg = cfg
        self.audio_projection = AudioProjection(cfg)
        self.va_head = VAHead(cfg)
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
            "va_pred": self.va_head(fused),
            "mood_logits": self.mood_head(fused),
            "gate_weights": gate,
        }
