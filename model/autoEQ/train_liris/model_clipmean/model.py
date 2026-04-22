"""AutoEQModelLirisCLIPMean — Base Model architecture with identical dims.

Unlike the AST variant (which changes audio_raw_dim 2048 → 768 and thus
reshapes AudioProjection), the CLIP frame-mean variant keeps visual_dim=512
and audio_raw_dim=2048 unchanged. The entire model graph (AudioProjection,
GateNetwork, VAHead, MoodHead) is byte-identical to BASE, so this class is
essentially a tagging alias — only the upstream frozen visual representation
differs.

This is the purest form of OAT: no trainable-parameter count difference
between BASE and the CLIPMean variant (both = 3,417,099 params). Any CCC
delta observed is attributable solely to the visual encoder's inductive bias.
"""

from __future__ import annotations

from ..model import AutoEQModelLiris
from .config import TrainLirisConfigCLIPMean


class AutoEQModelLirisCLIPMean(AutoEQModelLiris):
    """Alias subclass parameterized for the CLIP frame-mean variant.

    Kept distinct from parent for two reasons:
      1. ablation logs unambiguously tag the variant
      2. future CLIPMean-specific overrides can land here without touching BASE
    """

    def __init__(self, cfg: TrainLirisConfigCLIPMean):
        super().__init__(cfg)
