"""AutoEQModelLirisAST — Base Model architecture with AST feature dim (768).

AudioProjection in the parent AutoEQModelLiris uses `mid = cfg.audio_raw_dim // 2`,
so the Base Model's "2-layer MLP + LN + GELU + Dropout" pattern is preserved
automatically when audio_raw_dim=768 (routing: 768 → 384 → 512).

Parameter footprint (Base Model vs AST variant):
    PANNs (BASE):  2048 → 1024 → 512,  AudioProjection params = 2,626,048
    AST:            768 →  384 → 512,  AudioProjection params =   492,032  (−81%)

Gate / VAHead / MoodHead blocks are inherited unchanged — no variant-specific overrides.
"""

from __future__ import annotations

from ..model import AutoEQModelLiris
from .config import TrainLirisConfigAST


class AutoEQModelLirisAST(AutoEQModelLiris):
    """Alias subclass parameterized for AST (audio_raw_dim=768).

    Kept distinct from parent for two reasons:
      1. ablation logs unambiguously tag the variant
      2. future AST-specific overrides can land here without touching BASE
    """

    def __init__(self, cfg: TrainLirisConfigAST):
        super().__init__(cfg)
