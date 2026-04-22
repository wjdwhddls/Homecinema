"""Config for Phase 2a-6 Head Structure ablation.

Inherits everything from TrainLirisConfigFusion (Phase 2a-5) — including the
`fusion_mode` field — and only overrides run naming. Head structure is
determined by subpackage identity (AutoEQModelLirisHeadSplit always builds
SeparateVAHead).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model_fusion.config import TrainLirisConfigFusion


@dataclass
class TrainLirisConfigHeadSplit(TrainLirisConfigFusion):
    """Phase 2a-6 Head-split variant.

    Overrides vs TrainLirisConfigFusion:
      run_name / output_dir — variant-specific (CLI may further customize
                              to '2a6_split_{fusion}_s{seed}').

    Inherits from TrainLirisConfigFusion:
      fusion_mode ∈ {"gate" (default), "concat", "gmu", "gmu_notanh"}

    Inherits from TrainLirisConfig (BASE):
      ALL hyperparameters (K=7, va_norm='A', head_dropout=0.3, wd=1e-4, etc.)
    """

    run_name: str = "2a6_head_split"
    output_dir: str = "runs/phase2a/2a6_head_split"
