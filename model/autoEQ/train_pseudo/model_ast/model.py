"""AST ablation model — thin wrapper around AutoEQModelCog.

The whole fused tower (AudioProjection, GateNetworkCog, VAHeadCog, MoodHeadCog,
modality dropout, feature noise) is dimension-parametric via ``TrainCogConfig``.
Swapping PANNs → AST only changes ``audio_raw_dim`` (2048 → 768), which the
``AudioProjectionCog`` Linear layer consumes. Nothing else differs, so we
subclass instead of duplicating ~150 lines of model code.

If AST-specific architectural tweaks are needed later (e.g. attention pooling
over AST token sequence instead of [CLS]), override ``forward`` here.
"""

from __future__ import annotations

from ..model_base.model import AutoEQModelCog


class AutoEQModelAST(AutoEQModelCog):
    """Same architecture as AutoEQModelCog, parameterised for AST feature dim.

    Expected inputs (unchanged from baseline):
        visual_feat: (B, 512)  — from frozen X-CLIP
        audio_feat:  (B, 768)  — from frozen AST [CLS] embedding
    """


__all__ = ["AutoEQModelAST"]
