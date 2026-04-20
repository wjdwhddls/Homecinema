"""AST + GMU model — GMU architecture parameterised for AST audio dim.

``AutoEQModelGMU.__init__`` already consumes ``config.audio_raw_dim`` when
building ``audio_projection`` and instantiates ``GMUFusion``. Subclassing
with ``TrainCogConfigASTGMU`` (audio_raw_dim=768) is enough — no forward
override needed.
"""

from __future__ import annotations

from ..model_gmu.model import AutoEQModelGMU


class AutoEQModelASTGMU(AutoEQModelGMU):
    """GMU fusion with AST audio features.

    Expected inputs:
        visual_feat: (B, 512)  — frozen X-CLIP
        audio_feat:  (B, 768)  — frozen AST [CLS] embedding
    """


__all__ = ["AutoEQModelASTGMU"]
