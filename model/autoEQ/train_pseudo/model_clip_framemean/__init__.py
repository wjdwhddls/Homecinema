"""CLIP image + frame-mean pooling visual encoder variant.

Paired ablation against ``model_base/`` (X-CLIP base patch32) that holds the
CLIP ViT-B/32 foundation and WIT-400M image-text contrastive pretraining
**fixed**, and removes only X-CLIP's learned video-aware temporal attention
and prompt encoder. Each of the 8 window frames is encoded independently by
CLIP's image tower; the per-frame 512-dim pooler outputs are mean-pooled into
a single 512-dim vector.

This keeps the comparison single-axis (PANNs vs AST 대칭):
    - Foundation:    CLIP ViT-B/32 (identical)
    - Pretraining:   WIT-400M image-text contrastive (identical)
    - Architecture:  image-only + uniform mean  ≠  X-CLIP's video-aware temporal attention
    - Everything else (PANNs CNN14 audio, gated concat fusion, VA/Mood/Gate
      heads, loss, dataset, trainer, LOMO 9-fold protocol) identical.

Isolates the effect of X-CLIP's temporal modeling layer on movie V/A
regression. See ``PLAN.md`` for the implementation roadmap.
"""

from .config import TrainCogConfigClipFrameMean
from .model import AutoEQModelClipFrameMean

__all__ = ["AutoEQModelClipFrameMean", "TrainCogConfigClipFrameMean"]
