"""AST (Audio Spectrogram Transformer, Gong 2021) audio encoder variant.

Paired ablation against ``model_base/`` (PANNs CNN14) to isolate the effect of
CNN vs Transformer audio backbones. Everything else (video backbone, gated
concat fusion, heads, loss, dataset, trainer, LOMO 9-fold protocol) is
identical. See ``PLAN.md`` for the implementation roadmap.
"""

from .config import TrainCogConfigAST
from .model import AutoEQModelAST

__all__ = ["AutoEQModelAST", "TrainCogConfigAST"]
