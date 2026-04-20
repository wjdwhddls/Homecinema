"""Combined variant: AST audio encoder + GMU fusion.

Stacks two independently-verified axes from the ablation study — AST
(`model_ast`) for the audio side and GMU fusion (`model_gmu`) for the
multimodal aggregation — to measure whether both effects compose when
applied together on top of the σ=OFF baseline. Visual encoder stays X-CLIP
and the mood head remains active (multi-task training).
"""

from .config import TrainCogConfigASTGMU
from .model import AutoEQModelASTGMU

__all__ = ["AutoEQModelASTGMU", "TrainCogConfigASTGMU"]
