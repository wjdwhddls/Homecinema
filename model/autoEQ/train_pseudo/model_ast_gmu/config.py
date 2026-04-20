"""AST + GMU ablation config — AST audio dim (768) + GMU fusion dims inherited.

Extends ``TrainCogConfigGMU`` (which already sets ``fused_dim=512`` and
disables gate-entropy loss) and only overrides ``audio_raw_dim`` so
``AudioProjectionCog`` builds a 768 → 512 Linear matching AST [CLS] features.
"""

from dataclasses import dataclass

from ..model_gmu.config import TrainCogConfigGMU


@dataclass
class TrainCogConfigASTGMU(TrainCogConfigGMU):
    audio_raw_dim: int = 768  # AST [CLS] embedding dim
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_max_length: int = 1024

    wandb_project: str = "moodeq_cog_ast_gmu"
