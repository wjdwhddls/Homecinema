"""AST ablation config — subclasses TrainCogConfig, overrides audio fields.

Matches spec V3.3 §4-3 baseline layout (X-CLIP video + gated concat fusion +
VA / Mood / Gate heads) with only the audio encoder swapped from PANNs CNN14
(2048-dim) to AST (768-dim, MIT/ast-finetuned-audioset-10-10-0.4593).
"""

from dataclasses import dataclass

from ..config import TrainCogConfig


@dataclass
class TrainCogConfigAST(TrainCogConfig):
    # --- Audio encoder (AST, replaces PANNs in model_base) ---
    audio_raw_dim: int = 768                                           # AST [CLS] embedding
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"    # HF hub id
    # AST's default input is 10s @ 16 kHz → 1024 frames @ 10 ms hop.
    # Our analysis windows are 4s, so max_length is still the AST default
    # (shorter inputs are padded by ASTFeatureExtractor).
    ast_max_length: int = 1024

    # Wandb — distinguish the two ablation tracks by project name if enabled.
    wandb_project: str = "moodeq_cog_ast"
