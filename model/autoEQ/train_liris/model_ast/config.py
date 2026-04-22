"""Config for Phase 2a-3 AST variant.

Inherits ALL Base Model defaults from TrainLirisConfig and overrides only
the audio encoder identity fields + run paths. OAT principle: single axis
changed (audio_raw_dim 2048 → 768), everything else identical.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainLirisConfig


@dataclass
class TrainLirisConfigAST(TrainLirisConfig):
    """Phase 2a-3 AST variant config.

    Overrides vs TrainLirisConfig (Base Model, 2026-04-21 Phase 2a-2 winner):
      audio_raw_dim         2048 → 768   (AST [CLS] embedding dim)
      feature_file          liris_panns_v5spec → liris_ast_v5spec
      run_name / output_dir baseline_2a0 → 2a3_ast

    Adds:
      ast_model_name        Huggingface hub id of the frozen AST checkpoint
      audio_sample_rate_hz  16000 (AST required sample rate)
    """

    # --- Audio encoder identity (Phase 2a-3) ---
    audio_raw_dim: int = 768
    ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    audio_sample_rate_hz: int = 16000

    # --- Feature file path (own directory) ---
    feature_file: str = "data/features/liris_ast_v5spec/features.pt"

    # --- Run naming (distinct from BASE) ---
    run_name: str = "2a3_ast"
    output_dir: str = "runs/phase2a/2a3_ast"
