"""Config for Phase 2a-4 CLIP frame-mean variant.

Inherits ALL Base Model defaults from TrainLirisConfig and overrides only
the visual encoder identity fields + run paths. OAT principle: the single
axis changed is the visual encoder (X-CLIP base-patch32 video Transformer
→ CLIP ViT-B/32 image + per-frame mean-pool). visual_dim stays 512, so the
downstream model architecture is byte-identical to BASE.

Audio encoder (PANNs), audio_raw_dim (2048), and all training hypers
(K=7, va_norm="A", head_dropout=0.3, wd=1e-4, augmentations) are inherited
UNCHANGED so only the visual representation differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainLirisConfig


@dataclass
class TrainLirisConfigCLIPMean(TrainLirisConfig):
    """Phase 2a-4 CLIP frame-mean variant config.

    Overrides vs TrainLirisConfig (Base Model, 2026-04-21 Phase 2a-2 winner):
      feature_file          liris_panns_v5spec → liris_clipmean_v5spec
      run_name / output_dir baseline_2a0 → 2a4_clipmean

    Adds:
      clip_model_name       Huggingface hub id of the frozen CLIP checkpoint
    """

    # --- Visual encoder identity (Phase 2a-4) ---
    # visual_dim stays 512 (CLIP ViT-B/32 projection dim == X-CLIP video dim).
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # --- Feature file path (own directory) ---
    feature_file: str = "data/features/liris_clipmean_v5spec/features.pt"

    # --- Run naming (distinct from BASE) ---
    run_name: str = "2a4_clipmean"
    output_dir: str = "runs/phase2a/2a4_clipmean"
