"""TrainLirisConfig — FROZEN BASE MODEL (2026-04-21).

This file defines THE base model used for all subsequent Phase 2a / 2b
ablations and the final Phase 3 evaluation. DO NOT change these defaults
mid-experiment — any change invalidates downstream comparisons.

Deviations from V5-FINAL §9-1 line 237 (spec) are intentional engineering
fixes applied on 2026-04-21 after diagnosing the spec architecture:

  * head_dropout           0.0   → 0.3      (observed output-layer overfit)
  * weight_decay           1e-5  → 1e-4     (stronger global L2)
  * use_full_learning_set  False → True     (LIRIS paper protocol, 64/16/80)
  * num_mood_classes       4     → 7        (Phase 2a-2 winner, +0.014 CCC, p<0.05)

Model class changes (see model.py):
  * AudioProjection: Linear(2048→512) →
        Linear(2048→1024) + LayerNorm + GELU + Dropout(0.1) + Linear(1024→512) + LayerNorm
  * VA/Mood Head hidden: ReLU → LayerNorm + ReLU

Total parameters: 1.84M → 3.42M (+86%).
Measured overfit gap@best: 0.201 → 0.045 (−77%).

All other fields track V5-FINAL §9-1 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainLirisConfig:
    # --- Feature dimensions ---
    visual_dim: int = 512
    audio_raw_dim: int = 2048
    audio_proj_dim: int = 512
    fused_dim: int = 1024

    # --- Task heads ---
    # BASE 2026-04-21 (Phase 2a-2 winner):
    #   §8 strict K=7 gate FAIL on Strategy A (JA=0%), but Phase 2a-2 measured
    #   K=7 outperforming K=4 by +0.014 CCC (paired t=6.14, p<0.05) due to
    #   richer multi-task mood signal (§21 pragmatic model selection).
    #   Strict §8 readers can revert via `--num-mood-classes 4`.
    num_mood_classes: int = 7
    gate_hidden_dim: int = 256
    head_hidden_dim: int = 256

    # --- Optimizer (§9-1 line 237) ---
    lr: float = 1e-4
    weight_decay: float = 1e-4                  # BASE 2026-04-21: spec 1e-5 → 1e-4
    batch_size: int = 32                        # §9-1: batch_size=32
    epochs: int = 40                            # §9-1: epochs=40
    warmup_steps: int = 500                     # §9-1: warmup=500
    use_cosine_schedule: bool = True            # linear warmup + cosine decay
    grad_clip_norm: float = 1.0                 # §9-1: grad_clip=1.0

    # --- Regularization ---
    # BASE 2026-04-21: spec had head_dropout=0.0 (implicit); we use 0.3 after
    # diagnosing output-layer overfit. Dropout is applied between head hidden
    # LayerNorm+ReLU and final Linear.
    head_dropout: float = 0.3

    # --- Loss weights (§9-1) ---
    lambda_va: float = 1.0
    lambda_mood: float = 0.3                    # §9-1: λ_mood=0.3
    lambda_gate_entropy: float = 0.05           # §9-1: λ_gate_entropy=0.05

    # --- CCC hybrid loss (§9-1) ---
    use_ccc_loss: bool = True
    ccc_loss_weight: float = 0.3                # §9-1: ccc_hybrid_w=0.3

    # --- Augmentations (§9-1) ---
    modality_dropout_p: float = 0.05            # §9-1: modality_dropout_p=0.05
    feature_noise_std: float = 0.03             # §9-1: feature_noise_std=0.03
    mixup_prob: float = 0.5                     # §9-1: mixup_prob=0.5
    mixup_alpha: float = 0.4                    # §9-1: mixup_alpha=0.4
    # Target Shrinkage (per-axis p75 AND, ε=0.05) — §9-1 / §2-2 실측
    target_shrinkage_eps: float = 0.05
    v_var_threshold: float = 0.117
    a_var_threshold: float = 0.164
    shrinkage_logic: str = "AND"

    # --- Early stopping (§9-1) ---
    # Lexicographic (mean_CCC, mean_Pearson, -mean_MAE), patience=10
    early_stop_patience: int = 10

    # --- EMA (off per spec; exposed for ablation only) ---
    use_ema: bool = False
    ema_decay: float = 0.999

    # --- Overfit monitor (§10-2) ---
    overfit_gap_threshold: float = 0.10

    # --- Data / split ---
    # V5-FINAL §7 step 6 + spec-compliant audio precompute (stride=2s, pad_to=10s)
    feature_file: str = "data/features/liris_panns_v5spec/features.pt"
    metadata_csv: str = "dataset/autoEQ/liris/liris_metadata.csv"
    use_official_split: bool = True             # §9-1: LIRIS 40/40/80 films
    # BASE 2026-04-21: enable LIRIS paper protocol (Baveye 2015) — merge
    # learning+validation → 80 films, carve out 16 films for early-stop val
    # (deterministic via split_seed). Addresses V5-FINAL §15 risk #7.
    use_full_learning_set: bool = True

    # --- Audio precompute parameters (§3 + §9-1; used by liris_preprocess) ---
    audio_crop_sec: float = 4.0                 # §6-1 sanity + §3 window
    audio_stride_sec: float = 2.0               # §3 line 119 training stride
    audio_pad_to_sec: float = 10.0              # §9-1 pad_audio_to_10s=auto

    # --- V/A normalization strategy (Phase 2a-1) ---
    # "A": (v_raw - 3) / 2  (V5-FINAL §2-1, default — V3.2 원설계)
    # "B": per-axis min-max stretch using train split → v_norm, a_norm ∈ [-1, +1]
    #      Triggers mood_k7 / quadrant_k4 recomputation.
    va_norm_strategy: str = "A"

    # --- Misc (§9-1) ---
    seed: int = 42                              # §9-1: seed=42
    # Training defaults to CPU — features are precomputed and the head is only
    # 1.84 M params, so CPU is fast (<1s/epoch) and avoids MPS backward-pass
    # NaN issues observed with this architecture. MPS kept as an opt-in.
    device: str = "cpu"                         # "cpu" | "mps" | "auto"
    num_workers: int = 0                        # macOS fork safety

    # --- Logging / outputs ---
    run_name: str = "baseline_2a0"
    output_dir: str = "runs/phase2a/baseline_2a0"
    use_wandb: bool = False
    wandb_project: str = "moodeq_liris"
    wandb_run_name: str = ""

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
