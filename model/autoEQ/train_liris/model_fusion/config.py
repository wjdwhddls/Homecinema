"""Config for Phase 2a-5 Fusion mechanism ablation.

Inherits ALL Base Model defaults from TrainLirisConfig and adds a single
`fusion_mode` switch. OAT principle: the sole axis under comparison is the
visual-audio fusion stage:

    "gate"   — BASE: softmax MLP → per-sample (w_v, w_a) → concat[w_v·v, w_a·a]
    "concat" — null baseline: plain concat[v, a]
    "gmu"    — wide GMU (Arevalo et al. 2017 variant, d_out=fused_dim=1024):
                 h_v = tanh(W_v · v)
                 h_a = tanh(W_a · a)
                 z   = sigmoid(W_z · [v; a])
                 fused = z ⊙ h_v + (1-z) ⊙ h_a

All three variants output a 1024-d fused vector so downstream VA/Mood heads
stay byte-identical to BASE. Gate (the BASE) reuses the inherited
`GateNetwork`/fusion path unchanged; the "fusion_mode=gate" variant MUST be
bit-equivalent to `AutoEQModelLiris(TrainLirisConfig())` under matched weights.

Feature file is BASE's `liris_panns_v5spec/features.pt` — NO precompute needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TrainLirisConfig


@dataclass
class TrainLirisConfigFusion(TrainLirisConfig):
    """Phase 2a-5 Fusion variant config.

    Overrides vs TrainLirisConfig (Base Model, 2026-04-21 Phase 2a-4):
      run_name / output_dir — variant-specific

    Adds:
      fusion_mode: {"gate", "concat", "gmu"}
    """

    fusion_mode: str = "gate"

    # Run naming (distinct from BASE). Specific runs override to
    # 2a5_concat_sN / 2a5_gmu_sN via CLI.
    run_name: str = "2a5_fusion"
    output_dir: str = "runs/phase2a/2a5_fusion"
