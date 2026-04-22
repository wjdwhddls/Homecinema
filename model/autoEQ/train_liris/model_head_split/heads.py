"""SeparateVAHead — independent prediction paths for Valence and Arousal.

Replaces the BASE's joint VAHead (single 1024 → 256 → 2 MLP that outputs
[V, A] together) with two fully independent MLPs (one for V, one for A).
Each head has its own Linear + LayerNorm + ReLU + Dropout + Linear stack.

Output interface is preserved: forward() returns a (B, 2) tensor with
column 0 = V prediction, column 1 = A prediction — bit-compatible with
downstream loss code and `PrecomputedLirisDataset` which constructs
`va_target = [v_norm, a_norm]`.

Parameter accounting (hidden_dim=256, fused_dim=1024):
    per head:  Linear(1024, 256)  = 262,400
               LayerNorm(256)     =     512
               Linear(256, 1)     =     257
               ─────────────────────────────
               subtotal           = 263,169
    SeparateVAHead total: 2 × 263,169 = 526,338
    vs joint VAHead:     263,426
    delta (per-head vs joint): +262,912

Rationale (why each head keeps hidden_dim=256 rather than halving to 128):
    The joint head's hidden_dim=256 is a shared representation branched to 2
    outputs. Each task (V or A) effectively uses the full 256 implicitly. In
    the separate design we give each path its own 256-dim hidden space so
    the per-task representational capacity matches the joint case. This
    trades +262K params for clean per-head capacity (param increase ~7.7%
    of BASE 3.42M).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import TrainLirisConfig


class SeparateVAHead(nn.Module):
    """Two independent (1024 → 256 → 1) heads, concatenated at output."""

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        self.v_head = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        self.a_head = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward(self, fused: Tensor) -> Tensor:
        v = self.v_head(fused)  # (B, 1)
        a = self.a_head(fused)  # (B, 1)
        return torch.cat([v, a], dim=-1)  # (B, 2) — [V, A], matches BASE interface
