"""Fusion modules for Phase 2a-5.

Three modes under comparison; the `gate` mode reuses the BASE's GateNetwork
directly (via AutoEQModelLiris parent), so only `concat` and `gmu` are
defined here.

All modules consume (v, a_proj) of shape (B, visual_dim), (B, audio_proj_dim)
(both 512-d in BASE) and return:
    fused       — (B, fused_dim=1024)
    gate_weights — (B, 2)  modality-summary for entropy loss compatibility
                           (constant 0.5/0.5 for "concat", mean(z)-based for "gmu")

`fused_dim` is the downstream VAHead/MoodHead input dim (1024 in BASE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import TrainLirisConfig


class SimpleConcatFusion(nn.Module):
    """Null baseline: plain concat, zero learnable fusion params.

    Returns a constant (0.5, 0.5) gate tensor so downstream gate-entropy
    regularizer still has a compatible input. Since the tensor has no
    gradient path back to the model, λ_gate_entropy contributes a
    BATCH-INVARIANT constant to the total loss (does not affect gradients).
    """

    def forward(self, v: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        fused = torch.cat([v, a], dim=-1)  # (B, 1024)
        B = v.size(0)
        gate = torch.full((B, 2), 0.5, dtype=v.dtype, device=v.device)
        return fused, gate


class GMUFusion(nn.Module):
    """Wide GMU (Arevalo et al. 2017 variant, d_out = fused_dim).

    Parameters (BASE: visual=512, audio_proj=512, fused=1024):
        W_v : Linear(512, 1024)   = 525,312
        W_a : Linear(512, 1024)   = 525,312
        W_z : Linear(1024, 1024)  = 1,049,600
        total fusion              = 2,100,224

    Design note: a "narrow" GMU (d_out=512) would shrink the downstream head
    input dim. We keep d_out=1024 so VAHead/MoodHead stay byte-identical to
    BASE — true OAT on heads, fusion the sole varying axis.

    The returned (B, 2) gate_weights = (mean(z), 1 - mean(z)) summarizes the
    per-dim gate as an effective modality weight, compatible with the BASE
    gate-entropy regularizer.
    """

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        d_in_vis = cfg.visual_dim
        d_in_aud = cfg.audio_proj_dim
        d_out = cfg.fused_dim
        d_concat = d_in_vis + d_in_aud
        self.W_v = nn.Linear(d_in_vis, d_out)
        self.W_a = nn.Linear(d_in_aud, d_out)
        self.W_z = nn.Linear(d_concat, d_out)

    def forward(self, v: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        h_v = torch.tanh(self.W_v(v))
        h_a = torch.tanh(self.W_a(a))
        z = torch.sigmoid(self.W_z(torch.cat([v, a], dim=-1)))
        fused = z * h_v + (1.0 - z) * h_a  # (B, fused_dim)
        mean_z = z.mean(dim=-1, keepdim=True)
        gate = torch.cat([mean_z, 1.0 - mean_z], dim=-1)  # (B, 2)
        return fused, gate


class GMUNoTanhFusion(nn.Module):
    """GMU variant with tanh removed on h_v/h_a (Option A after 2026-04-21 diagnosis).

    Motivation: vanilla GMU's tanh bounded fused output to [-1, +1] (GMU fused
    std 0.33 vs BASE fused std 0.43, range [−0.94, 0.94] vs BASE [−7.86, 3.57]).
    Dynamic range compression hurt downstream head capacity. Removing tanh
    restores full output magnitude while keeping the per-dim sigmoid gate —
    the core GMU contribution.

        h_v = W_v · v               # LINEAR (no tanh)
        h_a = W_a · a               # LINEAR (no tanh)
        z   = sigmoid(W_z · [v; a])
        fused = z ⊙ h_v + (1-z) ⊙ h_a

    Same param count as vanilla GMU (2,100,224).
    """

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        d_in_vis = cfg.visual_dim
        d_in_aud = cfg.audio_proj_dim
        d_out = cfg.fused_dim
        d_concat = d_in_vis + d_in_aud
        self.W_v = nn.Linear(d_in_vis, d_out)
        self.W_a = nn.Linear(d_in_aud, d_out)
        self.W_z = nn.Linear(d_concat, d_out)

    def forward(self, v: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        h_v = self.W_v(v)       # (B, fused_dim), linear
        h_a = self.W_a(a)       # (B, fused_dim), linear
        z = torch.sigmoid(self.W_z(torch.cat([v, a], dim=-1)))
        fused = z * h_v + (1.0 - z) * h_a
        mean_z = z.mean(dim=-1, keepdim=True)
        gate = torch.cat([mean_z, 1.0 - mean_z], dim=-1)
        return fused, gate
