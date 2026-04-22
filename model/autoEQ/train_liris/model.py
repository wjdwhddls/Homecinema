"""AutoEQModelLiris — X-CLIP + PANNs + Gate + Intermediate Fusion + VA + Mood Head.

V5-FINAL §3 with engineering fixes (2026-04-21, per overfit diagnosis):
  * AudioProjection: Linear(2048→512) → 2-layer MLP with LayerNorm + GELU.
    Original single Linear stayed near init (1.02× L2 growth in baseline);
    the MLP + Norm pattern enables stable gradient flow through the largest
    block (57% of params).
  * VAHead / MoodHead: LayerNorm added between hidden Linear and activation
    to keep hidden-layer activations well-scaled, so the hidden weights can
    actually learn (baseline showed 1.01-1.04× growth — near-frozen).

See runs/phase2a/overfit_location_diagnostic.json for the evidence that
motivated these changes.

Gate network and fusion operation itself are unchanged — they work well.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TrainLirisConfig


class AudioProjection(nn.Module):
    """PANNs 2048-d → 512-d via 2-layer MLP with LayerNorm + GELU.

    Parameters:
        original spec: Linear(2048→512)               = 1,049,088
        enhanced:      (2048→1024) + LN + GELU + Drop + (1024→512) + LN
                                                      = 2,626,048
    """

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        mid = cfg.audio_raw_dim // 2  # 1024
        self.net = nn.Sequential(
            nn.Linear(cfg.audio_raw_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid, cfg.audio_proj_dim),
            nn.LayerNorm(cfg.audio_proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GateNetwork(nn.Module):
    """Unchanged: concat([v, a_proj]) → MLP → softmax (w_v, w_a)."""

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.gate_hidden_dim, 2),
        )

    def forward(self, v: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([v, a], dim=-1)
        return F.softmax(self.mlp(x), dim=-1)


class VAHead(nn.Module):
    """1024 → 256 → 2 with LayerNorm on hidden activation."""

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, 2),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.mlp(fused)


class MoodHead(nn.Module):
    """1024 → 256 → K with LayerNorm on hidden activation."""

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.fused_dim, cfg.head_hidden_dim),
            nn.LayerNorm(cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.num_mood_classes),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.mlp(fused)


class AutoEQModelLiris(nn.Module):
    """V3.2 baseline + 2026-04-21 engineering fixes (see module docstring).

    Inputs are precomputed X-CLIP (512) + PANNs (2048) features.
    """

    def __init__(self, cfg: TrainLirisConfig):
        super().__init__()
        self.cfg = cfg
        self.audio_projection = AudioProjection(cfg)
        self.gate_network = GateNetwork(cfg)
        self.va_head = VAHead(cfg)
        self.mood_head = MoodHead(cfg)

    def _modality_dropout(self, v: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training or self.cfg.modality_dropout_p <= 0:
            return v, a
        p = self.cfg.modality_dropout_p
        B = v.size(0)
        drop_choice = torch.randint(0, 2, (B,), device=v.device)
        drop_trigger = torch.rand(B, device=v.device) < p
        drop_visual = drop_trigger & (drop_choice == 0)
        drop_audio = drop_trigger & (drop_choice == 1)
        v = v.clone()
        a = a.clone()
        v[drop_visual] = 0.0
        a[drop_audio] = 0.0
        return v, a

    def _feature_noise(self, v: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training or self.cfg.feature_noise_std <= 0:
            return v, a
        std = self.cfg.feature_noise_std
        v_live = (v.abs().sum(dim=-1, keepdim=True) > 0).float()
        a_live = (a.abs().sum(dim=-1, keepdim=True) > 0).float()
        v = v + torch.randn_like(v) * std * v_live
        a = a + torch.randn_like(a) * std * a_live
        return v, a

    def forward(self, visual_feat: Tensor, audio_feat: Tensor) -> dict[str, Tensor]:
        """
        Args:
            visual_feat: (B, 512)  X-CLIP embedding.
            audio_feat:  (B, 2048) PANNs CNN14 embedding.
        Returns dict: va_pred (B,2), mood_logits (B,K), gate_weights (B,2)
        """
        a_proj = self.audio_projection(audio_feat)
        v, a = self._modality_dropout(visual_feat, a_proj)
        v, a = self._feature_noise(v, a)

        gate = self.gate_network(v, a)
        w_v = gate[:, 0:1]
        w_a = gate[:, 1:2]
        fused = torch.cat([w_v * v, w_a * a], dim=-1)

        return {
            "va_pred": self.va_head(fused),
            "mood_logits": self.mood_head(fused),
            "gate_weights": gate,
        }
