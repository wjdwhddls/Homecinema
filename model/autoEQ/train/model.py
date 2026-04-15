import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TrainConfig


class AudioProjection(nn.Module):
    """PANNs 2048-dim -> 512-dim linear projection (trainable)."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.proj = nn.Linear(config.audio_raw_dim, config.audio_proj_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class GateNetwork(nn.Module):
    """2-layer MLP producing modality weights (w_v, w_a) that sum to 1."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.fused_dim, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, 2),
        )

    def forward(self, v: Tensor, a: Tensor) -> Tensor:
        """
        Args:
            v: (B, 512) visual features
            a: (B, 512) projected audio features
        Returns:
            (B, 2) gate weights, summing to 1 per row
        """
        x = torch.cat([v, a], dim=-1)  # (B, 1024)
        logits = self.mlp(x)  # (B, 2)
        return F.softmax(logits, dim=-1)


class VAHead(nn.Module):
    """V/A regression head: predicts (valence, arousal)."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.fused_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, 2),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.mlp(fused)


class MoodHead(nn.Module):
    """Mood classification head: 7 GEMS categories."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.fused_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, config.num_mood_classes),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.mlp(fused)


class CongruenceHead(nn.Module):
    """Congruence classification head: 3 classes (congruent/slight/strong).

    Input: L2-normalized |v - a| (512-dim) + gate weights (2-dim) = 514-dim.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.cong_head_input_dim, config.cong_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.cong_head_hidden_dim, config.num_cong_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class AutoEQModel(nn.Module):
    """Full trainable model for AutoEQ mood analysis.

    Frozen encoders (X-CLIP, PANNs) are external; this model receives
    pre-computed features and handles projection, gating, fusion, and heads.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.audio_projection = AudioProjection(config)
        self.gate_network = GateNetwork(config)
        self.va_head = VAHead(config)
        self.mood_head = MoodHead(config)
        self.cong_head = CongruenceHead(config)

    def _apply_modality_dropout(
        self,
        v: Tensor,
        a: Tensor,
        cong_label: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply modality dropout only to congruent samples during training.

        With probability p, zero out either visual or audio features
        (never both). Only applied when cong_label == 0.
        """
        if not self.training:
            return v, a

        p = self.config.modality_dropout_p
        B = v.size(0)

        # Mask: which samples are congruent
        congruent_mask = (cong_label == 0)  # (B,)

        # Random decision per sample: drop visual (0) or audio (1)
        drop_choice = torch.randint(0, 2, (B,), device=v.device)

        # Random trigger: whether to apply dropout at all
        drop_trigger = (torch.rand(B, device=v.device) < p)

        # Only apply to congruent samples that triggered
        should_drop = congruent_mask & drop_trigger  # (B,)

        # Build masks
        drop_visual = should_drop & (drop_choice == 0)  # (B,)
        drop_audio = should_drop & (drop_choice == 1)  # (B,)

        # Apply (unsqueeze for broadcasting over feature dim)
        v = v.clone()
        a = a.clone()
        v[drop_visual] = 0.0
        a[drop_audio] = 0.0

        return v, a

    def forward(
        self,
        visual_feat: Tensor,
        audio_feat: Tensor,
        cong_label: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Args:
            visual_feat: (B, 512) from frozen X-CLIP
            audio_feat: (B, 2048) from frozen PANNs CNN14
            cong_label: (B,) congruence labels, needed for modality dropout

        Returns:
            dict with keys: va_pred, mood_logits, cong_logits, gate_weights
        """
        # Audio projection: 2048 -> 512
        a_proj = self.audio_projection(audio_feat)  # (B, 512)

        # Modality dropout (training only, congruent samples only)
        if cong_label is not None:
            v, a = self._apply_modality_dropout(visual_feat, a_proj, cong_label)
        else:
            v, a = visual_feat, a_proj

        # Gate network
        gate_weights = self.gate_network(v, a)  # (B, 2)
        w_v = gate_weights[:, 0:1]  # (B, 1)
        w_a = gate_weights[:, 1:2]  # (B, 1)

        # Intermediate fusion
        fused = torch.cat([w_v * v, w_a * a], dim=-1)  # (B, 1024)

        # V/A and Mood heads
        va_pred = self.va_head(fused)  # (B, 2)
        mood_logits = self.mood_head(fused)  # (B, 7)

        # Congruence head: L2-normalized |v - a| + gate weights
        diff = torch.abs(
            F.normalize(visual_feat, p=2, dim=-1)
            - F.normalize(a_proj, p=2, dim=-1)
        )  # (B, 512)

        if self.training:
            noise = torch.randn_like(diff) * self.config.cong_noise_std
            diff = diff + noise

        cong_input = torch.cat([diff, gate_weights], dim=-1)  # (B, 514)
        cong_logits = self.cong_head(cong_input)  # (B, 3)

        return {
            "va_pred": va_pred,
            "mood_logits": mood_logits,
            "cong_logits": cong_logits,
            "gate_weights": gate_weights,
        }
