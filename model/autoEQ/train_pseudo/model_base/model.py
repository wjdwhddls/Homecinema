import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import TrainCogConfig


class AudioProjectionCog(nn.Module):
    """PANNs 2048-dim -> 512-dim linear projection (trainable)."""

    def __init__(self, config: TrainCogConfig):
        super().__init__()
        self.proj = nn.Linear(config.audio_raw_dim, config.audio_proj_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class GateNetworkCog(nn.Module):
    """2-layer MLP producing modality weights (w_v, w_a) summing to 1."""

    def __init__(self, config: TrainCogConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.fused_dim, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, 2),
        )

    def forward(self, v: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([v, a], dim=-1)  # (B, 1024)
        logits = self.mlp(x)
        return F.softmax(logits, dim=-1)


class VAHeadCog(nn.Module):
    """V/A regression head."""

    def __init__(self, config: TrainCogConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.fused_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, 2),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.mlp(fused)


class MoodHeadCog(nn.Module):
    """Mood classification head. num_classes may be 7 (GEMS) or 4 (quadrant)
    depending on Phase 0 outcome.
    """

    def __init__(self, config: TrainCogConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.fused_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.head_hidden_dim, config.num_mood_classes),
        )

    def forward(self, fused: Tensor) -> Tensor:
        return self.mlp(fused)


class AutoEQModelCog(nn.Module):
    """CogniMuse variant of AutoEQModel.

    Differences from train.AutoEQModel:
      - No CongruenceHead, no cong_logits output
      - forward() takes only (visual_feat, audio_feat); no cong_label
      - modality_dropout is applied to every sample with probability p
        (was gated by cong_label==0 previously)
      - No cong-noise injection on |v-a|
    """

    def __init__(self, config: TrainCogConfig):
        super().__init__()
        self.config = config
        self.audio_projection = AudioProjectionCog(config)
        self.gate_network = GateNetworkCog(config)
        self.va_head = VAHeadCog(config)
        self.mood_head = MoodHeadCog(config)

    def _apply_modality_dropout(
        self,
        v: Tensor,
        a: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply modality dropout to every sample with probability p.

        No cong_label dependency. Picks visual-or-audio per sample and zeroes
        it out when the dropout trigger fires. Only active in training mode.
        """
        if not self.training:
            return v, a

        p = self.config.modality_dropout_p
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

    def _apply_feature_noise(
        self,
        v: Tensor,
        a: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Symmetric Gaussian on both modalities. σ identical keeps gate balanced.
        # Dropped-to-zero samples bypass noise so modality-dropout signal survives.
        if not self.training:
            return v, a
        std = self.config.feature_noise_std
        if std <= 0.0:
            return v, a
        v_live = (v.abs().sum(dim=-1, keepdim=True) > 0).float()
        a_live = (a.abs().sum(dim=-1, keepdim=True) > 0).float()
        v = v + torch.randn_like(v) * std * v_live
        a = a + torch.randn_like(a) * std * a_live
        return v, a

    def forward(
        self,
        visual_feat: Tensor,
        audio_feat: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            visual_feat: (B, 512) from frozen X-CLIP
            audio_feat: (B, 2048) from frozen PANNs CNN14
        Returns:
            dict with keys: va_pred, mood_logits, gate_weights
        """
        a_proj = self.audio_projection(audio_feat)  # (B, 512)
        v, a = self._apply_modality_dropout(visual_feat, a_proj)
        v, a = self._apply_feature_noise(v, a)

        gate_weights = self.gate_network(v, a)  # (B, 2)
        w_v = gate_weights[:, 0:1]
        w_a = gate_weights[:, 1:2]

        fused = torch.cat([w_v * v, w_a * a], dim=-1)  # (B, 1024)

        va_pred = self.va_head(fused)
        mood_logits = self.mood_head(fused)

        return {
            "va_pred": va_pred,
            "mood_logits": mood_logits,
            "gate_weights": gate_weights,
        }
