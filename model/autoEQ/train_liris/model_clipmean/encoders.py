"""Frozen CLIP image encoder + per-frame mean-pool for Phase 2a-4.

Standalone within model_clipmean/ to keep the new dependency localized
(train/encoders.py is not touched). Only used at feature-precompute time —
training itself consumes cached (512,) embeddings via PrecomputedLirisDatasetCLIPMean.

Reference:
    Radford et al., "Learning Transferable Visual Models From Natural Language
    Supervision" (ICML 2021). Checkpoint: openai/clip-vit-base-patch32.

Input normalization note
------------------------
`load_frames_uniform()` in precompute_liris.py applies ImageNet normalization
(mean=[.485,.456,.406], std=[.229,.224,.225]) which is what X-CLIP was trained
with. CLIP ViT-B/32 expects its own normalization
(mean=[.48145466,.4578275,.40821073], std=[.26862954,.26130258,.27577711]).
To keep the frame loader reusable and guarantee identical sampled pixels
across the two variants, the encoder un-normalizes then re-normalizes
internally — downstream the frame loader stays untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPFrameMeanEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 per-frame encoder with mean-pool over time.

    Input  : pixel_values (B, T, 3, H, W) float32, ImageNet-normalized
             (same tensor that XCLIPEncoder consumes).
    Output : video embedding (B, 512) — CLIP image-projection features
             mean-pooled across T frames. Temporal information is DISCARDED
             by construction; this is the ablation axis vs X-CLIP.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPModel  # lazy import

        self.model = CLIPModel.from_pretrained(model_name).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.embed_dim = int(self.model.config.projection_dim)  # 512 for ViT-B/32
        self.register_buffer("imagenet_mean", _IMAGENET_MEAN, persistent=False)
        self.register_buffer("imagenet_std", _IMAGENET_STD, persistent=False)
        self.register_buffer("clip_mean", _CLIP_MEAN, persistent=False)
        self.register_buffer("clip_std", _CLIP_STD, persistent=False)

    def train(self, mode: bool = True) -> "CLIPFrameMeanEncoder":
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.dim() != 5:
            raise ValueError(
                f"expected (B, T, C, H, W), got shape {tuple(pixel_values.shape)}"
            )
        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * T, C, H, W)
        # ImageNet-normalized → raw [0,1] → CLIP-normalized
        raw = flat * self.imagenet_std + self.imagenet_mean
        clip_input = (raw - self.clip_mean) / self.clip_std
        # Explicit two-step call: transformers 5.5.4's get_image_features()
        # returns BaseModelOutputWithPooling (raw vision_model output) without
        # applying visual_projection. Call vision_model + visual_projection
        # manually for version-independent behavior and to make the 768 → 512
        # projection explicit.
        vision_out = self.model.vision_model(pixel_values=clip_input)
        pooled = vision_out.pooler_output                              # (B*T, 768)
        feats = self.model.visual_projection(pooled)                   # (B*T, 512)
        feats = feats.reshape(B, T, -1)
        return feats.mean(dim=1)                                       # (B, 512)
