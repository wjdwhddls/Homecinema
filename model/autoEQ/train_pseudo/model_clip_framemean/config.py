"""CLIP frame-mean ablation config — subclasses TrainCogConfig.

The CLIP ViT-B/32 image encoder's ``pooler_output`` is 512-dim, identical to
X-CLIP's video embedding dim. This means the fused tower architecture
(``visual_dim=512``, ``fused_dim=1024``) and all gate/head layers are
**unchanged** from the baseline — the ablation is isolated to what feature
tensor lands at the input.
"""

from dataclasses import dataclass

from ..config import TrainCogConfig


@dataclass
class TrainCogConfigClipFrameMean(TrainCogConfig):
    # --- Visual encoder: CLIP image + frame-mean, replaces X-CLIP video tower ---
    # visual_dim stays 512 (same as X-CLIP), so fused_dim=1024 and no
    # projection layer is needed. Only the feature extractor differs.
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_num_frames: int = 8      # match baseline num_frames for fair comparison
    clip_pool: str = "frame_mean" # mean over per-frame pooler_output (ViT [CLS])

    # Wandb — distinguish the ablation tracks by project name if enabled.
    wandb_project: str = "moodeq_cog_clipimg"
