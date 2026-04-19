"""Frozen pretrained encoders for MoodEQ.

X-CLIP (video) and PANNs CNN14 (audio) wrappers used at feature
pre-computation time. Kept out of the training graph by design:
parameters are frozen and the modules stay in eval() mode.

SyntheticAutoEQDataset bypasses these entirely by sampling random
feature tensors — encoders here are only used when running against
real video/audio (precompute.py, once the dataset arrives).
"""

from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import torch
import torch.nn as nn
from torch import Tensor

from .config import TrainConfig

_PANNS_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
)
_PANNS_CNN14_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def _download(url: str, target: Path) -> None:
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as r, open(target, "wb") as f:
        shutil.copyfileobj(r, f)


def _ensure_panns_assets() -> str:
    """panns_inference wgets labels + checkpoint at init; macOS has no wget
    and Framework Python has no system CA trust — do it ourselves via certifi.
    """
    panns_dir = Path.home() / "panns_data"
    panns_dir.mkdir(parents=True, exist_ok=True)

    labels = panns_dir / "class_labels_indices.csv"
    if not labels.is_file():
        _download(_PANNS_LABELS_URL, labels)

    checkpoint = panns_dir / "Cnn14_mAP=0.431.pth"
    if not checkpoint.is_file():
        _download(_PANNS_CNN14_URL, checkpoint)
    return str(checkpoint)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


class XCLIPEncoder(nn.Module):
    """Frozen X-CLIP wrapper returning 512-dim pooled video embeddings.

    Input : (B, num_frames, 3, H, W) float tensor, ImageNet-normalized.
    Output: (B, 512) video embedding.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        from transformers import AutoModel  # lazy import

        self.config = config
        self.model = AutoModel.from_pretrained(config.xclip_model)
        _freeze(self)

    def train(self, mode: bool = True) -> "XCLIPEncoder":
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> Tensor:
        if pixel_values.dim() != 5:
            raise ValueError(
                f"expected (B, T, C, H, W), got shape {tuple(pixel_values.shape)}"
            )
        feats = self.model.get_video_features(pixel_values=pixel_values)
        if isinstance(feats, Tensor):
            return feats
        for attr in ("pooler_output", "video_embeds", "last_hidden_state"):
            val = getattr(feats, attr, None)
            if isinstance(val, Tensor):
                return val
        raise RuntimeError(f"could not extract video feature tensor from {type(feats)}")


class PANNsEncoder(nn.Module):
    """Frozen PANNs CNN14 wrapper returning 2048-dim audio embeddings.

    Input : (B, T) float tensor at 16kHz mono, range ~[-1, 1].
    Output: (B, 2048) embedding from CNN14's penultimate layer.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        default_ckpt = _ensure_panns_assets()
        from panns_inference import AudioTagging  # lazy import

        self.config = config
        checkpoint = config.panns_checkpoint or default_ckpt
        self.tagger = AudioTagging(checkpoint_path=checkpoint, device="cpu")
        _freeze(self.tagger.model)
        self.tagger.model.eval()

    def train(self, mode: bool = True) -> "PANNsEncoder":
        super().train(False)
        self.tagger.model.eval()
        return self

    def to(self, *args, **kwargs) -> "PANNsEncoder":
        super().to(*args, **kwargs)
        self.tagger.model.to(*args, **kwargs)
        device = next(self.tagger.model.parameters()).device
        self.tagger.device = device
        return self

    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        if waveform.dim() != 2:
            raise ValueError(
                f"expected (B, T) or (B, 1, T), got shape {tuple(waveform.shape)}"
            )
        device = next(self.tagger.model.parameters()).device
        waveform = waveform.to(device)
        _, embedding = self.tagger.inference(waveform)
        if isinstance(embedding, torch.Tensor):
            return embedding
        return torch.as_tensor(embedding, device=device)
