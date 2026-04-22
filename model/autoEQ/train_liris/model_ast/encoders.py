"""Frozen Audio Spectrogram Transformer (AST) wrapper for Phase 2a-3.

Standalone within model_ast/ to keep the AST dependency localized
(train/encoders.py is not touched). Only used at feature-precompute time —
training itself consumes cached (768,) embeddings via PrecomputedLirisDatasetAST.

Reference:
    Gong et al., "AST: Audio Spectrogram Transformer" (Interspeech 2021).
    Pretrained on AudioSet: MIT/ast-finetuned-audioset-10-10-0.4593
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ASTEncoder(nn.Module):
    """Frozen AST wrapper.

    Input  : waveform (B, T) float32 @ 16 kHz mono, range ~[-1, 1]
    Output : CLS embedding (B, 768)
    """

    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        from transformers import ASTFeatureExtractor, ASTModel  # lazy import
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.cls_dim = 768
        self.sample_rate = 16000

    @torch.no_grad()
    def forward(self, waveform: Tensor) -> Tensor:
        # ASTFeatureExtractor expects numpy / list inputs → batch loop.
        device = waveform.device
        outs: list[Tensor] = []
        for i in range(waveform.size(0)):
            inputs = self.feature_extractor(
                waveform[i].detach().cpu().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            iv = inputs["input_values"].to(device)
            out = self.model(iv)
            cls = out.last_hidden_state[:, 0, :]   # (1, 768)
            outs.append(cls)
        return torch.cat(outs, dim=0)              # (B, 768)
