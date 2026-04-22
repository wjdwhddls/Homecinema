"""Batch-predict V/A for a list of windows using a trained model.

Supports these training variants via the ``variant`` dispatch key:
    - ``base``       : train_pseudo + X-CLIP + PANNs(2048) + Gated Concat (V3.2)
    - ``gmu``        : train_pseudo + X-CLIP + PANNs(2048) + GMU fusion (V3.3)
    - ``ast_gmu``    : train_pseudo + X-CLIP + AST(768) + GMU fusion (V3.3)
    - ``liris_base`` : train_liris  + X-CLIP + PANNs(2048) + Gate + K=7 + A-norm
                      (Phase 2a-7 BASE Model FROZEN; requires num_mood_classes=7)

Reuses the frozen encoders from model.autoEQ.train for the PANNs branch; for the
AST branch it instantiates ``transformers.ASTModel`` inline (matching
``scripts/precompute_ast_features.py``) so features match the distribution the
model was trained on.

For each Window:
    1. Read 8 frames uniformly inside [start_sec, end_sec] via OpenCV
    2. Read 4s audio slice at 16 kHz mono via soundfile (pre-extracted WAV)
    3. Forward through encoders + model (eval mode, no augmentation)
    4. Collect va_pred + gate_weights (GMU returns a 2-dim scalar summary
       of its element-wise z; baseline returns scalar softmax weights)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch import Tensor

from ..train.encoders import PANNsEncoder, XCLIPEncoder
from ..train.precompute import resample_for_panns
from ..train_pseudo.cognimuse_preprocess import load_frames_from_mp4
from .types import Window, WindowVA

NUM_FRAMES = 8
FRAME_SIZE = 224
AUDIO_SR = 16000
WINDOW_SEC = 4

# Variant dispatch: maps variant key → (config class, model class, audio encoder kind).
# Config/model classes are resolved lazily via importlib to avoid importing
# heavy training modules when they are not needed.
VARIANTS: dict[str, dict[str, str]] = {
    "base": {
        "cfg_module": "model.autoEQ.train_pseudo.config",
        "cfg_cls":    "TrainCogConfig",
        "model_module": "model.autoEQ.train_pseudo.model_base.model",
        "model_cls":  "AutoEQModelCog",
        "audio_encoder": "panns",
    },
    "gmu": {
        "cfg_module": "model.autoEQ.train_pseudo.model_gmu.config",
        "cfg_cls":    "TrainCogConfigGMU",
        "model_module": "model.autoEQ.train_pseudo.model_gmu.model",
        "model_cls":  "AutoEQModelGMU",
        "audio_encoder": "panns",
    },
    "ast_gmu": {
        "cfg_module": "model.autoEQ.train_pseudo.model_ast_gmu.config",
        "cfg_cls":    "TrainCogConfigASTGMU",
        "model_module": "model.autoEQ.train_pseudo.model_ast_gmu.model",
        "model_cls":  "AutoEQModelASTGMU",
        "audio_encoder": "ast",
    },
    "liris_base": {
        "cfg_module": "model.autoEQ.train_liris.config",
        "cfg_cls":    "TrainLirisConfig",
        "model_module": "model.autoEQ.train_liris.model",
        "model_cls":  "AutoEQModelLiris",
        "audio_encoder": "panns",
    },
}


def _load_variant(variant: str) -> tuple[type, type, str]:
    """Resolve a variant key → (config_cls, model_cls, audio_encoder_kind)."""
    import importlib

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}'; expected one of {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    cfg_cls = getattr(importlib.import_module(spec["cfg_module"]), spec["cfg_cls"])
    model_cls = getattr(importlib.import_module(spec["model_module"]), spec["model_cls"])
    return cfg_cls, model_cls, spec["audio_encoder"]


class _ASTInferenceEncoder:
    """Inline AST encoder for inference — mirrors scripts/precompute_ast_features.py.

    Takes (B, T) 16 kHz waveforms → returns (B, 768) [CLS] embeddings on CPU
    (model itself lives on ``device``). This avoids pulling transformers into
    module-level imports when the caller only needs the PANNs path.
    """

    def __init__(self, model_name: str, device: torch.device):
        from transformers import ASTFeatureExtractor, ASTModel

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).eval().to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, waveforms: Tensor) -> Tensor:
        # waveforms: (B, T) 16 kHz float32
        wavs_np = [w.cpu().numpy() for w in waveforms]
        inputs = self.feature_extractor(
            wavs_np,
            sampling_rate=AUDIO_SR,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        out = self.model(input_values)
        return out.last_hidden_state[:, 0, :]  # (B, 768)


def _sample_frame_times(start_sec: float, end_sec: float,
                        num_frames: int = NUM_FRAMES) -> list[float]:
    """8 uniformly-spaced frame timestamps inside [start, end]."""
    span = end_sec - start_sec
    return [start_sec + span * (i + 0.5) / num_frames for i in range(num_frames)]


def _load_audio_slice(
    audio_16k_path: Path,
    start_sec: float,
    end_sec: float,
    target_sr: int = AUDIO_SR,
) -> Tensor:
    """Read [start, end] from a 16k-mono WAV, pad/trim to exactly 4s.

    Assumes audio_16k_path was produced by vad.extract_audio_16k_mono (so sr
    matches target_sr).
    """
    data, sr = sf.read(str(audio_16k_path), dtype="float32", always_2d=False)
    if sr != target_sr:
        # resample if mismatch (should not happen if pipeline uses extracted WAV)
        t = torch.from_numpy(data).float().unsqueeze(0)
        t = AF.resample(t, orig_freq=sr, new_freq=target_sr)
        data = t.squeeze(0).numpy()
    if data.ndim > 1:
        data = data.mean(axis=-1)
    total_samples = data.shape[0]
    s = int(round(max(0.0, start_sec) * target_sr))
    e = int(round(end_sec * target_sr))
    s = min(s, total_samples)
    e = min(max(e, s), total_samples)
    slice_np = data[s:e]
    expected = target_sr * WINDOW_SEC
    t = torch.from_numpy(slice_np).float()
    if t.numel() > expected:
        t = t[:expected]
    elif t.numel() < expected:
        pad = torch.zeros(expected - t.numel(), dtype=t.dtype)
        t = torch.cat([t, pad], dim=0)
    return t  # (T,)


def _batched(iterable: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def predict_windows(
    windows: list[Window],
    video_path: str | Path,
    audio_16k_path: str | Path,
    ckpt_path: str | Path,
    device: torch.device | None = None,
    batch_size: int = 16,
    num_mood_classes: int = 4,
    variant: str = "base",
    xclip: XCLIPEncoder | None = None,
    panns: PANNsEncoder | None = None,
    ast_encoder: _ASTInferenceEncoder | None = None,
) -> list[WindowVA]:
    """Run model inference over every Window; return aligned WindowVA list.

    Args:
        windows: scene-sliced windows.
        video_path: original video (frames extracted on-the-fly).
        audio_16k_path: extracted 16 kHz mono WAV (see vad.extract_audio_16k_mono).
        ckpt_path: trained model state_dict (.pt).
        device: torch device (defaults to cuda/mps/cpu auto).
        batch_size: windows per forward pass.
        num_mood_classes: must match training config (K=4 default).
        variant: one of ``base`` / ``gmu`` / ``ast_gmu`` — must match ckpt origin.
        xclip / panns / ast_encoder: optional pre-instantiated encoders (reuse across calls).
    """
    if not windows:
        return []

    device = device or _auto_device()
    video_path = Path(video_path)
    audio_16k_path = Path(audio_16k_path)

    cfg_cls, model_cls, audio_encoder_kind = _load_variant(variant)
    cfg = cfg_cls(num_mood_classes=num_mood_classes)

    # Visual encoder (X-CLIP) is common to all variants.
    if xclip is None:
        xclip = XCLIPEncoder(cfg)

    # Audio encoder dispatch — PANNs (2048-d) for base/gmu, AST (768-d) for ast_gmu.
    if audio_encoder_kind == "panns":
        if panns is None:
            panns = PANNsEncoder(cfg)
        audio_callable = _make_panns_audio_callable(panns)
    elif audio_encoder_kind == "ast":
        if ast_encoder is None:
            ast_model_name = getattr(cfg, "ast_model_name", "MIT/ast-finetuned-audioset-10-10-0.4593")
            ast_encoder = _ASTInferenceEncoder(ast_model_name, device=device)
        audio_callable = ast_encoder
    else:
        raise ValueError(f"unknown audio_encoder_kind '{audio_encoder_kind}'")

    model = model_cls(cfg).to(device).eval()
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    # train_liris ckpt wraps state_dict as {"epoch", "model", "cfg", "val_metrics"};
    # train_pseudo ckpt is a bare state_dict. Accept either.
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)

    results: list[WindowVA] = []
    for batch in _batched(windows, batch_size):
        frames_list: list[Tensor] = []
        wave_list: list[Tensor] = []
        for w in batch:
            ts = _sample_frame_times(w.start_sec, w.end_sec)
            frames = load_frames_from_mp4(video_path, ts, frame_size=FRAME_SIZE)
            wave = _load_audio_slice(audio_16k_path, w.start_sec, w.end_sec)
            frames_list.append(frames)
            wave_list.append(wave)
        frames_t = torch.stack(frames_list, dim=0)  # (B, T, C, H, W)
        wave_t = torch.stack(wave_list, dim=0)      # (B, T_audio) at 16 kHz

        visual_feat = xclip(frames_t)               # (B, 512)
        audio_feat = audio_callable(wave_t)         # (B, 2048) or (B, 768)

        visual_feat = visual_feat.to(device)
        audio_feat = audio_feat.to(device)
        with torch.no_grad():
            out = model(visual_feat, audio_feat)
        va_pred = out["va_pred"].cpu()
        gate = out["gate_weights"].cpu()
        for i, w in enumerate(batch):
            results.append(WindowVA(
                scene_idx=w.scene_idx,
                window_idx_in_scene=w.window_idx_in_scene,
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                valence=float(va_pred[i, 0].item()),
                arousal=float(va_pred[i, 1].item()),
                gate_w_v=float(gate[i, 0].item()),
                gate_w_a=float(gate[i, 1].item()),
            ))
    return results


def _make_panns_audio_callable(panns: PANNsEncoder):
    """Wrap PANNs so the predict_windows loop has a uniform (B, T) → (B, d) call."""
    def _call(wave_t: Tensor) -> Tensor:
        wav_for_panns = resample_for_panns(wave_t, src_sr=AUDIO_SR)
        return panns(wav_for_panns)
    return _call


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
