"""Batch-predict V/A for a list of windows using a trained train_pseudo model.

Reuses the frozen encoders (X-CLIP, PANNs) from model.autoEQ.train — same as
training preprocessing — so features match distribution the model was trained on.

For each Window:
    1. Read 8 frames uniformly inside [start_sec, end_sec] via OpenCV
    2. Read 4s audio slice at 16 kHz mono via soundfile (pre-extracted WAV)
    3. Forward through X-CLIP + PANNs + AutoEQModelCog (eval mode, no augmentation)
    4. Collect va_pred + gate_weights
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch import Tensor

from ..train.encoders import PANNsEncoder, XCLIPEncoder
from ..train.precompute import encode_window_batch
from ..train_pseudo.cognimuse_preprocess import load_frames_from_mp4
from ..train_pseudo.config import TrainCogConfig
from ..train_pseudo.model_base.model import AutoEQModelCog
from .types import Window, WindowVA

NUM_FRAMES = 8
FRAME_SIZE = 224
AUDIO_SR = 16000
WINDOW_SEC = 4


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
    xclip: XCLIPEncoder | None = None,
    panns: PANNsEncoder | None = None,
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
    """
    if not windows:
        return []

    device = device or _auto_device()
    video_path = Path(video_path)
    audio_16k_path = Path(audio_16k_path)

    cfg = TrainCogConfig(num_mood_classes=num_mood_classes)
    if xclip is None:
        xclip = XCLIPEncoder(cfg)
    if panns is None:
        panns = PANNsEncoder(cfg)

    model = AutoEQModelCog(cfg).to(device).eval()
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state)

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
        wave_t = torch.stack(wave_list, dim=0)  # (B, T_audio)
        visual_feat, audio_feat = encode_window_batch(
            frames_t, wave_t, xclip, panns, src_sr=AUDIO_SR
        )
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


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
