"""Feature pre-computation scaffold.

Walks a dataset of clips, splits each into 4-second windows, runs frozen
X-CLIP (video) and PANNs (audio) encoders, and saves resulting feature
tensors as .pt files keyed by window id.

The dataset-I/O surface (`load_clip_frames`, `load_clip_audio`,
`precompute_dataset`) is intentionally left as NotImplementedError stubs
until LIRIS-ACCEDE arrives; everything that operates on tensors is fully
implemented and covered by test_precompute.py.

Sample-rate contract: the project stores audio at 16 kHz
(TrainConfig.audio_sr) but PANNs CNN14 was trained at 32 kHz, so this
module resamples 16 → 32 kHz at the PANNs boundary via
torchaudio.functional.resample. Encoders themselves stay sample-rate
agnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio.functional as AF
from torch import Tensor

from .config import TrainConfig
from .encoders import PANNsEncoder, XCLIPEncoder


def split_into_windows(
    clip_duration_sec: float,
    window_sec: int = 4,
    stride_sec: int = 2,
) -> list[tuple[float, float]]:
    """Return (start, end) pairs covering the clip in fixed windows.

    Last window is dropped if it would exceed the clip boundary
    (spec 2-4: no zero-padded tails). Examples:
        8s  -> [(0,4), (2,6), (4,8)]
        10s -> [(0,4), (2,6), (4,8), (6,10)]
        12s -> [(0,4), (2,6), (4,8), (6,10), (8,12)]
    """
    if clip_duration_sec < window_sec:
        return []
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start + window_sec <= clip_duration_sec + 1e-9:
        windows.append((float(start), float(start + window_sec)))
        start += stride_sec
    return windows


def resample_for_panns(
    waveform: Tensor,
    src_sr: int,
    target_sr: int = 32000,
) -> Tensor:
    """Resample to the PANNs training rate. No-op if already at target.

    Accepts (B, T) or (B, 1, T); returns the same rank.
    """
    if src_sr == target_sr:
        return waveform
    squeezed = False
    if waveform.dim() == 3 and waveform.size(1) == 1:
        waveform = waveform.squeeze(1)
        squeezed = True
    out = AF.resample(waveform, orig_freq=src_sr, new_freq=target_sr)
    if squeezed:
        out = out.unsqueeze(1)
    return out


def encode_window_batch(
    frames: Tensor,
    waveform: Tensor,
    xclip: XCLIPEncoder,
    panns: PANNsEncoder,
    src_sr: int = 16000,
) -> tuple[Tensor, Tensor]:
    """Run both encoders on a batch of windows.

    Args:
        frames: (B, num_frames, 3, H, W) ImageNet-normalized video frames.
        waveform: (B, T) or (B, 1, T) audio at `src_sr` Hz.
        xclip / panns: pretrained frozen encoders.
        src_sr: sample rate of `waveform`; resampled to 32 kHz for PANNs.

    Returns:
        (visual_feat: (B, 512), audio_feat: (B, 2048))
    """
    visual = xclip(frames)
    wav_for_panns = resample_for_panns(waveform, src_sr=src_sr)
    audio = panns(wav_for_panns)
    return visual, audio


def save_features(
    visual: dict[str, Tensor],
    audio: dict[str, Tensor],
    metadata: dict[str, dict],
    output_dir: Path,
    split_name: str,
) -> None:
    """Persist precomputed features to three .pt files under `output_dir`.

    Files:
      {split}_visual.pt    -> {window_id: Tensor(512)}
      {split}_audio.pt     -> {window_id: Tensor(2048)}
      {split}_metadata.pt  -> {window_id: {clip_id, movie_id, valence, arousal, start, end}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(visual, output_dir / f"{split_name}_visual.pt")
    torch.save(audio, output_dir / f"{split_name}_audio.pt")
    torch.save(metadata, output_dir / f"{split_name}_metadata.pt")


def load_clip_frames(path: Path, timestamps_sec: list[float]) -> Tensor:
    """TODO (post-data-arrival): decode the video at `path` and return
    ImageNet-normalized (num_frames, 3, 224, 224) tensor sampled at the
    requested timestamps. Implement with torchvision.io.read_video or decord.
    """
    raise NotImplementedError(
        "load_clip_frames is a stub; fill in once LIRIS-ACCEDE arrives."
    )


def load_clip_audio(path: Path, start_sec: float, end_sec: float) -> Tensor:
    """TODO (post-data-arrival): load the audio slice [start_sec, end_sec]
    as mono 16 kHz tensor of shape (T,). Implement with torchaudio.load.
    """
    raise NotImplementedError(
        "load_clip_audio is a stub; fill in once LIRIS-ACCEDE arrives."
    )


def precompute_dataset(
    manifest_path: Path,
    output_dir: Path,
    xclip: XCLIPEncoder,
    panns: PANNsEncoder,
    split_name: str,
    batch_size: int = 16,
) -> None:
    """TODO (post-data-arrival): iterate the manifest, call
    `encode_window_batch` per batch, accumulate dicts, then `save_features`.
    """
    raise NotImplementedError(
        "precompute_dataset is a stub; fill in once LIRIS-ACCEDE arrives."
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MoodEQ feature pre-computation (scaffold; data-arrival stub)",
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--split_name", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = TrainConfig()
    xclip = XCLIPEncoder(cfg)
    panns = PANNsEncoder(cfg)
    precompute_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        xclip=xclip,
        panns=panns,
        split_name=args.split_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
