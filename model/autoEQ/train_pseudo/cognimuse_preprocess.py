"""CogniMuse raw → feature .pt pipeline.

Independent I/O (does NOT call the `NotImplementedError` stubs in
`train/precompute.py`). Only reuses tensor-level helpers from there.

Assumed directory layout (adjust at implementation time if real files differ):

    <cognimuse_dir>/
    ├── BMI/
    │   ├── video.mp4            # video + embedded audio
    │   ├── experienced/
    │   │   ├── valence.txt      # 40 ms spaced floats, one per line
    │   │   └── arousal.txt
    │   └── intended/ (same layout)
    ├── CHI/ CRA/ DEP/ FNE/ GLA/ LOR/  (same layout)

Annotation sampling rate: 25 Hz (40 ms). Raw range assumed `[-1, 1]`;
if the real dump uses Likert [1, 7] or similar, redefine `_normalize_va`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..train.encoders import PANNsEncoder, XCLIPEncoder
from ..train.precompute import (
    encode_window_batch,
    resample_for_panns,  # re-exported for clarity (used inside encode_window_batch)
    save_features,
    split_into_windows,
)
from .config import TrainCogConfig
from .dataset import COGNIMUSE_MOVIES


ANNOTATION_SAMPLING_RATE_HZ = 25.0  # CogniMuse: 40 ms per frame
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---- V/A normalization -----------------------------------------------------


def _normalize_va(
    raw: np.ndarray, src_range: tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """Map raw V/A values to [-1, 1] via linear transform.

    v_norm = 2 · (raw − a) / (b − a) − 1

    If src_range is already (-1, 1), returns as float32 unchanged.
    Asserts that < 0.1% of normalized samples fall outside [-1, 1]
    (guards against silent unit-of-measurement bugs).
    """
    a, b = src_range
    if (a, b) == (-1.0, 1.0):
        return raw.astype(np.float32)
    normed = 2.0 * (raw - a) / (b - a) - 1.0
    oob_pct = float((np.abs(normed) > 1.0).mean())
    assert oob_pct < 0.001, (
        f"V/A normalize: {oob_pct:.3%} samples out of [-1, 1] "
        f"(src_range={src_range}) — check dataset units."
    )
    return normed.astype(np.float32)


# ---- Annotation loader -----------------------------------------------------


def load_va_annotation(
    movie_dir: Path, annotation_source: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load 40 ms-spaced V/A streams.

    Returns (valence, arousal, sampling_rate_hz). `annotation_source` ∈
    {'experienced', 'intended', 'mean'}. 'mean' averages experienced and
    intended after aligning to the shorter length.
    """
    if annotation_source in {"experienced", "intended"}:
        v_path = movie_dir / annotation_source / "valence.txt"
        a_path = movie_dir / annotation_source / "arousal.txt"
        v_raw = np.loadtxt(v_path)
        a_raw = np.loadtxt(a_path)
        return (
            _normalize_va(v_raw),
            _normalize_va(a_raw),
            ANNOTATION_SAMPLING_RATE_HZ,
        )
    if annotation_source == "mean":
        v_exp, a_exp, sr = load_va_annotation(movie_dir, "experienced")
        v_int, a_int, _ = load_va_annotation(movie_dir, "intended")
        L = min(len(v_exp), len(v_int))
        return (
            0.5 * (v_exp[:L] + v_int[:L]),
            0.5 * (a_exp[:L] + a_int[:L]),
            sr,
        )
    raise ValueError(f"Unknown annotation_source: {annotation_source}")


def aggregate_window_va(
    v_arr: np.ndarray,
    a_arr: np.ndarray,
    sr_hz: float,
    t0: float,
    t1: float,
) -> tuple[float, float, float, float]:
    """Mean + std of the V/A samples falling inside [t0, t1].

    Returns (mean_v, mean_a, std_v, std_a). Clamps slice to available range.
    """
    idx_start = max(0, int(round(t0 * sr_hz)))
    idx_end = min(len(v_arr), int(round(t1 * sr_hz)))
    if idx_end <= idx_start:
        return 0.0, 0.0, 0.0, 0.0
    v_slice = v_arr[idx_start:idx_end]
    a_slice = a_arr[idx_start:idx_end]
    return (
        float(v_slice.mean()),
        float(a_slice.mean()),
        float(v_slice.std()),
        float(a_slice.std()),
    )


# ---- MP4 I/O ---------------------------------------------------------------


def get_video_duration(video_path: Path) -> float:
    """Return duration in seconds via torchvision.io metadata."""
    from torchvision.io import VideoReader  # lazy

    reader = VideoReader(str(video_path), "video")
    metadata = reader.get_metadata()
    return float(metadata["video"]["duration"][0])


def load_frames_from_mp4(
    video_path: Path,
    timestamps: list[float],
    frame_size: int = 224,
) -> Tensor:
    """Load frames at the given seconds and return ImageNet-normalized tensor.

    Uses OpenCV for cross-torchvision-version compatibility (torchvision >=0.16
    removed `torchvision.io.read_video`). Matches the approach used in
    `pseudo_label/layer1_visual.py:sample_frames`.

    Returns: (num_frames, 3, frame_size, frame_size) float32 in normalized
    ImageNet space (required by X-CLIP).
    """
    import cv2
    from torchvision.transforms.functional import resize

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read frames from {video_path}")

    frames_out: list[Tensor] = []
    last_good: Tensor | None = None
    for ts in timestamps:
        idx = int(round(float(ts) * fps))
        idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            if last_good is not None:
                frames_out.append(last_good)
                continue
            cap.release()
            raise RuntimeError(f"frame read failed at ts={ts}s in {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame = resize(frame, [frame_size, frame_size], antialias=True)
        frames_out.append(frame)
        last_good = frame
    cap.release()

    frames_t = torch.stack(frames_out, dim=0)  # (num_frames, C, H, W)
    # ImageNet normalize
    frames_t = (frames_t - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
    return frames_t


def load_audio_from_mp4(
    video_path: Path,
    t0: float,
    t1: float,
    target_sr: int = 16000,
) -> Tensor:
    """Load mono audio slice [t0, t1] at target_sr. Returns (T,) float32."""
    from torchvision.io import read_video
    import torchaudio.functional as AF

    video, audio, info = read_video(
        str(video_path), start_pts=t0, end_pts=t1, pts_unit="sec"
    )
    # audio: (channels, T) at info['audio_fps']
    src_sr = int(info.get("audio_fps", 48000))
    if audio.dim() == 2 and audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)  # -> mono (1, T)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.float()
    if src_sr != target_sr:
        audio = AF.resample(audio, orig_freq=src_sr, new_freq=target_sr)
    # Trim / pad to exact length
    expected = int(round((t1 - t0) * target_sr))
    T = audio.size(-1)
    if T > expected:
        audio = audio[..., :expected]
    elif T < expected:
        pad = torch.zeros(1, expected - T, dtype=audio.dtype)
        audio = torch.cat([audio, pad], dim=-1)
    return audio.squeeze(0)


def compute_manifest_sha(cognimuse_dir: Path) -> dict[str, str]:
    """SHA-256 digest of each movie's video.mp4 — determinism witness."""
    out: dict[str, str] = {}
    for code in COGNIMUSE_MOVIES:
        mp4 = cognimuse_dir / code / "video.mp4"
        if not mp4.is_file():
            out[code] = "MISSING"
            continue
        h = hashlib.sha256()
        with open(mp4, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[code] = h.hexdigest()
    return out


# ---- Main preprocess loop --------------------------------------------------


def preprocess_cognimuse(
    cognimuse_dir: Path,
    output_dir: Path,
    annotation_source: str = "experienced",
    window_sec: int = 4,
    stride_sec: int = 2,
    num_frames: int = 8,
    audio_sr: int = 16000,
    xclip: XCLIPEncoder | None = None,
    panns: PANNsEncoder | None = None,
    batch_size: int = 16,
    config: TrainCogConfig | None = None,
) -> dict:
    """Run 7-movie preprocess → {cognimuse_visual,audio,metadata}.pt + manifest.

    Returns the manifest dict that was written to disk.
    """
    cognimuse_dir = Path(cognimuse_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or TrainCogConfig()
    if xclip is None:
        xclip = XCLIPEncoder(cfg)
    if panns is None:
        panns = PANNsEncoder(cfg)

    visual_dict: dict[str, Tensor] = {}
    audio_dict: dict[str, Tensor] = {}
    metadata_dict: dict[str, dict] = {}
    manifest: dict = {
        "annotation_source": annotation_source,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "num_frames": num_frames,
        "audio_sr": audio_sr,
        "movies": {},
        "file_sha": compute_manifest_sha(cognimuse_dir),
        "cognimuse_movies_constant": COGNIMUSE_MOVIES,
    }

    for movie_code in COGNIMUSE_MOVIES:
        movie_id = COGNIMUSE_MOVIES.index(movie_code)
        movie_dir = cognimuse_dir / movie_code
        video_path = movie_dir / "video.mp4"

        v_arr, a_arr, sr_hz = load_va_annotation(movie_dir, annotation_source)
        duration = get_video_duration(video_path)
        windows = split_into_windows(duration, window_sec, stride_sec)

        # Batched encoder calls for speed
        batch_frames: list[Tensor] = []
        batch_wave: list[Tensor] = []
        batch_wids: list[str] = []
        batch_meta: list[dict] = []

        def _flush_batch() -> None:
            if not batch_frames:
                return
            frames_t = torch.stack(batch_frames, dim=0)  # (B, T, C, H, W)
            wave_t = torch.stack(batch_wave, dim=0)  # (B, T_audio)
            visual, audio = encode_window_batch(
                frames_t, wave_t, xclip, panns, src_sr=audio_sr
            )
            for i, wid in enumerate(batch_wids):
                visual_dict[wid] = visual[i].detach().cpu()
                audio_dict[wid] = audio[i].detach().cpu()
                metadata_dict[wid] = batch_meta[i]
            batch_frames.clear()
            batch_wave.clear()
            batch_wids.clear()
            batch_meta.clear()

        for idx, (t0, t1) in enumerate(windows):
            mean_v, mean_a, std_v, std_a = aggregate_window_va(
                v_arr, a_arr, sr_hz, t0, t1
            )
            # 8 frames uniformly inside [t0, t1]
            frame_ts = [t0 + (t1 - t0) * (i + 0.5) / num_frames for i in range(num_frames)]
            frames = load_frames_from_mp4(video_path, frame_ts)
            waveform = load_audio_from_mp4(video_path, t0, t1, target_sr=audio_sr)

            wid = f"{movie_code}_{idx:05d}"
            batch_frames.append(frames)
            batch_wave.append(waveform)
            batch_wids.append(wid)
            batch_meta.append(
                {
                    "movie_id": movie_id,
                    "movie_code": movie_code,
                    "valence": mean_v,
                    "arousal": mean_a,
                    "valence_std": std_v,
                    "arousal_std": std_a,
                    "t0": float(t0),
                    "t1": float(t1),
                    "annotation_source": annotation_source,
                }
            )
            if len(batch_frames) >= batch_size:
                _flush_batch()

        _flush_batch()

        manifest["movies"][movie_code] = {
            "window_count": sum(
                1 for wid in metadata_dict if metadata_dict[wid]["movie_code"] == movie_code
            ),
            "duration_sec": duration,
        }

    save_features(visual_dict, audio_dict, metadata_dict, output_dir, "cognimuse")
    with open(output_dir / "cognimuse_preprocess_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---- CLI -------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CogniMuse preprocess → feature .pt")
    p.add_argument("--cognimuse_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--annotation",
        type=str,
        default="experienced",
        choices=["experienced", "intended", "mean"],
    )
    p.add_argument("--window_sec", type=int, default=4)
    p.add_argument("--stride_sec", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = TrainCogConfig(
        cognimuse_dir=str(args.cognimuse_dir),
        cognimuse_annotation=args.annotation,
        cognimuse_window_sec=args.window_sec,
        cognimuse_stride_sec=args.stride_sec,
    )
    preprocess_cognimuse(
        cognimuse_dir=args.cognimuse_dir,
        output_dir=args.output_dir,
        annotation_source=args.annotation,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        batch_size=args.batch_size,
        config=cfg,
    )


if __name__ == "__main__":
    main()
