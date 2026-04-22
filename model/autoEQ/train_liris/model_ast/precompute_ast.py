"""Phase 2a-3 Audio Encoder ablation — precompute X-CLIP + AST features.

Mirrors `model/autoEQ/train_liris/precompute_liris.py` 1:1, with three axis-of-
comparison changes (V5-FINAL §21 OAT):

    AUDIO_SR               32000 (PANNs)  → 16000 (AST requirement)
    audio encoder          PANNsEncoder   → ASTEncoder (model_ast.encoders)
    feature dict key       "panns"        → "ast"

Visual (X-CLIP, 8-frame uniform @ 224×224) is IDENTICAL so that the only axis
under ablation is the audio encoder.

Output layout
-------------
  data/features/liris_ast_v5spec/
    features.pt   dict[name: str] → {"xclip": Tensor(512,), "ast": Tensor(768,)}
    progress.json running stats, timestamp, device, completed count, crop count

Resumability
------------
  If features.pt exists we load it and skip clips already present.
  A checkpoint save happens every --save-every clips (default 500).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from ..precompute_liris import load_frames_uniform, save_checkpoint
from ...train.config import TrainConfig
from ...train.encoders import XCLIPEncoder
from .encoders import ASTEncoder

# --- Constants ----------------------------------------------------------------
AUDIO_SR = 16000         # AST requirement (vs 32000 for PANNs)
CROP_SEC = 4.0           # §3 window (unchanged)
STRIDE_SEC = 2.0         # §3 stride (unchanged)
PAD_TO_SEC = 10.0        # §9-1 pad_audio_to_10s=auto (unchanged)


# --- Audio --------------------------------------------------------------------


def extract_audio_16k(video_path: Path, out_wav: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vn", "-ar", str(AUDIO_SR), "-ac", "1", str(out_wav)],
        check=True, capture_output=True,
    )


def load_audio_strided_crops(
    video_path: Path,
    crop_sec: float = CROP_SEC,
    stride_sec: float = STRIDE_SEC,
    pad_to_sec: float = PAD_TO_SEC,
) -> torch.Tensor:
    """Load mono 16 kHz audio, zero-pad to pad_to_sec if shorter, then emit
    sliding crops of `crop_sec` seconds at `stride_sec` spacing.

    Returns a tensor of shape (N_crops, crop_samples).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        wav = Path(tmp.name)
        extract_audio_16k(video_path, wav)
        arr, sr = librosa.load(wav, sr=AUDIO_SR, mono=True)
    assert sr == AUDIO_SR

    pad_samples = int(round(pad_to_sec * sr))
    if arr.shape[0] < pad_samples:
        arr = np.pad(arr, (0, pad_samples - arr.shape[0]))

    crop_samples = int(round(crop_sec * sr))
    stride_samples = int(round(stride_sec * sr))
    total = arr.shape[0]
    if total < crop_samples:
        arr = np.pad(arr, (0, crop_samples - total))
        total = arr.shape[0]

    starts: list[int] = []
    s = 0
    while s + crop_samples <= total:
        starts.append(s)
        s += stride_samples
    if not starts:
        starts = [0]

    crops = np.stack([arr[s : s + crop_samples] for s in starts], axis=0)  # (N, T)
    return torch.from_numpy(crops).float()


# --- Main ---------------------------------------------------------------------


def run(
    metadata_csv: Path,
    liris_dir: Path,
    output_dir: Path,
    device: str,
    save_every: int,
    limit: int | None,
    crop_sec: float,
    stride_sec: float,
    pad_to_sec: float,
    ast_model_name: str,
) -> int:
    if device == "mps" and not torch.backends.mps.is_available():
        print("[precompute_ast] MPS not available, falling back to cpu", file=sys.stderr)
        device = "cpu"

    df = pd.read_csv(metadata_csv)
    if limit:
        df = df.head(limit)
    print(f"[precompute_ast] metadata    : {metadata_csv}")
    print(f"[precompute_ast] clips       : {len(df)}")
    print(f"[precompute_ast] device      : {device}")
    print(f"[precompute_ast] output_dir  : {output_dir}")
    print(f"[precompute_ast] ast_model   : {ast_model_name}")
    print(
        f"[precompute_ast] audio       : sr={AUDIO_SR} crop={crop_sec}s "
        f"stride={stride_sec}s pad_to={pad_to_sec}s"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_file = output_dir / "features.pt"
    features: dict = {}
    if feature_file.is_file():
        features = torch.load(feature_file, map_location="cpu", weights_only=False)
        print(f"[precompute_ast] resume from : {feature_file} ({len(features)} clips done)")

    todo = [row for _, row in df.iterrows() if row["name"] not in features]
    print(f"[precompute_ast] remaining   : {len(todo)}")
    if not todo:
        print("[precompute_ast] nothing to do.")
        return 0

    cfg = TrainConfig()
    print("[precompute_ast] loading encoders ...")
    t_load = time.time()
    xclip = XCLIPEncoder(cfg).to(device)
    ast = ASTEncoder(model_name=ast_model_name).to(device)
    xclip.eval()
    ast.eval()
    print(f"[precompute_ast] encoders ready in {time.time()-t_load:.1f}s")

    stats = {
        "device": device,
        "total_clips": int(len(df)),
        "processed": len(features),
        "failed": [],
        "audio_sample_rate_hz": AUDIO_SR,
        "audio_crop_sec": crop_sec,
        "audio_stride_sec": stride_sec,
        "audio_pad_to_sec": pad_to_sec,
        "ast_model_name": ast_model_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": 0.0,
        "total_audio_forwards": 0,
    }
    t_start = time.time()
    last_ck = t_start
    print_every = 25

    for i, row in enumerate(todo):
        name = row["name"]
        clip_path = liris_dir / name
        if not clip_path.is_file():
            stats["failed"].append({"name": name, "reason": "missing file"})
            continue
        try:
            frames = load_frames_uniform(clip_path).to(device)
            audio_crops = load_audio_strided_crops(
                clip_path, crop_sec=crop_sec, stride_sec=stride_sec, pad_to_sec=pad_to_sec
            ).to(device)                                       # (N, T)
            with torch.no_grad():
                v_feat = xclip(frames).squeeze(0).cpu()        # (512,)
                a_feats = ast(audio_crops).cpu()               # (N, 768)
                a_feat = a_feats.mean(dim=0)                   # (768,)  mean-pool
            features[name] = {"xclip": v_feat, "ast": a_feat}
            stats["total_audio_forwards"] += int(audio_crops.size(0))
        except Exception as e:
            stats["failed"].append({"name": name, "reason": str(e)})
            continue

        processed_now = len(features)
        if processed_now % print_every == 0 or i == len(todo) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(todo) - (i + 1)) / max(rate, 1e-6)
            print(
                f"[{processed_now:>5d}/{len(df)}] {name} "
                f"rate={rate:.2f}/s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
            )

        if (time.time() - last_ck) > (save_every if save_every > 60 else save_every * 0.5) \
                and processed_now % save_every == 0:
            stats["processed"] = processed_now
            stats["elapsed_sec"] = round(time.time() - t_start, 1)
            save_checkpoint(features, output_dir, stats)
            last_ck = time.time()
            print(f"[precompute_ast] checkpoint saved ({processed_now}/{len(df)})")

    stats["processed"] = len(features)
    stats["elapsed_sec"] = round(time.time() - t_start, 1)
    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_checkpoint(features, output_dir, stats)
    print(f"\n[precompute_ast] DONE. {len(features)}/{len(df)} clips, "
          f"{len(stats['failed'])} failed, {stats['elapsed_sec']/60:.1f} min, "
          f"{stats['total_audio_forwards']} audio forwards")
    if stats["failed"]:
        print(f"[precompute_ast] failed: {stats['failed'][:5]} ...")
    return 0 if not stats["failed"] else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path,
                   default=Path("dataset/autoEQ/liris/liris_metadata.csv"))
    p.add_argument("--liris-dir", type=Path,
                   default=Path("dataset/autoEQ/liris/data/data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/features/liris_ast_v5spec"))
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--crop-sec", type=float, default=CROP_SEC)
    p.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    p.add_argument("--pad-to-sec", type=float, default=PAD_TO_SEC)
    p.add_argument("--ast-model-name", type=str,
                   default="MIT/ast-finetuned-audioset-10-10-0.4593")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.metadata, args.liris_dir, args.output_dir,
               args.device, args.save_every, args.limit,
               args.crop_sec, args.stride_sec, args.pad_to_sec,
               args.ast_model_name)


if __name__ == "__main__":
    raise SystemExit(main())
