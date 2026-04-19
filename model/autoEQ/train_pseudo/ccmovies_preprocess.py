"""CCMovies → feature .pt pipeline for MoodEQ pseudo-labeled dataset.

Reads `dataset/autoEQ/CCMovies/labels/final_labels.csv` + per-window media at
`dataset/autoEQ/CCMovies/windows/<film_id>/<window_id>.{mp4,wav}` and emits
three tensors consumable by `PrecomputedCogDataset`:

    <output_dir>/<split_name>_visual.pt      — dict[window_id, Tensor(512,)]
    <output_dir>/<split_name>_audio.pt       — dict[window_id, Tensor(2048,)]
    <output_dir>/<split_name>_metadata.pt    — dict[window_id, dict]

Filtering rules:
  - Keep rows where source ∈ {auto_agreement, gemini_only}
  - Drop rows with NaN in final_v / final_a
  (disagreement / excluded → removed)

Column remapping to the schema PrecomputedCogDataset expects:
  final_v         → valence
  final_a         → arousal
  ensemble_std_v  → valence_std
  ensemble_std_a  → arousal_std

Usage:
    python -m model.autoEQ.train_pseudo.ccmovies_preprocess \\
        --labels_csv  dataset/autoEQ/CCMovies/labels/final_labels.csv \\
        --windows_dir dataset/autoEQ/CCMovies/windows \\
        --output_dir  data/features/ccmovies
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from ..train.encoders import PANNsEncoder, XCLIPEncoder
from ..train.precompute import encode_window_batch
from .cognimuse_preprocess import load_frames_from_mp4
from .config import TrainCogConfig


CCMOVIES: list[str] = [
    "agent_327",
    "big_buck_bunny",
    "caminandes_3",
    "cosmos_laundromat",
    "elephants_dream",
    "sintel",
    "spring",
    "tears_of_steel",
    "valkaama_highlight",
]
assert CCMOVIES == sorted(CCMOVIES), (
    "CCMOVIES must stay alphabetical to keep movie_id stable across runs"
)

VALID_SOURCES: set[str] = {"auto_agreement", "gemini_only"}
WINDOW_SEC: int = 4


def load_wav(wav_path: Path, target_sr: int = 16000) -> Tensor:
    """Load a .wav → mono 1-D tensor of length target_sr * WINDOW_SEC.

    Uses `soundfile` to avoid torchaudio's new TorchCodec dependency
    (torchaudio ≥2.9 requires TorchCodec for .load()). Resampling via
    torchaudio.functional.resample which is pure-tensor (no codec needed).
    """
    import soundfile as sf
    import torchaudio.functional as AF

    data, src_sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(data).T  # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if src_sr != target_sr:
        wav = AF.resample(wav, orig_freq=src_sr, new_freq=target_sr)
    expected = target_sr * WINDOW_SEC
    T = wav.size(-1)
    if T > expected:
        wav = wav[..., :expected]
    elif T < expected:
        wav = torch.cat(
            [wav, torch.zeros(1, expected - T, dtype=wav.dtype)], dim=-1
        )
    return wav.squeeze(0)


def preprocess_ccmovies(
    labels_csv: Path,
    windows_dir: Path,
    output_dir: Path,
    split_name: str = "ccmovies",
    num_frames: int = 8,
    frame_size: int = 224,
    audio_sr: int = 16000,
    xclip: XCLIPEncoder | None = None,
    panns: PANNsEncoder | None = None,
    batch_size: int = 16,
    config: TrainCogConfig | None = None,
) -> dict:
    """Run CCMovies preprocess → {<split>_visual,audio,metadata}.pt + manifest.

    Returns the manifest dict.
    """
    labels_csv = Path(labels_csv)
    windows_dir = Path(windows_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or TrainCogConfig()
    if xclip is None:
        xclip = XCLIPEncoder(cfg)
    if panns is None:
        panns = PANNsEncoder(cfg)

    df = pd.read_csv(labels_csv)
    before = len(df)
    df = df[df["source"].isin(VALID_SOURCES)].copy()
    df = df.dropna(subset=["final_v", "final_a"])
    after = len(df)
    print(f"[info] filtered {before} → {after} rows "
          f"(kept source∈{sorted(VALID_SOURCES)}, dropped NaN V/A)")

    visual_dict: dict[str, Tensor] = {}
    audio_dict: dict[str, Tensor] = {}
    metadata_dict: dict[str, dict] = {}
    per_film_counts: dict[str, int] = {}
    per_split_counts: dict[str, int] = {}

    frame_ts = [WINDOW_SEC * (i + 0.5) / num_frames for i in range(num_frames)]

    batch_frames: list[Tensor] = []
    batch_wave: list[Tensor] = []
    batch_wids: list[str] = []
    batch_meta: list[dict] = []

    def _flush() -> None:
        if not batch_frames:
            return
        frames_t = torch.stack(batch_frames, dim=0)
        wave_t = torch.stack(batch_wave, dim=0)
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

    for _, row in df.iterrows():
        film_id = str(row["film_id"])
        window_id = str(row["window_id"])
        if film_id not in CCMOVIES:
            print(f"[warn] unknown film_id {film_id}; skipped")
            continue
        mp4 = windows_dir / film_id / f"{window_id}.mp4"
        wav = windows_dir / film_id / f"{window_id}.wav"
        if not mp4.is_file() or not wav.is_file():
            print(f"[warn] missing media {film_id}/{window_id}; skipped")
            continue
        frames = load_frames_from_mp4(mp4, frame_ts, frame_size=frame_size)
        waveform = load_wav(wav, target_sr=audio_sr)

        batch_frames.append(frames)
        batch_wave.append(waveform)
        batch_wids.append(window_id)
        has_human = bool(row.get("has_human_label", False))
        batch_meta.append({
            "movie_id": CCMOVIES.index(film_id),
            "movie_code": film_id,
            "valence": float(row["final_v"]),
            "arousal": float(row["final_a"]),
            "valence_std": float(row.get("ensemble_std_v") or 0.0),
            "arousal_std": float(row.get("ensemble_std_a") or 0.0),
            "t0": float(row["t0"]),
            "t1": float(row["t1"]),
            "source": str(row["source"]),
            "split": str(row["split"]),
            "confidence": float(row.get("confidence") or 1.0),
            "weight": float(row.get("weight") or 1.0),
            "has_human_label": has_human,
            "human_v": float(row["human_v"]) if has_human else float("nan"),
            "human_a": float(row["human_a"]) if has_human else float("nan"),
        })
        per_film_counts[film_id] = per_film_counts.get(film_id, 0) + 1
        split_key = str(row["split"])
        per_split_counts[split_key] = per_split_counts.get(split_key, 0) + 1
        if len(batch_frames) >= batch_size:
            _flush()
    _flush()

    torch.save(visual_dict, output_dir / f"{split_name}_visual.pt")
    torch.save(audio_dict, output_dir / f"{split_name}_audio.pt")
    torch.save(metadata_dict, output_dir / f"{split_name}_metadata.pt")

    manifest = {
        "labels_csv": str(labels_csv),
        "windows_dir": str(windows_dir),
        "ccmovies_constant": CCMOVIES,
        "split_name": split_name,
        "num_frames": num_frames,
        "frame_size": frame_size,
        "audio_sr": audio_sr,
        "window_sec": WINDOW_SEC,
        "per_film_counts": per_film_counts,
        "per_split_counts": per_split_counts,
        "total_windows": len(metadata_dict),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote {len(metadata_dict)} windows → {output_dir}")
    print(f"[info] per_split: {per_split_counts}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", type=Path, required=True)
    p.add_argument("--windows_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--split_name", type=str, default="ccmovies")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--audio_sr", type=int, default=16000)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    preprocess_ccmovies(
        labels_csv=args.labels_csv,
        windows_dir=args.windows_dir,
        output_dir=args.output_dir,
        split_name=args.split_name,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        audio_sr=args.audio_sr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
