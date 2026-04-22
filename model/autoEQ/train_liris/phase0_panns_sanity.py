"""Phase 0 — PANNs 4s input sanity check (V5-FINAL §6-1).

Tests whether a 4s center crop of a LIRIS clip yields a PANNs CNN14 embedding
close enough to the full-clip embedding for downstream VA regression.

Procedure
---------
1. Sample N clips from LIRIS-ACCEDE-data/data/.
2. Extract audio via ffmpeg at 32kHz mono (PANNs native rate).
3. Compute PANNs(full) and PANNs(center_crop_4s) embeddings.
4. Cosine similarity per clip + summary stats.
5. Persist runs/phase0_sanity/panns_sanity.json.

Verdict (mean cosine similarity)
--------------------------------
  >= 0.90  : OK          — 4s input is safe for the V3.2 design.
  0.80~0.90: WARN        — proceed with caution.
  < 0.80   : FAIL        — revisit audio window strategy.

Usage
-----
  python -m model.autoEQ.train_liris.phase0_panns_sanity \
      --liris-dir dataset/autoEQ/liris/data/data \
      --num-clips 10 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, median, stdev

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from ..train.config import TrainConfig
from ..train.encoders import PANNsEncoder

PANNS_SR = 32000
CROP_SEC = 4.0


def extract_audio_32k_mono(video_path: Path, out_wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", str(PANNS_SR), "-ac", "1",
            str(out_wav),
        ],
        check=True, capture_output=True,
    )


def center_crop(waveform: np.ndarray, sr: int, crop_sec: float) -> np.ndarray:
    n_target = int(round(crop_sec * sr))
    if waveform.shape[0] <= n_target:
        pad = n_target - waveform.shape[0]
        return np.pad(waveform, (0, pad), mode="constant")
    start = (waveform.shape[0] - n_target) // 2
    return waveform[start : start + n_target]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def sample_clips(liris_dir: Path, num_clips: int, seed: int) -> list[Path]:
    all_clips = sorted(liris_dir.glob("ACCEDE*.mp4"))
    if not all_clips:
        raise FileNotFoundError(f"No ACCEDE*.mp4 found under {liris_dir}")
    rng = random.Random(seed)
    return rng.sample(all_clips, min(num_clips, len(all_clips)))


def run_sanity(
    liris_dir: Path,
    num_clips: int,
    seed: int,
    output_dir: Path,
) -> dict:
    print(f"[phase0] LIRIS dir : {liris_dir}")
    print(f"[phase0] num_clips : {num_clips}, seed={seed}")

    clips = sample_clips(liris_dir, num_clips, seed)
    print(f"[phase0] sampled   : {[c.name for c in clips]}")

    cfg = TrainConfig()
    print("[phase0] loading PANNs CNN14 (frozen) ...")
    panns = PANNsEncoder(cfg)
    panns.eval()

    per_clip: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for idx, clip in enumerate(clips):
            t0 = time.time()
            wav_path = tmp / f"{clip.stem}.wav"
            extract_audio_32k_mono(clip, wav_path)
            full_np, sr = librosa.load(wav_path, sr=PANNS_SR, mono=True)
            assert sr == PANNS_SR, f"unexpected sr={sr}"
            full_sec = float(full_np.shape[0] / sr)

            crop_np = center_crop(full_np, sr, CROP_SEC)

            full_t = torch.from_numpy(full_np).float().unsqueeze(0)
            crop_t = torch.from_numpy(crop_np).float().unsqueeze(0)

            with torch.no_grad():
                emb_full = panns(full_t).squeeze(0).cpu()
                emb_crop = panns(crop_t).squeeze(0).cpu()

            sim = cosine(emb_full, emb_crop)
            dt = time.time() - t0
            per_clip.append(
                {
                    "clip": clip.name,
                    "duration_sec": round(full_sec, 3),
                    "cosine": round(sim, 5),
                    "elapsed_sec": round(dt, 2),
                }
            )
            print(
                f"[{idx+1:>2}/{len(clips)}] {clip.name} "
                f"dur={full_sec:5.2f}s cos={sim:.4f} ({dt:.1f}s)"
            )

    sims = [c["cosine"] for c in per_clip]
    mean_sim = mean(sims)
    if mean_sim >= 0.90:
        verdict = "OK"
    elif mean_sim >= 0.80:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    summary = {
        "mean": round(mean_sim, 5),
        "median": round(median(sims), 5),
        "std": round(stdev(sims) if len(sims) > 1 else 0.0, 5),
        "min": round(min(sims), 5),
        "max": round(max(sims), 5),
        "n": len(sims),
        "verdict": verdict,
    }

    report = {
        "task": "phase0_panns_sanity",
        "plan_section": "V5-FINAL §6-1",
        "panns_sr": PANNS_SR,
        "crop_sec": CROP_SEC,
        "seed": seed,
        "liris_dir": str(liris_dir),
        "num_clips": num_clips,
        "per_clip": per_clip,
        "summary": summary,
        "thresholds": {"ok": 0.90, "warn": 0.80},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "panns_sanity.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[phase0] summary    : {summary}")
    print(f"[phase0] verdict    : {verdict} (mean={mean_sim:.4f})")
    print(f"[phase0] wrote      : {out_path}")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--liris-dir",
        type=Path,
        default=Path("dataset/autoEQ/liris/data/data"),
    )
    p.add_argument("--num-clips", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase0_sanity"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.liris_dir.is_dir():
        print(f"error: liris_dir not found: {args.liris_dir}", file=sys.stderr)
        return 2
    report = run_sanity(args.liris_dir, args.num_clips, args.seed, args.output_dir)
    return 0 if report["summary"]["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
