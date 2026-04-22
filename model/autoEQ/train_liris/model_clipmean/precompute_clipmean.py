"""Phase 2a-4 Visual Encoder ablation — precompute CLIP frame-mean visual features.

Design
------
Only the visual encoder changes (X-CLIP base-patch32 → CLIP ViT-B/32
frame-mean). The audio side (PANNs CNN14 @ 32 kHz, stride=2s, pad_to=10s) is
**deterministic** and byte-identical to the BASE Phase 2a-3 precompute, so
by default this script **reuses** the cached PANNs tensors from
`data/features/liris_panns_v5spec/features.pt` and only runs CLIP inference.

This guarantees:
  * OAT purity — the audio feature vectors entering the training loop are
    literally the same bytes; any CCC delta is caused solely by the visual
    encoder swap.
  * Runtime — skip ~hours of PANNs forwards; CPU CLIP inference 9,800 × 8
    frames is the only cost.

Output layout
-------------
  data/features/liris_clipmean_v5spec/
    features.pt   dict[name: str] → {"clipmean": Tensor(512,), "panns": Tensor(2048,)}
    progress.json running stats, timestamp, device, completed count

Resumability
------------
  If features.pt exists we load it and skip clips already present.
  A checkpoint save happens every --save-every clips (default 500).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from ..precompute_liris import load_frames_uniform, save_checkpoint
from .encoders import CLIPFrameMeanEncoder


# --- Main ---------------------------------------------------------------------


def run(
    metadata_csv: Path,
    liris_dir: Path,
    output_dir: Path,
    panns_cache: Path,
    device: str,
    save_every: int,
    limit: int | None,
    clip_model_name: str,
) -> int:
    if device == "mps" and not torch.backends.mps.is_available():
        print("[precompute_clipmean] MPS not available, falling back to cpu", file=sys.stderr)
        device = "cpu"

    df = pd.read_csv(metadata_csv)
    if limit:
        df = df.head(limit)
    print(f"[precompute_clipmean] metadata    : {metadata_csv}")
    print(f"[precompute_clipmean] clips       : {len(df)}")
    print(f"[precompute_clipmean] device      : {device}")
    print(f"[precompute_clipmean] output_dir  : {output_dir}")
    print(f"[precompute_clipmean] panns_cache : {panns_cache}")
    print(f"[precompute_clipmean] clip_model  : {clip_model_name}")

    # --- Load PANNs cache (reuse, audio side byte-identical to BASE) ---------
    if not panns_cache.is_file():
        print(
            f"[precompute_clipmean] FATAL: PANNs cache not found at {panns_cache}. "
            f"Run train_liris/precompute_liris.py first (or pass --panns-cache).",
            file=sys.stderr,
        )
        return 2
    print("[precompute_clipmean] loading PANNs cache ...")
    panns_features = torch.load(panns_cache, map_location="cpu", weights_only=False)
    print(f"[precompute_clipmean] panns cache : {len(panns_features)} clips")
    # Sanity — every clip we need must be in the cache.
    missing = [row["name"] for _, row in df.iterrows() if row["name"] not in panns_features]
    if missing:
        print(
            f"[precompute_clipmean] FATAL: {len(missing)} clips missing from PANNs cache "
            f"(first: {missing[:3]}).",
            file=sys.stderr,
        )
        return 3

    # --- Load/resume output --------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_file = output_dir / "features.pt"
    features: dict = {}
    if feature_file.is_file():
        features = torch.load(feature_file, map_location="cpu", weights_only=False)
        print(f"[precompute_clipmean] resume from : {feature_file} ({len(features)} clips done)")

    todo = [row for _, row in df.iterrows() if row["name"] not in features]
    print(f"[precompute_clipmean] remaining   : {len(todo)}")
    if not todo:
        print("[precompute_clipmean] nothing to do.")
        return 0

    # --- Load visual encoder -------------------------------------------------
    print("[precompute_clipmean] loading CLIP ...")
    t_load = time.time()
    clip = CLIPFrameMeanEncoder(model_name=clip_model_name).to(device)
    clip.eval()
    print(f"[precompute_clipmean] clip ready in {time.time()-t_load:.1f}s")

    stats = {
        "device": device,
        "total_clips": int(len(df)),
        "processed": len(features),
        "failed": [],
        "clip_model_name": clip_model_name,
        "panns_cache": str(panns_cache),
        "panns_reused": True,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": 0.0,
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
            frames = load_frames_uniform(clip_path).to(device)       # (1, 8, 3, 224, 224)
            with torch.no_grad():
                v_feat = clip(frames).squeeze(0).cpu()               # (512,)
            a_feat = panns_features[name]["panns"].clone()           # (2048,) reused
            features[name] = {"clipmean": v_feat, "panns": a_feat}
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
            print(f"[precompute_clipmean] checkpoint saved ({processed_now}/{len(df)})")

    stats["processed"] = len(features)
    stats["elapsed_sec"] = round(time.time() - t_start, 1)
    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_checkpoint(features, output_dir, stats)
    print(f"\n[precompute_clipmean] DONE. {len(features)}/{len(df)} clips, "
          f"{len(stats['failed'])} failed, {stats['elapsed_sec']/60:.1f} min")
    if stats["failed"]:
        print(f"[precompute_clipmean] failed: {stats['failed'][:5]} ...")
    return 0 if not stats["failed"] else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path,
                   default=Path("dataset/autoEQ/liris/liris_metadata.csv"))
    p.add_argument("--liris-dir", type=Path,
                   default=Path("dataset/autoEQ/liris/data/data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/features/liris_clipmean_v5spec"))
    p.add_argument("--panns-cache", type=Path,
                   default=Path("data/features/liris_panns_v5spec/features.pt"),
                   help="Existing PANNs features.pt to reuse (audio side byte-identical to BASE).")
    p.add_argument("--device", choices=["mps", "cpu"], default="cpu")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--clip-model-name", type=str,
                   default="openai/clip-vit-base-patch32")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.metadata, args.liris_dir, args.output_dir,
               args.panns_cache, args.device, args.save_every, args.limit,
               args.clip_model_name)


if __name__ == "__main__":
    raise SystemExit(main())
