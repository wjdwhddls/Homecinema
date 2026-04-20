"""Precompute CLIP image + frame-mean visual features for the clip_framemean ablation.

Produces a new feature directory that mirrors ``data/features/ccmovies/`` but
replaces the X-CLIP video cache with per-frame CLIP ViT-B/32 pooler_output
averaged over the 8 window frames. Audio features and metadata are **copied
byte-for-byte** from the baseline cache so that the only variable in the
paired comparison is the visual encoder's aggregation strategy.

Fairness guardrails (single-axis ablation, PANNs-vs-AST-style):
    - Foundation kept identical:    CLIP ViT-B/32 (same as X-CLIP's backbone)
    - Pretraining kept identical:   WIT-400M image-text contrastive
    - Frame timestamps kept identical to X-CLIP precompute:
          frame_ts = [WINDOW_SEC * (i + 0.5) / num_frames for i in range(8)]
    - Image normalization kept identical to baseline's X-CLIP pipeline
      (ImageNet mean/std, as in ``cognimuse_preprocess.load_frames_from_mp4``)
      so both encoders receive byte-for-byte identical pixel tensors.
    - Only X-CLIP's learned temporal-attention / prompt-encoder stack is
      replaced by a uniform frame mean.

Layout produced:
    <out_dir>/ccmovies_visual.pt    (new, dict[wid -> (512,)])
    <out_dir>/ccmovies_audio.pt     (copy of baseline, dict[wid -> (2048,)])
    <out_dir>/ccmovies_metadata.pt  (copy of baseline)
    <out_dir>/manifest.json         (model revision, feature dims, etc.)

Usage:
    python scripts/precompute_clipimg_framemean_features.py \\
        --baseline-feature-dir data/features/ccmovies \\
        --windows-dir dataset/autoEQ/CCMovies/windows \\
        --out-dir data/features/ccmovies_clipimg \\
        --device mps
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
WINDOW_SEC = 4.0
NUM_FRAMES = 8
FRAME_SIZE = 224


def _resolve_mp4_path(windows_dir: Path, wid: str) -> Path:
    """Window IDs are ``<film>_<idx>``; file lives under
    ``windows_dir/<film>/<wid>.mp4``. Greedy match by film prefix.
    """
    for film_dir in sorted(windows_dir.iterdir()):
        if not film_dir.is_dir():
            continue
        if wid.startswith(film_dir.name + "_"):
            candidate = film_dir / f"{wid}.mp4"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no mp4 found for window id '{wid}' under {windows_dir}")


def extract_clip_framemean_embeddings(
    window_ids: list[str],
    windows_dir: Path,
    model_name: str,
    device: str,
    progress_every: int = 25,
) -> tuple[dict[str, Tensor], dict]:
    """Load mp4 windows, sample 8 frames per window with the same timestamps
    as the baseline X-CLIP precompute, encode each frame through CLIP's
    image tower, and mean-pool to a single 512-dim vector per window.
    """
    # CLIPVisionModelWithProjection exposes `image_embeds` — the 512-dim
    # projection head output that lives in the same contrastive space as
    # X-CLIP's `get_video_features()` (512-dim video_embeds). Using raw
    # `CLIPVisionModel.pooler_output` would give 768-dim ViT hidden state,
    # which breaks dim parity with the baseline fused tower (visual_dim=512).
    from transformers import CLIPVisionModelWithProjection

    # We deliberately reuse the baseline's frame loader so the pixel tensors
    # are bit-identical to what X-CLIP consumed.
    from model.autoEQ.train_pseudo.cognimuse_preprocess import load_frames_from_mp4

    print(f"[CLIP] loading {model_name} on {device} ...")
    model = CLIPVisionModelWithProjection.from_pretrained(model_name)
    model.eval()
    model.to(device)

    frame_ts = [WINDOW_SEC * (i + 0.5) / NUM_FRAMES for i in range(NUM_FRAMES)]

    embeddings: dict[str, Tensor] = {}
    t0 = time.time()
    with torch.inference_mode():
        for i, wid in enumerate(window_ids):
            mp4 = _resolve_mp4_path(windows_dir, wid)
            # frames: (num_frames, 3, H, W), ImageNet-normalized (same as X-CLIP pipeline)
            frames = load_frames_from_mp4(mp4, frame_ts, frame_size=FRAME_SIZE)
            pixel_values = frames.to(device)  # (8, 3, 224, 224)
            out = model(pixel_values=pixel_values)
            # `image_embeds`: (num_frames, 512), already L2-normalizable contrastive
            # embedding in the same space as X-CLIP's video_embeds.
            frame_embeds = out.image_embeds          # (8, 512)
            pooled = frame_embeds.mean(dim=0)        # (512,)
            embeddings[wid] = pooled.detach().to("cpu").float()

            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                remaining = (len(window_ids) - (i + 1)) / rate if rate > 0 else 0.0
                print(
                    f"  [{i + 1:4d}/{len(window_ids):4d}] {rate:.1f} win/s, "
                    f"~{remaining:.0f}s left"
                )

    manifest = {
        "model_name": model_name,
        "feature_dim": int(next(iter(embeddings.values())).shape[-1]),
        "num_windows": len(embeddings),
        "num_frames": NUM_FRAMES,
        "frame_size": FRAME_SIZE,
        "window_sec": WINDOW_SEC,
        "frame_timestamps_sec": frame_ts,
        "image_normalization": "ImageNet mean/std (inherited from baseline pipeline)",
        "pool": "frame_mean over pooler_output (ViT [CLS])",
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return embeddings, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-feature-dir",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "ccmovies",
        help="source of audio/metadata .pt files to copy verbatim",
    )
    parser.add_argument(
        "--windows-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "autoEQ" / "CCMovies" / "windows",
        help="directory containing <film>/<wid>.mp4 files",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="ccmovies",
        help=".pt filename prefix (e.g. ccmovies -> ccmovies_visual.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "ccmovies_clipimg",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_CLIP_MODEL)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip heavy work; validate inputs and print the plan",
    )
    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Validate inputs
    visual_src = args.baseline_feature_dir / f"{args.split_name}_visual.pt"
    audio_src = args.baseline_feature_dir / f"{args.split_name}_audio.pt"
    meta_src = args.baseline_feature_dir / f"{args.split_name}_metadata.pt"
    for p in (audio_src, meta_src):
        if not p.is_file():
            raise FileNotFoundError(f"missing baseline feature file: {p}")
    if not args.windows_dir.is_dir():
        raise FileNotFoundError(f"windows dir not found: {args.windows_dir}")

    metadata = torch.load(meta_src, weights_only=False)
    window_ids = sorted(metadata.keys())
    print(f"Baseline feature dir : {args.baseline_feature_dir}")
    print(f"Windows dir          : {args.windows_dir}")
    print(f"Output feature dir   : {args.out_dir}")
    print(f"Windows to encode    : {len(window_ids)}")
    print(f"Model                : {args.model_name}")
    print(f"Num frames / window  : {NUM_FRAMES}  (frame_ts same as baseline X-CLIP)")
    print(f"Device               : {device}")
    if args.dry_run:
        print("[dry-run] skipping encoding + writes.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio + metadata byte-for-byte (identical inputs for paired comparison)
    shutil.copyfile(audio_src, args.out_dir / audio_src.name)
    shutil.copyfile(meta_src, args.out_dir / meta_src.name)
    print("Copied audio + metadata caches.")

    # Encode visual via CLIP image + frame-mean
    embeddings, manifest = extract_clip_framemean_embeddings(
        window_ids,
        args.windows_dir,
        args.model_name,
        device,
    )

    visual_dst = args.out_dir / visual_src.name
    torch.save(embeddings, visual_dst)
    print(f"Saved {len(embeddings)} CLIP frame-mean embeddings to {visual_dst}")

    manifest["baseline_feature_dir"] = str(args.baseline_feature_dir)
    manifest["windows_dir"] = str(args.windows_dir)
    manifest["split_name"] = args.split_name
    try:
        import transformers

        manifest["transformers_version"] = transformers.__version__
    except Exception:
        pass
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
