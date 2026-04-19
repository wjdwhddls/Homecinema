"""Precompute AST audio features for the `model_ast` ablation.

Produces a new feature directory that mirrors
``data/features/ccmovies/`` but replaces the PANNs audio cache with
AST [CLS]-token embeddings. Visual features and metadata are **copied
byte-for-byte** from the baseline cache so that the only variable in the
paired comparison is the audio encoder.

Layout produced:
    <out_dir>/ccmovies_visual.pt     (copy of baseline, dict[wid -> (512,)])
    <out_dir>/ccmovies_metadata.pt   (copy of baseline)
    <out_dir>/ccmovies_audio.pt      (new, dict[wid -> (768,)])
    <out_dir>/manifest.json          (model revision, feature dims, sr, etc.)

Usage:
    python scripts/precompute_ast_features.py \\
        --baseline-feature-dir data/features/ccmovies \\
        --windows-dir dataset/autoEQ/CCMovies/windows \\
        --out-dir data/features/ccmovies_ast \\
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


DEFAULT_AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _load_wav_16k(path: Path) -> Tensor:
    """Load mono 16 kHz waveform as float32 tensor (samples,)."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    t = torch.from_numpy(audio)
    if sr != 16000:
        import torchaudio.functional as AF

        t = AF.resample(t, orig_freq=sr, new_freq=16000)
    return t


def _resolve_wav_path(windows_dir: Path, wid: str) -> Path:
    """Window IDs are of the form ``<film>_<idx>``; file lives under
    ``windows_dir/<film>/<wid>.wav``.

    Finds the film prefix greedily against the directory list.
    """
    for film_dir in sorted(windows_dir.iterdir()):
        if not film_dir.is_dir():
            continue
        if wid.startswith(film_dir.name + "_"):
            candidate = film_dir / f"{wid}.wav"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no wav file found for window id '{wid}' under {windows_dir}")


def extract_ast_embeddings(
    window_ids: list[str],
    windows_dir: Path,
    model_name: str,
    device: str,
    progress_every: int = 25,
) -> tuple[dict[str, Tensor], dict]:
    """Run AST over every window and return a dict[wid -> (768,) CPU tensor]
    plus a manifest with the model revision and feature layout.
    """
    from transformers import ASTFeatureExtractor, ASTModel

    print(f"[AST] loading {model_name} on {device} ...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    model = ASTModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    embeddings: dict[str, Tensor] = {}
    t0 = time.time()
    with torch.inference_mode():
        for i, wid in enumerate(window_ids):
            wav = _load_wav_16k(_resolve_wav_path(windows_dir, wid))
            inputs = feature_extractor(
                wav.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_values = inputs["input_values"].to(device)
            out = model(input_values)
            # ASTModel returns BaseModelOutputWithPooling; [CLS] token is
            # last_hidden_state[:, 0, :]. pooler_output is an MLP on top of
            # that so either works; we take the raw [CLS] for minimalism.
            cls = out.last_hidden_state[:, 0, :].squeeze(0).detach().to("cpu")
            embeddings[wid] = cls.float()

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
        "sampling_rate_hz": 16000,
        "device": device,
        "extractor": "ASTFeatureExtractor (defaults)",
        "token": "last_hidden_state[:, 0, :] (CLS)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return embeddings, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-feature-dir",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "ccmovies",
        help="source of visual/metadata .pt files to copy verbatim",
    )
    parser.add_argument(
        "--windows-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "autoEQ" / "CCMovies" / "windows",
        help="directory containing <film>/<wid>.wav files",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="ccmovies",
        help=".pt filename prefix (e.g. ccmovies -> ccmovies_audio.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "ccmovies_ast",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_AST_MODEL)
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
    for p in (visual_src, meta_src):
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
    print(f"Device               : {device}")
    if args.dry_run:
        print("[dry-run] skipping encoding + writes.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Copy visual + metadata byte-for-byte (identical inputs for paired comparison)
    shutil.copyfile(visual_src, args.out_dir / visual_src.name)
    shutil.copyfile(meta_src, args.out_dir / meta_src.name)
    print("Copied visual + metadata caches.")

    # Encode audio via AST
    embeddings, manifest = extract_ast_embeddings(
        window_ids,
        args.windows_dir,
        args.model_name,
        device,
    )

    audio_dst = args.out_dir / audio_src.name
    torch.save(embeddings, audio_dst)
    print(f"Saved {len(embeddings)} AST embeddings to {audio_dst}")

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
