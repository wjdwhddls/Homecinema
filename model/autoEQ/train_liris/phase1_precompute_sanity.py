"""Phase 1 Step 2 pre-flight — MPS (Apple Silicon) vs CPU parity & speed check.

Runs X-CLIP + PANNs on the same 10 LIRIS clips twice (CPU first, then MPS).
For each encoder:
  - cosine similarity between the CPU and MPS feature vectors (should be ≥ 0.99)
  - wall-clock time per clip

Verdict
-------
  mean cosine ≥ 0.99  → OK, full 9,800-clip precompute on MPS is safe
  0.95~0.99           → WARN (numerical drift worth investigating)
  < 0.95              → FAIL (op fallback suspected; stay on CPU)

Usage
-----
  python -m model.autoEQ.train_liris.phase1_precompute_sanity \
      --liris-dir dataset/autoEQ/liris/data/data --num-clips 10 --seed 42
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
from statistics import mean

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from ..train.config import TrainConfig
from ..train.encoders import PANNsEncoder, XCLIPEncoder

AUDIO_SR = 32000
CROP_SEC = 4.0
NUM_FRAMES = 8
FRAME_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def extract_audio_32k(video_path: Path, out_wav: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vn", "-ar", str(AUDIO_SR), "-ac", "1", str(out_wav)],
        check=True, capture_output=True,
    )


def load_audio_crop(video_path: Path) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        wav = Path(tmp.name)
        extract_audio_32k(video_path, wav)
        arr, sr = librosa.load(wav, sr=AUDIO_SR, mono=True)
    assert sr == AUDIO_SR
    n = int(round(CROP_SEC * sr))
    if arr.shape[0] >= n:
        start = (arr.shape[0] - n) // 2
        arr = arr[start : start + n]
    else:
        arr = np.pad(arr, (0, n - arr.shape[0]))
    return torch.from_numpy(arr).float().unsqueeze(0)  # (1, T)


def load_frames(video_path: Path, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    import cv2
    from torchvision.transforms.functional import resize

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read frames: {video_path}")
    duration = total / fps
    timestamps = np.linspace(0, duration, num_frames + 2)[1:-1]  # skip edges

    frames: list[torch.Tensor] = []
    for ts in timestamps:
        idx = max(0, min(total - 1, int(round(ts * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = cap.read()
        if not ret or bgr is None:
            raise RuntimeError(f"frame read fail at ts={ts}s in {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        f = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        f = resize(f, [FRAME_SIZE, FRAME_SIZE], antialias=True)
        frames.append(f)
    cap.release()

    t = torch.stack(frames, dim=0)  # (T, C, H, W)
    t = (t - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
    return t.unsqueeze(0)  # (1, T, C, H, W)


def encode_on_device(
    clips: list[Path],
    device: str,
) -> dict:
    cfg = TrainConfig()
    print(f"\n[device={device}] loading encoders ...")
    xclip = XCLIPEncoder(cfg).to(device)
    panns = PANNsEncoder(cfg).to(device)
    xclip.eval()
    panns.eval()

    records = []
    t0 = time.time()
    for i, clip in enumerate(clips):
        c0 = time.time()
        frames = load_frames(clip).to(device)
        audio = load_audio_crop(clip).to(device)
        with torch.no_grad():
            v_feat = xclip(frames).squeeze(0).cpu()
            a_feat = panns(audio).squeeze(0).cpu()
        dt = time.time() - c0
        records.append(
            {
                "clip": clip.name,
                "v_feat": v_feat,
                "a_feat": a_feat,
                "dt": dt,
            }
        )
        print(
            f"  [{i+1:>2}/{len(clips)}] {clip.name} "
            f"v={tuple(v_feat.shape)} a={tuple(a_feat.shape)} ({dt:.2f}s)"
        )
    total = time.time() - t0
    print(f"[device={device}] total {total:.1f}s, mean {total/len(clips):.2f}s/clip")
    return {"records": records, "total_sec": total}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def sample_clips(liris_dir: Path, n: int, seed: int) -> list[Path]:
    all_clips = sorted(liris_dir.glob("ACCEDE*.mp4"))
    rng = random.Random(seed)
    return rng.sample(all_clips, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--liris-dir", type=Path, default=Path("dataset/autoEQ/liris/data/data"))
    ap.add_argument("--num-clips", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=Path("runs/phase1_precompute_sanity"))
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        print("MPS not available; aborting.", file=sys.stderr)
        return 2

    clips = sample_clips(args.liris_dir, args.num_clips, args.seed)
    print(f"[sanity] clips: {[c.name for c in clips]}")

    cpu = encode_on_device(clips, "cpu")
    mps = encode_on_device(clips, "mps")

    per_clip = []
    v_sims, a_sims = [], []
    for c, m in zip(cpu["records"], mps["records"]):
        assert c["clip"] == m["clip"]
        v_sim = cosine(c["v_feat"], m["v_feat"])
        a_sim = cosine(c["a_feat"], m["a_feat"])
        v_sims.append(v_sim)
        a_sims.append(a_sim)
        per_clip.append(
            {
                "clip": c["clip"],
                "xclip_cos": round(v_sim, 6),
                "panns_cos": round(a_sim, 6),
                "cpu_dt": round(c["dt"], 3),
                "mps_dt": round(m["dt"], 3),
            }
        )

    v_mean, a_mean = mean(v_sims), mean(a_sims)
    worst = min(v_mean, a_mean)
    if worst >= 0.99:
        verdict = "OK"
    elif worst >= 0.95:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    summary = {
        "xclip_cos_mean": round(v_mean, 6),
        "xclip_cos_min": round(min(v_sims), 6),
        "panns_cos_mean": round(a_mean, 6),
        "panns_cos_min": round(min(a_sims), 6),
        "cpu_total_sec": round(cpu["total_sec"], 2),
        "mps_total_sec": round(mps["total_sec"], 2),
        "speedup": round(cpu["total_sec"] / max(mps["total_sec"], 1e-6), 2),
        "eta_9800_cpu_hours": round(cpu["total_sec"] * 9800 / len(clips) / 3600, 2),
        "eta_9800_mps_hours": round(mps["total_sec"] * 9800 / len(clips) / 3600, 2),
        "verdict": verdict,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "precompute_sanity.json").write_text(
        json.dumps({"per_clip": per_clip, "summary": summary}, indent=2)
    )

    print("\n[sanity] === SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:>22s}: {v}")
    print(f"\n[sanity] verdict: {verdict}")
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
