"""Phase 4-A Step 3 — COGNIMUSE X-CLIP + PANNs features precompute.

precompute_liris.py 의 fork.

주요 차이 vs LIRIS:
    - LIRIS는 clip이 독립 파일, COGNIMUSE는 한 영화가 30분 연속 파일
    - 따라서 start_sec / end_sec 구간을 metadata 에서 읽어 영상 디코딩 시
      해당 구간만 추출해야 한다.
    - 나머지는 모두 BASE 동일: audio crop=4s / stride=2s / pad_to=10s,
      visual 8 frames uniform, sample rate 32000, frame 224×224 ImageNet norm.

출력 레이아웃:
    data/features/cognimuse_panns_v5spec/
      features.pt    dict[name: str] → {"xclip": Tensor(512,), "panns": Tensor(2048,)}
      progress.json  device, crop constants, processed count

재현성:
    BASE features.pt 와 bit-identical 파이프라인 (audio/video 상수 동일).
    BASE weights가 forward 시 의미 있는 예측을 만들도록 feature distribution
    을 가능한 한 LIRIS 와 정렬.
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

from model.autoEQ.train.config import TrainConfig
from model.autoEQ.train.encoders import PANNsEncoder, XCLIPEncoder

AUDIO_SR = 32000
CROP_SEC = 4.0
STRIDE_SEC = 2.0
PAD_TO_SEC = 10.0
NUM_FRAMES = 8
FRAME_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def extract_audio_segment(
    video_path: Path,
    out_wav: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """FFmpeg로 [start_sec, end_sec] 오디오 구간을 32kHz mono wav로 추출."""
    duration = end_sec - start_sec
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-vn", "-ar", str(AUDIO_SR), "-ac", "1",
            str(out_wav),
        ],
        check=True,
        capture_output=True,
    )


def load_audio_strided_crops(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    crop_sec: float = CROP_SEC,
    stride_sec: float = STRIDE_SEC,
    pad_to_sec: float = PAD_TO_SEC,
) -> torch.Tensor:
    """구간 [start_sec, end_sec]에서 sliding crops 추출. 반환: (N_crops, crop_samples)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        wav = Path(tmp.name)
        extract_audio_segment(video_path, wav, start_sec, end_sec)
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

    crops = np.stack([arr[s : s + crop_samples] for s in starts], axis=0)
    return torch.from_numpy(crops).float()


def load_frames_uniform_segment(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    num_frames: int = NUM_FRAMES,
) -> torch.Tensor:
    """[start_sec, end_sec] 구간에서 num_frames 개 uniform sample. 반환: (1, T, C, H, W)."""
    import cv2
    from torchvision.transforms.functional import resize

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"cannot read frames: {video_path}")

    # timestamps: edges skipped inside [start_sec, end_sec]
    timestamps = np.linspace(start_sec, end_sec, num_frames + 2)[1:-1]

    frames: list[torch.Tensor] = []
    for ts in timestamps:
        idx = max(0, min(total - 1, int(round(ts * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = cap.read()
        if not ret or bgr is None:
            cap.release()
            raise RuntimeError(f"frame read fail at ts={ts}s in {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        f = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        f = resize(f, [FRAME_SIZE, FRAME_SIZE], antialias=True)
        frames.append(f)
    cap.release()

    t = torch.stack(frames, dim=0)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0)


def resolve_video_path(row: pd.Series, cognimuse_root: Path) -> Path:
    """metadata row에서 video 경로 해석.

    Data/ 및 Data-NotOfficial/ 양쪽을 시도 (영화별 위치가 다름).
    """
    folder = row["video_folder"]
    basename = row["video_basename"]
    candidates = [
        cognimuse_root / "Data" / folder / f"{basename}.avi",
        cognimuse_root / "Data-NotOfficial" / folder / f"{basename}.avi",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"video not found for {row['name']}: tried {[str(c) for c in candidates]}"
    )


def save_checkpoint(features: dict, output_dir: Path, stats: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "features.pt.tmp"
    torch.save(features, tmp)
    tmp.replace(output_dir / "features.pt")
    (output_dir / "progress.json").write_text(json.dumps(stats, indent=2))


def run(
    metadata_csv: Path,
    cognimuse_root: Path,
    output_dir: Path,
    device: str,
    save_every: int,
    limit: int | None,
    crop_sec: float,
    stride_sec: float,
    pad_to_sec: float,
) -> int:
    if device == "mps" and not torch.backends.mps.is_available():
        print("[precompute] MPS not available, falling back to cpu", file=sys.stderr)
        device = "cpu"

    df = pd.read_csv(metadata_csv)
    if limit:
        df = df.head(limit)
    print(f"[precompute] metadata       : {metadata_csv}")
    print(f"[precompute] cognimuse_root : {cognimuse_root}")
    print(f"[precompute] windows        : {len(df)}")
    print(f"[precompute] device         : {device}")
    print(f"[precompute] output_dir     : {output_dir}")
    print(f"[precompute] audio          : crop={crop_sec}s stride={stride_sec}s pad_to={pad_to_sec}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_file = output_dir / "features.pt"
    features: dict = {}
    if feature_file.is_file():
        features = torch.load(feature_file, map_location="cpu", weights_only=False)
        print(f"[precompute] resume from: {feature_file} ({len(features)} windows done)")

    todo = [row for _, row in df.iterrows() if row["name"] not in features]
    print(f"[precompute] remaining      : {len(todo)}")
    if not todo:
        print("[precompute] nothing to do.")
        return 0

    cfg = TrainConfig()
    print("[precompute] loading encoders ...")
    t_load = time.time()
    xclip = XCLIPEncoder(cfg).to(device)
    panns = PANNsEncoder(cfg).to(device)
    xclip.eval()
    panns.eval()
    print(f"[precompute] encoders ready in {time.time() - t_load:.1f}s")

    stats = {
        "device": device,
        "total_windows": int(len(df)),
        "processed": len(features),
        "failed": [],
        "audio_crop_sec": crop_sec,
        "audio_stride_sec": stride_sec,
        "audio_pad_to_sec": pad_to_sec,
        "num_frames": NUM_FRAMES,
        "frame_size": FRAME_SIZE,
        "sample_rate": AUDIO_SR,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": 0.0,
        "total_audio_forwards": 0,
        "source": "cognimuse_hollywood_12",
    }
    t_start = time.time()
    last_ck = t_start
    print_every = 25

    for i, row in enumerate(todo):
        name = row["name"]
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        try:
            video_path = resolve_video_path(row, cognimuse_root)
        except FileNotFoundError as e:
            stats["failed"].append({"name": name, "reason": str(e)})
            continue

        try:
            frames = load_frames_uniform_segment(
                video_path, start_sec, end_sec, num_frames=NUM_FRAMES
            ).to(device)
            audio_crops = load_audio_strided_crops(
                video_path, start_sec, end_sec,
                crop_sec=crop_sec, stride_sec=stride_sec, pad_to_sec=pad_to_sec,
            ).to(device)
            with torch.no_grad():
                v_feat = xclip(frames).squeeze(0).cpu()
                a_feats = panns(audio_crops).cpu()
                a_feat = a_feats.mean(dim=0)
            features[name] = {"xclip": v_feat, "panns": a_feat}
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
            print(f"[precompute] checkpoint saved ({processed_now}/{len(df)})")

    stats["processed"] = len(features)
    stats["elapsed_sec"] = round(time.time() - t_start, 1)
    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_checkpoint(features, output_dir, stats)
    print(
        f"\n[precompute] DONE. {len(features)}/{len(df)} windows, "
        f"{len(stats['failed'])} failed, {stats['elapsed_sec']/60:.1f} min, "
        f"{stats['total_audio_forwards']} audio forwards"
    )
    if stats["failed"]:
        print(f"[precompute] failed (first 5): {stats['failed'][:5]}")
    return 0 if not stats["failed"] else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("dataset/autoEQ/cognimuse/cognimuse_metadata.csv"),
    )
    p.add_argument(
        "--cognimuse-root",
        type=Path,
        default=Path("dataset/autoEQ/cognimuse"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features/cognimuse_panns_v5spec"),
    )
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--crop-sec", type=float, default=CROP_SEC)
    p.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    p.add_argument("--pad-to-sec", type=float, default=PAD_TO_SEC)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        args.metadata, args.cognimuse_root, args.output_dir,
        args.device, args.save_every, args.limit,
        args.crop_sec, args.stride_sec, args.pad_to_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
