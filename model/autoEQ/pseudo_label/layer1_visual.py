"""Layer 1 Visual 앙상블 — EmoNet(face) + VEATIC(context ViT) + CLIP(zero-shot) V/A 추론.

각 4s window MP4에서 균일 프레임 샘플링 후 3개 모델로 V/A 예측.

  - EmoNet: face 검출(SFD) → 256x256 crop → V/A [-1, 1] 반환. face 미검출 시 NaN + detected=False.
  - VEATIC: 5-frame sequence (640x480) → ResNet50 + ViT → raw V/A (비정규화; aggregate에서 robust scale).
  - CLIP: OpenCLIP ViT-B/32 zero-shot. 4 프레임 평균 embedding을 valence/arousal 프롬프트 bank와 비교.

출력 CSV 컬럼:
  film_id, window_id, emonet_v, emonet_a, emonet_detected,
                     veatic_v, veatic_a, clip_v, clip_a

Usage:
  python -m model.autoEQ.pseudo_label.layer1_visual \\
    --windows_dir dataset/autoEQ/CCMovies/windows \\
    --emonet_repo third_party/emonet \\
    --veatic_repo third_party/VEATIC \\
    --veatic_weight third_party/VEATIC/weights/veatic_pretrain.pth \\
    --output_csv dataset/autoEQ/CCMovies/labels/layer1_visual.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


EMONET_IMAGE_SIZE = 256
VEATIC_NUM_FRAMES = 5
VEATIC_FRAME_H = 480
VEATIC_FRAME_W = 640
CLIP_NUM_FRAMES = 4
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

CLIP_VALENCE_POS = [
    "a joyful pleasant scene",
    "a happy heartwarming moment",
    "a bright uplifting scene",
]
CLIP_VALENCE_NEG = [
    "a sad distressing scene",
    "a dark unpleasant moment",
    "a gloomy depressing scene",
]
CLIP_AROUSAL_HIGH = [
    "an intense exciting action scene",
    "a chaotic thrilling moment",
    "a fast-paced energetic scene",
]
CLIP_AROUSAL_LOW = [
    "a calm peaceful quiet scene",
    "a tranquil serene moment",
    "a slow contemplative scene",
]


def iter_window_mp4s(windows_dir: Path) -> Iterable[tuple[str, str, Path]]:
    for film_dir in sorted(windows_dir.iterdir()):
        if not film_dir.is_dir():
            continue
        for mp4 in sorted(film_dir.glob("*.mp4")):
            yield film_dir.name, mp4.stem, mp4


def sample_frames(mp4_path: Path, n_frames: int) -> np.ndarray | None:
    """Return (n_frames, H, W, 3) uint8 RGB. None if failed."""
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    if total < n_frames:
        indices = list(range(total)) + [total - 1] * (n_frames - total)
    else:
        indices = [int(round(i * (total - 1) / (n_frames - 1))) for i in range(n_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret:
            cap.release()
            return None
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


class EmoNetPredictor:
    def __init__(self, emonet_repo: Path, device: str):
        sys.path.insert(0, str(emonet_repo))
        from emonet.models import EmoNet  # type: ignore
        from face_alignment.detection.sfd.sfd_detector import SFDDetector

        state_path = emonet_repo / "pretrained" / "emonet_8.pth"
        state_dict = torch.load(str(state_path), map_location="cpu", weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net = EmoNet(n_expression=8).to(device)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        self.net = net
        self.device = device
        self.detector = SFDDetector(device)

    @torch.no_grad()
    def predict(self, frames_rgb: np.ndarray) -> tuple[float, float, bool]:
        """frames_rgb: (N, H, W, 3) uint8. Returns (v, a, detected)."""
        vs, as_ = [], []
        for frame in frames_rgb:
            bgr = frame[:, :, ::-1]
            faces = self.detector.detect_from_image(bgr.copy())
            if not faces:
                continue
            # largest face by area
            x1, y1, x2, y2, _ = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2, :]
            crop = cv2.resize(crop, (EMONET_IMAGE_SIZE, EMONET_IMAGE_SIZE))
            t = torch.from_numpy(crop).float().permute(2, 0, 1).to(self.device) / 255.0
            out = self.net(t.unsqueeze(0))
            vs.append(float(out["valence"].clamp(-1, 1).cpu().item()))
            as_.append(float(out["arousal"].clamp(-1, 1).cpu().item()))
        if not vs:
            return float("nan"), float("nan"), False
        return float(np.mean(vs)), float(np.mean(as_)), True


class VeaticPredictor:
    def __init__(self, veatic_repo: Path, weight_path: Path, device: str):
        import importlib.util

        # VEATIC의 model.py는 패키지 `model`과 이름이 겹쳐 일반 import 불가.
        # ViT.py를 먼저 sys.modules['ViT']로 등록한 뒤 model.py를 spec으로 로드한다.
        repo = str(veatic_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        vit_spec = importlib.util.spec_from_file_location("ViT", veatic_repo / "ViT.py")
        vit_mod = importlib.util.module_from_spec(vit_spec)
        sys.modules["ViT"] = vit_mod
        vit_spec.loader.exec_module(vit_mod)  # type: ignore

        model_spec = importlib.util.spec_from_file_location(
            "veatic_model", veatic_repo / "model.py"
        )
        model_mod = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_mod)  # type: ignore
        VEATIC_baseline = model_mod.VEATIC_baseline

        net = VEATIC_baseline(num_frames=VEATIC_NUM_FRAMES, num_classes=2).to(device)
        state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] VEATIC missing keys: {len(missing)} (first 3: {missing[:3]})")
        if unexpected:
            print(f"[warn] VEATIC unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
        net.eval()
        self.net = net
        self.device = device

    @torch.no_grad()
    def predict(self, frames_rgb: np.ndarray) -> tuple[float, float]:
        """frames_rgb: (>=5, H, W, 3) uint8. Returns (v, a)."""
        # re-sample to exactly VEATIC_NUM_FRAMES evenly
        n = frames_rgb.shape[0]
        if n >= VEATIC_NUM_FRAMES:
            idx = [int(round(i * (n - 1) / (VEATIC_NUM_FRAMES - 1))) for i in range(VEATIC_NUM_FRAMES)]
        else:
            idx = list(range(n)) + [n - 1] * (VEATIC_NUM_FRAMES - n)
        selected = frames_rgb[idx]
        resized = np.stack(
            [cv2.resize(f, (VEATIC_FRAME_W, VEATIC_FRAME_H)) for f in selected], axis=0
        )
        tensor = torch.from_numpy(resized).float().permute(0, 3, 1, 2) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
        out = self.net(tensor)  # (1, 2)
        v, a = out[0, 0].cpu().item(), out[0, 1].cpu().item()
        return float(v), float(a)


class ClipZeroShotPredictor:
    def __init__(self, device: str):
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        model = model.to(device).eval()
        self.model = model
        self.preprocess = preprocess
        self.device = device

        def encode(prompts):
            tokens = tokenizer(prompts).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.mean(dim=0, keepdim=True)

        self.v_pos = encode(CLIP_VALENCE_POS)
        self.v_neg = encode(CLIP_VALENCE_NEG)
        self.a_hi = encode(CLIP_AROUSAL_HIGH)
        self.a_lo = encode(CLIP_AROUSAL_LOW)

    @torch.no_grad()
    def predict(self, frames_rgb: np.ndarray) -> tuple[float, float]:
        from PIL import Image

        tensors = []
        for f in frames_rgb:
            pil = Image.fromarray(f)
            tensors.append(self.preprocess(pil))
        batch = torch.stack(tensors, dim=0).to(self.device)
        img_feat = self.model.encode_image(batch)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat.mean(dim=0, keepdim=True)

        v_score = (img_feat @ self.v_pos.T - img_feat @ self.v_neg.T).item()
        a_score = (img_feat @ self.a_hi.T - img_feat @ self.a_lo.T).item()
        # heuristic scale: raw cosine diffs are ~[-0.1, 0.1]. 10x + tanh → [-1, 1]-ish.
        return float(math.tanh(v_score * 10.0)), float(math.tanh(a_score * 10.0))


def run(
    windows_dir: Path,
    emonet_repo: Path,
    veatic_repo: Path,
    veatic_weight: Path,
    output_csv: Path,
    device: str,
    film_ids: set[str] | None = None,
) -> dict:
    print(f"[info] device: {device}")
    print(f"[info] loading EmoNet from {emonet_repo}")
    emonet = EmoNetPredictor(emonet_repo, device)
    print(f"[info] loading VEATIC from {veatic_repo} (weight: {veatic_weight.name})")
    veatic = VeaticPredictor(veatic_repo, veatic_weight, device)
    print(f"[info] loading CLIP {CLIP_MODEL_NAME}/{CLIP_PRETRAINED}")
    clip = ClipZeroShotPredictor(device)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_total = n_err = n_face_ok = 0

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "film_id", "window_id",
            "emonet_v", "emonet_a", "emonet_detected",
            "veatic_v", "veatic_a",
            "clip_v", "clip_a",
        ])

        for film_id, window_id, mp4 in iter_window_mp4s(windows_dir):
            if film_ids is not None and film_id not in film_ids:
                continue
            n_total += 1
            frames = sample_frames(mp4, max(CLIP_NUM_FRAMES, VEATIC_NUM_FRAMES))
            if frames is None:
                print(f"[err] frame sample failed: {mp4.name}")
                n_err += 1
                writer.writerow([film_id, window_id, "", "", "0", "", "", "", ""])
                continue

            try:
                emo_v, emo_a, detected = emonet.predict(frames[:CLIP_NUM_FRAMES])
                v_v, v_a = veatic.predict(frames)
                c_v, c_a = clip.predict(frames[:CLIP_NUM_FRAMES])
            except Exception as e:  # noqa: BLE001
                print(f"[err] inference failed {film_id}/{window_id}: {e}")
                n_err += 1
                writer.writerow([film_id, window_id, "", "", "0", "", "", "", ""])
                continue

            if detected:
                n_face_ok += 1
            writer.writerow([
                film_id, window_id,
                f"{emo_v:.6f}" if not math.isnan(emo_v) else "",
                f"{emo_a:.6f}" if not math.isnan(emo_a) else "",
                "1" if detected else "0",
                f"{v_v:.6f}", f"{v_a:.6f}",
                f"{c_v:.6f}", f"{c_a:.6f}",
            ])

            if n_total % 50 == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed
                print(f"[info] processed {n_total} windows ({rate:.2f}/s, faces ok: {n_face_ok})")

    dt = time.time() - t0
    summary = {
        "total": n_total,
        "errors": n_err,
        "face_detected": n_face_ok,
        "face_detect_rate": n_face_ok / max(1, n_total),
        "elapsed_sec": dt,
    }
    print(f"[done] {summary}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows_dir", type=Path, required=True)
    p.add_argument("--emonet_repo", type=Path, default=Path("third_party/emonet"))
    p.add_argument("--veatic_repo", type=Path, default=Path("third_party/VEATIC"))
    p.add_argument(
        "--veatic_weight",
        type=Path,
        default=Path("third_party/VEATIC/weights/veatic_pretrain.pth"),
    )
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / mps / cpu. auto-detect if omitted")
    p.add_argument("--film_ids", type=str, default=None,
                   help="comma-separated film ids to limit run (e.g. big_buck_bunny,sintel)")
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    film_ids = set(args.film_ids.split(",")) if args.film_ids else None

    run(
        windows_dir=args.windows_dir,
        emonet_repo=args.emonet_repo,
        veatic_repo=args.veatic_repo,
        veatic_weight=args.veatic_weight,
        output_csv=args.output_csv,
        device=device,
        film_ids=film_ids,
    )


if __name__ == "__main__":
    main()
