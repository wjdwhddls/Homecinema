"""Layer 1 Audio 앙상블 — Essentia 3 V/A 모델 추론.

각 4s window WAV (16kHz mono)에 대해:
  1. MSD-MusiCNN embedding (200-dim) 계산 (1회, 공유)
  2. DEAM / emoMusic / MuSe 3개 head로 (V, A) 예측
  3. [1, 9] → [-1, 1] 정규화
  4. 결과를 <window_id, model, valence, arousal> CSV로 저장

각 모델별 공식 CCC (모델 메타에 기재):
  - DEAM: V=0.778, A=0.647
  - emoMusic: (각 json 참조)
  - MuSe: (각 json 참조)

Usage:
    python -m model.autoEQ.pseudo_label.layer1_essentia \\
        --windows_dir dataset/autoEQ/CCMovies/windows \\
        --weights_dir model/autoEQ/pseudo_label/weights \\
        --output_csv dataset/autoEQ/CCMovies/labels/layer1_essentia.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable


ESSENTIA_MODELS = [
    ("deam", "deam-msd-musicnn-2.pb"),
    ("emomusic", "emomusic-msd-musicnn-2.pb"),
    ("muse", "muse-msd-musicnn-2.pb"),
]

EMBEDDING_MODEL = "msd-musicnn-1.pb"
EMBEDDING_OUTPUT_LAYER = "model/dense/BiasAdd"
HEAD_OUTPUT_LAYER = "model/Identity"
TARGET_SR = 16000


def _scale_19_to_pm1(x: float) -> float:
    """[1, 9] → [-1, 1]."""
    return (x - 5.0) / 4.0


def iter_window_wavs(windows_dir: Path) -> Iterable[tuple[str, str, Path]]:
    """(film_id, window_id, wav_path) 시퀀스. film_id = 디렉터리명, window_id = stem."""
    for film_dir in sorted(windows_dir.iterdir()):
        if not film_dir.is_dir():
            continue
        for wav in sorted(film_dir.glob("*.wav")):
            yield film_dir.name, wav.stem, wav


def run(
    windows_dir: Path,
    weights_dir: Path,
    output_csv: Path,
    film_ids: set[str] | None = None,
) -> dict:
    """전체 window 돌면서 3 모델 inference."""
    # lazy import — essentia는 무거움
    from essentia.standard import (
        MonoLoader,
        TensorflowPredictMusiCNN,
        TensorflowPredict2D,
    )

    emb_path = weights_dir / EMBEDDING_MODEL
    assert emb_path.is_file(), f"Embedding model not found: {emb_path}"

    # embedding model 1회 로드
    print(f"[info] loading embedding model: {emb_path.name}")
    emb_predictor = TensorflowPredictMusiCNN(
        graphFilename=str(emb_path), output=EMBEDDING_OUTPUT_LAYER
    )

    # 3 head model 로드
    head_predictors: dict[str, object] = {}
    for ds_name, pb_name in ESSENTIA_MODELS:
        head_path = weights_dir / pb_name
        assert head_path.is_file(), f"Head model not found: {head_path}"
        print(f"[info] loading {ds_name}: {pb_name}")
        head_predictors[ds_name] = TensorflowPredict2D(
            graphFilename=str(head_path), output=HEAD_OUTPUT_LAYER
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "film_id", "window_id",
        "deam_v", "deam_a",
        "emomusic_v", "emomusic_a",
        "muse_v", "muse_a",
    ]

    n_total = 0
    n_ok = 0
    n_skip = 0
    n_err = 0
    t_start = time.time()

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for film_id, window_id, wav_path in iter_window_wavs(windows_dir):
            if film_ids and film_id not in film_ids:
                n_skip += 1
                continue
            n_total += 1
            try:
                audio = MonoLoader(
                    filename=str(wav_path), sampleRate=TARGET_SR, resampleQuality=4
                )()
                if len(audio) < TARGET_SR:  # < 1s → 의심스러운 window
                    raise ValueError(f"audio too short: {len(audio)} samples")

                embeddings = emb_predictor(audio)  # (N_frames_emb, 200)

                row: dict = {"film_id": film_id, "window_id": window_id}
                for ds_name, _ in ESSENTIA_MODELS:
                    preds = head_predictors[ds_name](embeddings)  # (N_frames, 2)
                    # essentia convention: columns are (valence, arousal)
                    v_raw = float(preds[:, 0].mean())
                    a_raw = float(preds[:, 1].mean())
                    row[f"{ds_name}_v"] = _scale_19_to_pm1(v_raw)
                    row[f"{ds_name}_a"] = _scale_19_to_pm1(a_raw)
                writer.writerow(row)
                n_ok += 1
                if n_ok % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = n_ok / elapsed
                    print(f"  [{n_ok}/{n_total}] {elapsed:.0f}s  {rate:.1f} win/s  {film_id}/{window_id}")
            except Exception as e:
                n_err += 1
                print(f"  ERROR {film_id}/{window_id}: {e}", file=sys.stderr)

    elapsed = time.time() - t_start
    summary = {
        "total_seen": n_total,
        "ok": n_ok,
        "skipped": n_skip,
        "errors": n_err,
        "elapsed_sec": elapsed,
        "output_csv": str(output_csv),
    }
    print(f"\n[info] OK={n_ok} err={n_err} skipped={n_skip}  {elapsed:.1f}s")
    print(f"[info] output: {output_csv}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Layer 1 audio ensemble — Essentia 3 heads")
    p.add_argument("--windows_dir", type=Path, required=True)
    p.add_argument("--weights_dir", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--film_ids", type=str, default="",
                   help="comma-separated film IDs to process (default: all)")
    args = p.parse_args(argv)

    film_ids = None
    if args.film_ids:
        film_ids = {x.strip() for x in args.film_ids.split(",") if x.strip()}

    run(args.windows_dir, args.weights_dir, args.output_csv, film_ids=film_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
