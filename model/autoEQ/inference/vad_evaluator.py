"""vad_evaluator.py — Silero VAD F1 측정.

Day 13 (작업 13):
- 수동 라벨링한 ground truth와 Silero VAD 결과를 비교
- frame-level precision / recall / F1 계산
- 목표: F1 ≥ 0.85

Ground truth 라벨링 권장:
- Audacity로 wav 열기 → Label Track으로 대사 구간 마킹 → Export Labels
- 형식: <start_sec>\t<end_sec>\t<label> (label은 무엇이든 OK)
- 또는 직접 Python list로 작성: [{"start": 1.2, "end": 3.5}, ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .audio_analyzer import detect_speech
from .paths import OUTPUTS_DIR, audio_path


def load_audacity_labels(label_path) -> list[dict]:
    """Audacity Export Labels 형식 (.txt) 로드."""
    segments = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                start = float(parts[0])
                end = float(parts[1])
                segments.append({"start": start, "end": end})
    return segments


def segments_to_frames(segments: list[dict], duration: float, frame_ms: int = 10) -> np.ndarray:
    """[{start, end}, ...] → 프레임 단위 binary mask (1=speech, 0=non)."""
    n_frames = int(duration * 1000 / frame_ms)
    mask = np.zeros(n_frames, dtype=int)
    for seg in segments:
        start_frame = int(seg["start"] * 1000 / frame_ms)
        end_frame = int(seg["end"] * 1000 / frame_ms)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        mask[start_frame:end_frame] = 1
    return mask


def compute_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Frame-level precision / recall / F1 + accuracy."""
    n = min(len(gt_mask), len(pred_mask))
    gt = gt_mask[:n]
    pred = pred_mask[:n]

    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    tn = int(np.sum((gt == 0) & (pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    return {
        "n_frames": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "speech_ratio_gt": round(gt.mean(), 4),
        "speech_ratio_pred": round(pred.mean(), 4),
    }


def evaluate_vad(
    audio_file,
    ground_truth_segments: list[dict],
    threshold: float = 0.5,
    frame_ms: int = 10,
) -> dict:
    """단일 오디오 파일에 대해 VAD F1 평가."""
    from .utils import get_duration

    duration = get_duration(audio_file)

    # Silero VAD 예측
    pred_segments = detect_speech(audio_file, threshold=threshold)

    # 프레임 마스크 변환
    gt_mask = segments_to_frames(ground_truth_segments, duration, frame_ms)
    pred_mask = segments_to_frames(pred_segments, duration, frame_ms)

    metrics = compute_metrics(gt_mask, pred_mask)
    metrics["audio_file"] = str(audio_file)
    metrics["duration_sec"] = round(duration, 2)
    metrics["threshold"] = threshold
    metrics["n_gt_segments"] = len(ground_truth_segments)
    metrics["n_pred_segments"] = len(pred_segments)

    return metrics


def threshold_sweep(
    audio_file, ground_truth_segments: list[dict],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """여러 threshold에서 F1 스윕 (최적 threshold 탐색)."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []
    print(f"\n  [Threshold 스윕]")
    for th in thresholds:
        m = evaluate_vad(audio_file, ground_truth_segments, threshold=th)
        results.append(m)
        print(
            f"    th={th}: P={m['precision']:.3f}, R={m['recall']:.3f}, "
            f"F1={m['f1']:.3f}"
        )

    best = max(results, key=lambda x: x["f1"])
    print(f"\n  ✓ 최적 threshold = {best['threshold']} (F1={best['f1']:.3f})")
    return results


def evaluate_all(audio_gt_pairs: list[tuple]) -> dict:
    """여러 영상의 평균 F1.

    Args:
        audio_gt_pairs: [(audio_path, [{start, end}, ...]), ...]
    """
    print("=" * 60)
    print("VAD F1 평가")
    print("=" * 60)

    all_metrics = []
    for audio_file, gt_segments in audio_gt_pairs:
        if not Path(audio_file).exists():
            print(f"\n⏭ {audio_file} 없음, 스킵")
            continue
        if not gt_segments:
            print(f"\n⏭ {audio_file}: ground truth 없음, 스킵")
            continue

        print(f"\n=== {audio_file} (GT: {len(gt_segments)}개 구간) ===")
        m = evaluate_vad(audio_file, gt_segments)
        print(
            f"  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}"
        )
        print(
            f"  Precision = {m['precision']:.4f}"
            f"\n  Recall    = {m['recall']:.4f}"
            f"\n  F1        = {m['f1']:.4f}"
            f"\n  Accuracy  = {m['accuracy']:.4f}"
        )
        all_metrics.append(m)

    if not all_metrics:
        print("\n  ⚠️ 평가 가능한 데이터 없음")
        return {}

    # 매크로 평균
    macro = {
        "macro_precision": round(np.mean([m["precision"] for m in all_metrics]), 4),
        "macro_recall": round(np.mean([m["recall"] for m in all_metrics]), 4),
        "macro_f1": round(np.mean([m["f1"] for m in all_metrics]), 4),
        "n_files": len(all_metrics),
    }

    print(f"\n=== 매크로 평균 ({len(all_metrics)}개 파일) ===")
    print(f"  Precision = {macro['macro_precision']:.4f}")
    print(f"  Recall    = {macro['macro_recall']:.4f}")
    print(f"  F1        = {macro['macro_f1']:.4f}")

    if macro["macro_f1"] >= 0.85:
        print(f"\n  ✓ 목표 F1 ≥ 0.85 달성")
    else:
        print(
            f"\n  ⚠️ F1 {macro['macro_f1']:.4f} < 0.85 — threshold_sweep으로 최적화 검토"
        )

    # JSON 저장
    out_dir = OUTPUTS_DIR / "vad_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vad_metrics.json"
    out_path.write_text(
        json.dumps({"per_file": all_metrics, "macro": macro}, indent=2),
        encoding="utf-8",
    )
    print(f"\n  💾 결과 저장: {out_path}")

    return {"per_file": all_metrics, "macro": macro}


# ────────────────────────────────────────────────────────
# 사용 예시
# ────────────────────────────────────────────────────────
"""
사용 방법:

1. Audacity로 wav 파일 열고 대사 구간 마킹
   → File > Export > Export Labels
   → labels_topgun.txt 같은 형식으로 저장

2. Python에서 평가:
   from model.autoEQ.inference.vad_evaluator import (
       load_audacity_labels, evaluate_all
   )
   from model.autoEQ.inference.paths import audio_path

   gt_topgun = load_audacity_labels("labels_topgun.txt")
   gt_lalaland = load_audacity_labels("labels_lalaland.txt")

   evaluate_all([
       (audio_path("topgun", "32k"), gt_topgun),
       (audio_path("lalaland", "32k"), gt_lalaland),
   ])

3. 또는 직접 라벨 정의:
   gt = [{"start": 1.2, "end": 3.5}, {"start": 5.8, "end": 7.2}, ...]
"""


if __name__ == "__main__":
    print("vad_evaluator는 직접 데이터를 넘겨주세요:")
    print()
    print("# Audacity 라벨 사용:")
    print("  python -m model.autoEQ.inference.vad_evaluator \\")
    print("    <audio_file> <audacity_labels.txt>")
    print()
    print("# 또는 import해서 evaluate_all() 호출")

    if len(sys.argv) == 3:
        audio_file, label_file = sys.argv[1], sys.argv[2]
        gt = load_audacity_labels(label_file)
        print(f"\nGT {len(gt)}개 구간 로드")
        evaluate_all([(audio_file, gt)])
