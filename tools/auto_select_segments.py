"""auto_select_segments.py — 풀 트레일러에서 MUSHRA용 대표 Segment 4개 자동 선정.

트랙 5+ Phase X3. `run_mushra_curated.py`는 씬을 HARDCODED로 관리. 풀 트레일러
기반 평가를 위해 자동 diversity 선정 알고리즘 신규 작성 (Case C).

알고리즘:
  1) timeline.json에서 씬 로드. 풀 트레일러 오디오 파일 경로 입력받음.
  2) 10s 슬라이딩 윈도우 (stride 1s) 생성. 각 윈도우에 포함된 씬 계산.
  3) 윈도우별 diversity 점수 계산:
       - dominant_mood 강도 (aggregated confidence)
       - category diversity (윈도우 내 distinct mood 개수)
       - dialogue density (전 스펙트럼 커버 위해)
       - dissolve transition 포함 여부 (bonus)
  4) 그리디 선정 (4회):
       - 점수 최고 윈도우 픽
       - 해당 윈도우 시간대 + 같은 dominant category 제외 (다양성 보장)
       - 4개 선정 완료 시까지
  5) 각 segment에 대해:
       - original / v3_3 / v3_5_5 오디오 슬라이스 (ffmpeg / numpy slicing)
       - anchor 생성 (generate_anchor.generate_low_anchor)
       - 4조건 LUFS 매칭 (loudness_match.match_loudness, target = v3_3 LUFS)
  6) segment metadata JSON 저장 (plater에서 사용)

출력:
  evaluation/webmushra/configs/resources/audio/trailer_segments/<seg_id>/
    ├── original_matched.wav
    ├── anchor_matched.wav
    ├── v3_3_matched.wav
    └── v3_5_5_matched.wav
  tmp/full_trailer/topgun/segment_metadata.json (player consume용)

실행:
  PYTHONIOENCODING=utf-8 PYTHONPATH=. venv/Scripts/python.exe \\
      tools/auto_select_segments.py --job topgun --n 4 --duration 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.loudness_match import measure_integrated_lufs, match_loudness  # noqa: E402
from tools.generate_anchor import generate_low_anchor  # noqa: E402


# ────────────────────────────────────────────────────────
# 데이터 구조
# ────────────────────────────────────────────────────────
@dataclass
class SegmentCandidate:
    start_sec: float
    end_sec: float
    duration: float
    scene_ids: list[int]
    dominant_category: str
    dominant_prob: float
    mean_density: float
    n_categories: int
    has_dissolve: bool
    score: float

    def describe(self) -> str:
        dens = self.mean_density
        dens_tag = (
            "nodial" if dens < 0.05 else
            "lowdial" if dens < 0.25 else
            "middial" if dens < 0.5 else
            "withdial"
        )
        trans_tag = "dissol" if self.has_dissolve else "cut"
        cat_tag = self.dominant_category.split()[0].lower()[:7]
        return f"{cat_tag}_{dens_tag}_{trans_tag}"


# ────────────────────────────────────────────────────────
# 윈도우 선정 알고리즘
# ────────────────────────────────────────────────────────
def score_window(
    scenes_in_window: list[dict],
    transitions_in_window: list[str],
) -> tuple[float, str, float, float, int, bool]:
    """윈도우 diversity + 강도 점수.

    Returns:
        (score, dominant_category, dominant_prob, mean_density, n_categories, has_dissolve)
    """
    if not scenes_in_window:
        return (0.0, "", 0.0, 0.0, 0, False)

    cats = [s["aggregated"]["category"] for s in scenes_in_window]
    cat_counter = Counter(cats)
    dominant_category, _ = cat_counter.most_common(1)[0]
    n_categories = len(cat_counter)

    # 윈도우 내 dominant category 확률의 최대치 (가장 명확한 씬의 confidence)
    dominant_prob = max(
        s["aggregated"]["mood_probs_mean"].get(s["aggregated"]["category"], 0.0)
        for s in scenes_in_window
    )

    densities = [s["dialogue"]["density"] for s in scenes_in_window]
    mean_density = sum(densities) / len(densities)

    has_dissolve = "dissolve" in transitions_in_window

    # 가중치:
    #   dominant_prob 최대 1.0 → ×10 → 10
    #   category diversity 1~3 → ×3 → 3~9
    #   mean density (0~1) → ×5 → 0~5
    #   dissolve bonus 3
    score = dominant_prob * 10 + n_categories * 3 + mean_density * 5 + (3 if has_dissolve else 0)
    return score, dominant_category, dominant_prob, mean_density, n_categories, has_dissolve


def generate_candidates(
    scenes: list[dict],
    target_duration: float = 10.0,
    stride_sec: float = 1.0,
    trailer_duration: float | None = None,
) -> list[SegmentCandidate]:
    """슬라이딩 윈도우 모든 후보 생성 + 점수 계산."""
    if trailer_duration is None:
        trailer_duration = scenes[-1]["end_sec"]

    candidates: list[SegmentCandidate] = []
    t = 0.0
    while t + target_duration <= trailer_duration + 1e-6:
        win_start, win_end = t, t + target_duration
        # 이 윈도우와 겹치는 씬들
        in_win = [
            s for s in scenes
            if s["end_sec"] > win_start and s["start_sec"] < win_end
        ]
        # 윈도우 내부 경계에서 발생하는 transition (다음 씬 존재 시의 현재 씬 transition_out)
        transitions = [
            s.get("transition_out", "cut")
            for s in in_win
            if win_start < s["end_sec"] < win_end
        ]
        score, dom_cat, dom_prob, mean_d, n_cat, has_dis = score_window(in_win, transitions)
        candidates.append(SegmentCandidate(
            start_sec=round(win_start, 2),
            end_sec=round(win_end, 2),
            duration=target_duration,
            scene_ids=[s["scene_id"] for s in in_win],
            dominant_category=dom_cat,
            dominant_prob=round(dom_prob, 3),
            mean_density=round(mean_d, 3),
            n_categories=n_cat,
            has_dissolve=has_dis,
            score=round(score, 2),
        ))
        t += stride_sec

    return candidates


def select_segments_greedy(
    candidates: list[SegmentCandidate],
    n: int = 4,
    min_gap_sec: float = 1.0,
    enforce_category_diversity: bool = True,
) -> list[SegmentCandidate]:
    """그리디 선정: 점수 높은 순 + 시간 비겹침 + 카테고리 다양성."""
    picked: list[SegmentCandidate] = []
    used_cats: set[str] = set()
    ranked = sorted(candidates, key=lambda c: -c.score)

    for c in ranked:
        if len(picked) >= n:
            break
        # 시간 겹침 체크 (min_gap_sec 여유 포함)
        overlaps = any(
            not (c.end_sec + min_gap_sec <= p.start_sec or c.start_sec >= p.end_sec + min_gap_sec)
            for p in picked
        )
        if overlaps:
            continue
        # 카테고리 다양성
        if enforce_category_diversity and c.dominant_category in used_cats:
            continue
        picked.append(c)
        used_cats.add(c.dominant_category)

    # 4개 부족 시 카테고리 제약 완화
    if len(picked) < n and enforce_category_diversity:
        for c in ranked:
            if len(picked) >= n:
                break
            overlaps = any(
                not (c.end_sec + min_gap_sec <= p.start_sec or c.start_sec >= p.end_sec + min_gap_sec)
                for p in picked
            )
            if overlaps:
                continue
            if c in picked:
                continue
            picked.append(c)

    # 시간순 정렬
    return sorted(picked, key=lambda c: c.start_sec)


# ────────────────────────────────────────────────────────
# 오디오 슬라이싱 + 매칭
# ────────────────────────────────────────────────────────
def slice_wav(input_wav: Path, start_sec: float, end_sec: float, output_wav: Path) -> None:
    """wav에서 [start_sec, end_sec] 구간 발췌하여 output에 저장 (int16 유지)."""
    sr, a = wavfile.read(str(input_wav))
    start_idx = int(start_sec * sr)
    end_idx   = int(end_sec * sr)
    end_idx = min(end_idx, a.shape[0])
    sliced = a[start_idx:end_idx]
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_wav), sr, sliced)


def process_segment(
    seg_idx: int,
    seg: SegmentCandidate,
    full_wavs: dict,
    out_dir: Path,
) -> dict:
    """1개 segment 처리: 슬라이스 + anchor 생성 + 4조건 LUFS 매칭."""
    seg_id = f"seg{seg_idx + 1:02d}_{seg.describe()}"
    sd = out_dir / seg_id
    sd.mkdir(parents=True, exist_ok=True)

    # 1) original / v3_3 / v3_5_5 슬라이스 → wav
    raw_paths: dict[str, Path] = {}
    for name, full_path in full_wavs.items():
        raw_p = sd / f"{name}_raw.wav"
        slice_wav(full_path, seg.start_sec, seg.end_sec, raw_p)
        raw_paths[name] = raw_p

    # 2) anchor 생성 (원본 슬라이스에 3.5kHz LPF)
    anchor_raw = sd / "anchor_raw.wav"
    generate_low_anchor(raw_paths["original"], anchor_raw)
    raw_paths["anchor"] = anchor_raw

    # 3) target LUFS = v3_3 slice
    target = measure_integrated_lufs(raw_paths["v3_3"])
    target_lufs = target["lufs"]

    # 4) 4조건 LUFS 매칭
    match_report: dict[str, dict] = {}
    for cond in ["original", "anchor", "v3_3", "v3_5_5"]:
        src = raw_paths[cond]
        dst = sd / f"{cond}_matched.wav"
        r = match_loudness(src, target_lufs, dst, peak_ceiling_db=-1.0)
        match_report[cond] = {
            "gain_applied_db": round(r["gain_applied_db"], 2),
            "final_lufs":      round(r["final_lufs"], 2),
            "final_peak_db":   round(r["final_peak_db"], 2),
            "delta_to_target": round(r["final_lufs"] - target_lufs, 2),
        }

    # 5) raw 파일 정리 (matched만 유지)
    for p in raw_paths.values():
        try:
            p.unlink()
        except Exception:
            pass

    return {
        "seg_id": seg_id,
        "start_sec": seg.start_sec,
        "end_sec": seg.end_sec,
        "duration": seg.duration,
        "scene_ids": seg.scene_ids,
        "dominant_category": seg.dominant_category,
        "dominant_prob": seg.dominant_prob,
        "mean_density": seg.mean_density,
        "n_categories": seg.n_categories,
        "has_dissolve": seg.has_dissolve,
        "score": seg.score,
        "target_lufs": round(target_lufs, 2),
        "match_report": match_report,
    }


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
JOB_PATHS = {
    "topgun":   REPO / "backend" / "data" / "jobs" / "fe2ecad8-dc25-4131-adfe-ffeea6d977a1",
    "lalaland": REPO / "backend" / "data" / "jobs" / "lalaland-demo",
}
FT_BASE     = REPO / "tmp" / "full_trailer"
SEGMENTS_OUT = REPO / "evaluation" / "webmushra" / "configs" / "resources" / "audio" / "trailer_segments"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="topgun", choices=list(JOB_PATHS.keys()))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=1.0)
    args = ap.parse_args()

    job_dir = JOB_PATHS[args.job]
    ft_dir  = FT_BASE / args.job

    timeline = json.loads((job_dir / "timeline.json").read_text(encoding="utf-8"))
    scenes = timeline["scenes"]
    trailer_dur = scenes[-1]["end_sec"]

    full_wavs = {
        "original": ft_dir / "original.wav",
        "v3_3":     ft_dir / "v3_3.wav",
        "v3_5_5":   ft_dir / "v3_5_5.wav",
    }
    for k, p in full_wavs.items():
        if not p.exists():
            raise FileNotFoundError(f"{k}: {p} 누락 — Phase X1 먼저 실행")

    print(f"=== Auto-select segments for {args.job} ===")
    print(f"  trailer dur: {trailer_dur:.2f}s, scenes: {len(scenes)}")
    print(f"  target segments: {args.n}, window: {args.duration}s, stride: {args.stride}s")

    # 1) 후보 생성
    candidates = generate_candidates(
        scenes, target_duration=args.duration,
        stride_sec=args.stride, trailer_duration=trailer_dur,
    )
    print(f"  candidates generated: {len(candidates)}")

    # 2) 선정
    selected = select_segments_greedy(
        candidates, n=args.n, min_gap_sec=1.0, enforce_category_diversity=True,
    )
    print(f"  selected: {len(selected)}")

    # 3) segment별 처리
    out_base = SEGMENTS_OUT / args.job
    out_base.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, seg in enumerate(selected):
        seg_id = f"seg{i + 1:02d}_{seg.describe()}"
        print(f"\n--- [{i+1}/{len(selected)}] {seg_id}  "
              f"({seg.start_sec:.1f}~{seg.end_sec:.1f}s  "
              f"cat={seg.dominant_category}  density={seg.mean_density:.2f}  "
              f"dissolve={seg.has_dissolve}  score={seg.score:.2f}) ---")
        report = process_segment(i, seg, full_wavs, out_base)
        reports.append(report)
        for cond, r in report["match_report"].items():
            print(f"    {cond:8s}  gain={r['gain_applied_db']:+.2f}dB  "
                  f"LUFS={r['final_lufs']:+.2f} (Δ={r['delta_to_target']:+.2f})  "
                  f"peak={r['final_peak_db']:+.2f}")

    # 4) metadata 저장
    metadata_path = ft_dir / "segment_metadata.json"
    metadata = {
        "job": args.job,
        "trailer_duration": trailer_dur,
        "window_sec": args.duration,
        "n_segments": len(reports),
        "segments": reports,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✓ metadata → {metadata_path.relative_to(REPO)}")
    print(f"✓ 매칭 파일 → {out_base.relative_to(REPO)}/<seg_id>/*_matched.wav")


if __name__ == "__main__":
    main()
