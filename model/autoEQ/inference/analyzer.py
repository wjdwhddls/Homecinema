"""analyzer.py — 백엔드 진입점.

작업 11 (Day 4~5) + Day 10 통합:
- process_job(job_id): 백엔드가 호출하는 메인 진입점
- 입력:  data/jobs/{job_id}/original.{ext}
- 출력:  timeline.json, processed.mp4, processed_v3_1.mp4, processed_v3_2.mp4
- meta.json status: uploaded → analyzing → eq_processing → completed

A 모델 통합은 Day 10에 추가됩니다 (현재는 더미 확률 사용).

경로 정책:
- 워커는 paths.JOBS_DATA_DIR을 단일 진실 공급원으로 사용합니다.
- 백엔드와 같은 디렉토리를 가리키도록 환경변수 JOBS_DATA_DIR을 절대경로로
  세팅하거나, 백엔드를 프로젝트 루트에서 실행하세요 (그러면 backend/data/jobs/
  와 paths.JOBS_DATA_DIR 기본값이 일치).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio_analyzer import compute_dialogue_density, detect_speech
from .eq_engine import (
    BAND_FREQS,
    BAND_Q,
    DEFAULT_PRESET_VERSION,
    MOOD_CATEGORIES,
    PRESET_VERSIONS,
    compute_effective_eq_both_versions,
)
from .mux import mux_video_audio
from .paths import JOBS_DATA_DIR
from .playback import apply_timevarying_eq
from .scene_splitter import detect_scenes_with_transitions, merge_short_scenes
from .utils import get_video_info


# ────────────────────────────────────────────────────────
# 백엔드 storage와 동일한 meta.json 형식 (호환성 보장)
# storage helper를 직접 import하지 않는 이유:
#   백엔드의 config.py는 cwd 기준 './data/jobs'를 쓰기 때문에 워커가
#   다른 cwd에서 실행되면 두 시스템이 다른 디렉토리를 보게 됨.
#   여기서는 환경변수 JOBS_DATA_DIR(절대경로)을 기준으로 통일.
# ────────────────────────────────────────────────────────
def _job_dir(job_id: str) -> Path:
    return JOBS_DATA_DIR / job_id


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def _load_meta(job_id: str) -> dict:
    p = _meta_path(job_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_meta(job_id: str, meta: dict) -> None:
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    p = _meta_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def update_status(
    job_id: str,
    status: str,
    error: str | None = None,
    progress: float | None = None,
) -> None:
    """meta.json 상태 갱신. 백엔드 schemas.py의 JobStatus enum과 일치."""
    meta = _load_meta(job_id)
    meta["status"] = status
    if error is not None:
        meta["error_message"] = error
    if progress is not None:
        meta["analysis_progress"] = progress
    _save_meta(job_id, meta)


# ────────────────────────────────────────────────────────
# 더미 A 모델 출력 (Day 10에서 실제 Gate Network 호출로 교체)
# ────────────────────────────────────────────────────────
def dummy_a_model_output(scenes: list[dict]) -> None:
    """A 모델이 통합되기 전 임시 더미.

    씬 인덱스를 기반으로 7개 mood 카테고리를 순환시켜 시각화/통합 테스트가
    의미 있게 보이도록 합니다 (모든 씬이 같은 EQ가 되지 않도록).

    Day 10에 실제 Gate Network 호출로 교체.
    """
    n_cats = len(MOOD_CATEGORIES)
    for s in scenes:
        idx = s["scene_id"] % n_cats
        dominant = MOOD_CATEGORIES[idx]
        secondary = MOOD_CATEGORIES[(idx + 1) % n_cats]

        # 7개 카테고리에 확률 분포 (지배 0.65 + 부 0.20 + 나머지 균등)
        probs = {cat: 0.03 for cat in MOOD_CATEGORIES}
        probs[dominant] = 0.65
        probs[secondary] = 0.20

        s["aggregated"] = {
            "valence": 0.0,
            "arousal": 0.0,
            "category": dominant,
            "mood_probs_mean": probs,
            "gate_weights_mean": {"w_video": 0.5, "w_audio": 0.5},
        }


# ────────────────────────────────────────────────────────
# 메인 진입점
# ────────────────────────────────────────────────────────
def process_job(job_id: str) -> None:
    """백엔드 진입점.

    입력: data/jobs/{job_id}/original.{ext}
    출력:
      - timeline.json
      - processed.mp4 (canonical = DEFAULT_PRESET_VERSION 복사본)
      - processed_v3_1.mp4
      - processed_v3_2.mp4
    """
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise FileNotFoundError(f"Job 디렉토리 없음: {job_dir}")

    video_candidates = list(job_dir.glob("original.*"))
    if not video_candidates:
        raise FileNotFoundError(f"original.* 파일 없음 in {job_dir}")
    video_path = video_candidates[0]

    try:
        # ════════════════════════════════════════════
        # 1단계: 분석
        # ════════════════════════════════════════════
        update_status(job_id, "analyzing", progress=0.0)
        print(f"[{job_id}] 분석 시작...")

        video_info = get_video_info(str(video_path))
        if not video_info["has_audio"]:
            raise ValueError("오디오 트랙이 없는 영상은 처리 불가")
        video_duration = video_info["duration"]

        # 1a. 오디오 추출
        audio_32k = job_dir / "audio_32k.wav"
        audio_orig = job_dir / "audio_original.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn",
             "-ar", "32000", "-ac", "1", str(audio_32k)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", str(audio_orig)],
            check=True, capture_output=True,
        )
        update_status(job_id, "analyzing", progress=0.2)

        # 1b. 씬 분할 + 병합
        scenes_raw = detect_scenes_with_transitions(str(video_path))
        scenes = merge_short_scenes(scenes_raw, 2.0, preserve_dissolve=True)
        update_status(job_id, "analyzing", progress=0.4)

        # 1c. VAD + dialogue density
        speech = detect_speech(str(audio_32k))
        for s in scenes:
            density, segs = compute_dialogue_density(
                s["start_sec"], s["end_sec"], speech
            )
            s["dialogue"] = {
                "density": density,
                "segments_in_scene": segs,
            }

        # 1d. PANNs 윈도우 임베딩 (Day 10에 추가)
        # 1e. A의 Gate Network 호출 (Day 10에 교체)
        dummy_a_model_output(scenes)
        update_status(job_id, "analyzing", progress=0.7)

        # ════════════════════════════════════════════
        # 2단계: EQ 계산 (V3.1 + V3.2 동시)
        # ════════════════════════════════════════════
        update_status(job_id, "eq_processing", progress=0.0)

        for s in scenes:
            gains_both = compute_effective_eq_both_versions(
                s["aggregated"]["mood_probs_mean"],
                s["dialogue"]["density"],
                alpha_d=0.5, intensity=1.0,
            )
            default_gains = gains_both[DEFAULT_PRESET_VERSION]

            s["eq_preset"] = {
                "effective_bands": [
                    {"band_index": i + 1, "freq_hz": BAND_FREQS[i],
                     "gain_db": round(float(default_gains[i]), 2),
                     "q": BAND_Q[i]}
                    for i in range(10)
                ],
                "alt_versions": {
                    version: [
                        {"band_index": i + 1, "freq_hz": BAND_FREQS[i],
                         "gain_db": round(float(gains[i]), 2),
                         "q": BAND_Q[i]}
                        for i in range(10)
                    ]
                    for version, gains in gains_both.items()
                    if version != DEFAULT_PRESET_VERSION
                },
            }

        # timeline.json
        timeline = {
            "schema_version": "1.1",
            "metadata": {
                "video_filename": video_path.name,
                "video_duration_sec": video_duration,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "model_version": "spec_v3.2",
                "eq_preset_version": DEFAULT_PRESET_VERSION,
                "eq_preset_alt_versions": [
                    v for v in PRESET_VERSIONS.keys()
                    if v != DEFAULT_PRESET_VERSION
                ],
            },
            "scenes": scenes,
        }
        (job_dir / "timeline.json").write_text(
            json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        update_status(job_id, "eq_processing", progress=0.3)

        # ════════════════════════════════════════════
        # 3단계: 시간축 EQ 적용 (V3.1 + V3.2 동시)
        # ════════════════════════════════════════════
        scenes_eq_for_audio: dict[str, list[dict]] = {}
        for version in PRESET_VERSIONS.keys():
            scenes_eq_for_audio[version] = []
            for s in scenes:
                bands = (
                    s["eq_preset"]["effective_bands"]
                    if version == DEFAULT_PRESET_VERSION
                    else s["eq_preset"]["alt_versions"][version]
                )
                gains = np.array([b["gain_db"] for b in bands])
                scenes_eq_for_audio[version].append({
                    "start_sec": s["start_sec"],
                    "end_sec": s["end_sec"],
                    "transition_out": s["transition_out"],
                    "effective_gains": gains,
                })

        for version in PRESET_VERSIONS.keys():
            audio_eq = job_dir / f"audio_eq_{version}.wav"
            apply_timevarying_eq(
                str(audio_orig), str(audio_eq), scenes_eq_for_audio[version]
            )
            processed_path = job_dir / f"processed_{version}.mp4"
            mux_video_audio(str(video_path), str(audio_eq), str(processed_path))
            audio_eq.unlink(missing_ok=True)
            print(f"  ✓ {processed_path.name} 생성")

        update_status(job_id, "eq_processing", progress=0.9)

        # 백엔드 호환용 canonical processed.mp4 (재실행 시에도 항상 덮어씀)
        default_processed = job_dir / f"processed_{DEFAULT_PRESET_VERSION}.mp4"
        canonical = job_dir / "processed.mp4"
        if default_processed.exists():
            shutil.copy(default_processed, canonical)

        # 임시 wav 정리
        audio_32k.unlink(missing_ok=True)
        audio_orig.unlink(missing_ok=True)

        # processed_size_bytes 갱신 (모바일 JobStatusResponse에서 사용)
        meta = _load_meta(job_id)
        if canonical.exists():
            meta["processed_size_bytes"] = canonical.stat().st_size
        _save_meta(job_id, meta)

        update_status(job_id, "completed", progress=1.0)
        print(f"[{job_id}] 완료 — V3.1, V3.2 두 버전 모두 생성")

    except Exception as e:
        update_status(job_id, "failed", error=str(e))
        print(f"[{job_id}] 실패: {e}")
        raise


# ────────────────────────────────────────────────────────
# 검증 — 워커 산출물 확인
# ────────────────────────────────────────────────────────
def verify_worker(job_id: str) -> None:
    """워커 실행 결과 검증."""
    job_dir = _job_dir(job_id)

    assert (job_dir / "timeline.json").exists(), "timeline.json 없음"
    assert (job_dir / "processed.mp4").exists(), "processed.mp4 없음"
    assert (job_dir / "meta.json").exists(), "meta.json 없음"
    assert (job_dir / "processed_v3_1.mp4").exists()
    assert (job_dir / "processed_v3_2.mp4").exists()

    meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed", f"상태: {meta['status']}"

    timeline = json.loads((job_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["schema_version"] == "1.1"
    assert "video_duration_sec" in timeline["metadata"]
    assert len(timeline["scenes"]) > 0

    for s in timeline["scenes"]:
        assert "eq_preset" in s
        assert "effective_bands" in s["eq_preset"]
        assert len(s["eq_preset"]["effective_bands"]) == 10
        assert "aggregated" in s
        assert "mood_probs_mean" in s["aggregated"]
        assert "valence" in s["aggregated"]

    info = get_video_info(str(job_dir / "processed.mp4"))
    assert info["has_audio"], "오디오 트랙 없음"

    print(f"  ✓ 산출물 5개 모두 생성")
    print(f"  ✓ 상태: completed")
    print(f"  ✓ {len(timeline['scenes'])}개 씬 분석됨")
    print(f"  ✓ 영상: {info['width']}x{info['height']}, {info['duration']:.1f}초")


# ────────────────────────────────────────────────────────
# 직접 실행 (Stage 1 테스트: 워커 단독)
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m model.autoEQ.inference.analyzer <job_id>")
        print("  먼저 data/jobs/<job_id>/original.mp4 를 준비하세요.")
        sys.exit(1)

    job_id = sys.argv[1]
    print(f"=== process_job({job_id}) ===")
    process_job(job_id)

    print(f"\n=== verify_worker({job_id}) ===")
    verify_worker(job_id)
