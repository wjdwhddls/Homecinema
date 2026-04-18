"""run_mushra_curated.py — webMUSHRA 큐레이션 모드 드라이버.

Day 11~14 정식 평가용. 사용자 큐레이션 6 씬 (탑건 3 + 라라랜드 3, 각 8~12초)으로
24 클립 + YAML 생성. mushra_generator.py 무수정.

전략 (run_mushra_demo.py 패턴 승계):
- 원본/V3.1/V3.2 오디오는 analyzer가 이미 생성한 processed_v3_*.mp4에서 발췌
  (시변 EQ 결과를 그대로 평가하기 위함. mushra_generator의 균일-EQ 경로는 사용 안 함.)
- Anchor는 원본 클립에 저역 -6dB cut (make_anchor_eq 재사용).
- 길이 정렬 + webMUSHRA 12s 제한 충족 확인.
- 신규 씬 디렉토리 prefix는 `scene_` — 기존 `trailer_*_full/`(run_mushra_demo.py
  산출물)와 병존.

주의: generate_webmushra_yaml의 testId는 "mood_eq_v1" 하드코딩. 큐레이션 결과와
데모 결과가 동일 폴더(results/mood_eq_v1/)로 떨어지는 충돌을 피하려고,
main() 시작 시점에 기존 폴더를 "mood_eq_v1_demo_backup_<YYYYMMDD_HHMMSS>/"로
자동 이관한다 (backup_existing_results).

실행: PYTHONPATH=. venv/Scripts/python.exe tools/run_mushra_curated.py
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from model.autoEQ.inference.mushra_generator import (
    generate_webmushra_yaml,
    make_anchor_eq,
    verify_clip_set,
)
from model.autoEQ.inference.paths import (
    EVALUATION_RESULTS_DIR,
    MUSHRA_CLIPS_DIR,
    WEBMUSHRA_CONFIG_DIR,
    WEBMUSHRA_DIR,
    ensure_dirs,
)
from model.autoEQ.inference.playback import apply_eq_to_file


SOURCES = {
    "topgun": {
        "trailer": "data/trailers/trailer_topgun.mp4",
        "v3_1":    "backend/data/jobs/fe2ecad8-dc25-4131-adfe-ffeea6d977a1/processed_v3_1.mp4",
        "v3_2":    "backend/data/jobs/fe2ecad8-dc25-4131-adfe-ffeea6d977a1/processed_v3_2.mp4",
    },
    "lalaland": {
        "trailer": "data/trailers/trailer_lalaland.mp4",
        "v3_1":    "backend/data/jobs/lalaland-demo/processed_v3_1.mp4",
        "v3_2":    "backend/data/jobs/lalaland-demo/processed_v3_2.mp4",
    },
}

# 큐레이션 씬 (PR 본문 표 그대로). (scene_name, start_sec, end_sec)
CURATED_SCENES: dict[str, list[tuple[str, float, float]]] = {
    "topgun": [
        ("baseline",          8.0,  16.0),   # 임팩트 약한 오프닝 - EQ 과도성 검증
        ("dialogue_protect", 35.0,  47.0),   # 대사+음악 혼재 - V3.2 차별점
        ("category_eq",      69.0,  80.0),   # Tension/Power 카테고리 EQ
    ],
    "lalaland": [
        ("baseline",       0.0,   9.0),      # EQ 과도성 검증
        ("song_dialogue", 86.0,  95.0),      # VAD 한계 + V3.2
        ("wonder",       100.0, 111.0),      # Wonder 카테고리
    ],
}

WEBMUSHRA_MAX_DURATION = 12.0  # webMUSHRA 12s 제한

# generate_webmushra_yaml이 하드코딩한 testId — 결과 충돌 회피용
WEBMUSHRA_RESULTS_TEST_ID = "mood_eq_v1"


def backup_existing_results() -> Path | None:
    """기존 results/mood_eq_v1/ 존재 시 타임스탬프 붙여 이관.

    testId 하드코딩 때문에 데모/큐레이션 결과가 같은 폴더로 떨어지는 충돌 회피.
    Returns: 백업 경로 (있으면), 없으면 None.
    """
    results_root = WEBMUSHRA_DIR / "results" / WEBMUSHRA_RESULTS_TEST_ID
    if not results_root.exists():
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = (
        WEBMUSHRA_DIR / "results"
        / f"{WEBMUSHRA_RESULTS_TEST_ID}_demo_backup_{ts}"
    )
    shutil.move(str(results_root), str(backup_root))
    print(f"  🗄  기존 결과 백업: {results_root.name} → {backup_root.name}")
    return backup_root


def extract_audio_segment(
    video_path: str | Path,
    wav_path: Path,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 48000,
) -> None:
    """ffmpeg로 [start_sec, end_sec) 구간만 스테레오 wav로 발췌."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(video_path),
            "-t", f"{end_sec - start_sec:.3f}",
            "-vn",
            "-ar", str(sample_rate),
            "-ac", "2",
            "-acodec", "pcm_s16le",
            str(wav_path),
        ],
        check=True, capture_output=True,
    )


def build_scene_clip_set(
    video_key: str,
    scene_name: str,
    start_sec: float,
    end_sec: float,
    sources: dict[str, str],
) -> dict:
    scene_dir_name = f"scene_{video_key}_{scene_name}"
    out_dir = MUSHRA_CLIPS_DIR / scene_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    original = out_dir / "original.wav"
    v3_1     = out_dir / "v3_1.wav"
    v3_2     = out_dir / "v3_2.wav"
    anchor   = out_dir / "anchor.wav"

    print(f"  [{video_key}/{scene_name}] {start_sec:.1f}~{end_sec:.1f}s")
    extract_audio_segment(sources["trailer"], original, start_sec, end_sec)
    extract_audio_segment(sources["v3_1"],    v3_1,     start_sec, end_sec)
    extract_audio_segment(sources["v3_2"],    v3_2,     start_sec, end_sec)
    apply_eq_to_file(original, anchor, make_anchor_eq())

    from model.autoEQ.inference.utils import get_audio_info
    paths_to_align = [original, v3_1, v3_2, anchor]
    durations = [float(get_audio_info(p)["duration"]) for p in paths_to_align]
    target = min(durations)
    if max(durations) - target > 0.01:
        rounded = [round(d, 3) for d in durations]
        print(f"    길이 정렬: target={target:.3f}s (원래 {rounded})")
        for p in paths_to_align:
            tmp = p.with_suffix(".trim.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-t", f"{target:.3f}",
                 "-acodec", "pcm_s16le", str(tmp)],
                check=True, capture_output=True,
            )
            tmp.replace(p)

    final_duration = float(get_audio_info(original)["duration"])
    if final_duration > WEBMUSHRA_MAX_DURATION:
        raise ValueError(
            f"{scene_dir_name}: 최종 길이 {final_duration:.2f}s > "
            f"{WEBMUSHRA_MAX_DURATION}s (webMUSHRA 제한)"
        )

    return {
        "scene_id": scene_dir_name,
        "scene_dir_name": scene_dir_name,
        "duration_sec": round(final_duration, 2),
        "mood": scene_name,
        "density": 0.0,
        "clips": {
            "original": original.name,
            "v3_1":     v3_1.name,
            "v3_2":     v3_2.name,
            "anchor":   anchor.name,
        },
    }


def main() -> None:
    ensure_dirs()

    backup = backup_existing_results()
    if backup is None:
        print("  (기존 results/mood_eq_v1/ 없음 — 백업 스킵)")

    clip_sets: list[dict] = []

    for video_key, scenes in CURATED_SCENES.items():
        sources = SOURCES[video_key]
        for src_key, src_path in sources.items():
            if not Path(src_path).exists():
                raise FileNotFoundError(f"{video_key}/{src_key} 입력 없음: {src_path}")
        print(f"\n=== {video_key} ===")
        for scene_name, start, end in scenes:
            cs = build_scene_clip_set(video_key, scene_name, start, end, sources)
            try:
                verify_clip_set(cs)
            except AssertionError as e:
                print(f"    ⚠ verify 경고: {e}")
            clip_sets.append(cs)

    yaml_path = WEBMUSHRA_CONFIG_DIR / "mood_eq_curated.yaml"
    generate_webmushra_yaml(
        clip_sets, yaml_path, testname="Mood-EQ Curated (6 scenes)"
    )

    print("\n=== 완료 ===")
    print(f"YAML: {yaml_path}")
    print(f"총 {len(clip_sets)}개 씬 × 4 클립 = {len(clip_sets) * 4}개 wav")
    for cs in clip_sets:
        print(f"  {cs['scene_dir_name']} — {cs['duration_sec']}s")


if __name__ == "__main__":
    main()
