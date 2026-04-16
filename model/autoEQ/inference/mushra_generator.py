"""mushra_generator.py — webMUSHRA 평가용 클립 자동 생성.

작업 13~14 (Day 11~12):
- 4가지 조건(원본/V3.1/V3.2/Anchor) × 영상별 4~6개 씬 = 16~24개 클립
- webMUSHRA용 YAML config 자동 생성
- HTTP 서버는 webMUSHRA/ 디렉토리에서 별도 실행

평가 시나리오:
  - V3.1 vs V3.2 직접 비교
  - 원본 = hidden reference
  - Anchor (저역만 -6dB cut) = 명백한 저품질 기준
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .eq_engine import (
    BAND_FREQS,
    BAND_Q,
    EQ_PRESETS_V3_1,
    EQ_PRESETS_V3_2,
    compute_effective_eq,
    manual_label_to_probs,
)
from .paths import (
    MUSHRA_CLIPS_DIR,
    WEBMUSHRA_CONFIG_DIR,
    WEBMUSHRA_DIR,
    audio_path,
    ensure_dirs,
    trailer_path,
)
from .playback import apply_eq_to_file


def extract_audio_clip(audio_file, start_sec: float, end_sec: float, output_path) -> None:
    """ffmpeg로 오디오 클립 추출 (디스크 공간 절약 위해 wav가 아닌 wav 사용)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(audio_file),
            "-t", str(end_sec - start_sec),
            "-acodec", "pcm_s16le",
            str(output_path),
        ],
        check=True, capture_output=True,
    )


def make_anchor_eq() -> np.ndarray:
    """Anchor: B1~B3 (저역) -6dB cut. 명백히 저품질 기준."""
    gains = np.zeros(10)
    gains[0:3] = -6.0
    return gains


def generate_clips_for_scene(
    video_key: str,
    scene_idx: int,
    start_sec: float,
    end_sec: float,
    mood: str,
    prob: float,
    density: float,
) -> dict:
    """단일 씬에 대해 4가지 조건 클립 생성.

    Returns: 생성된 클립 경로 dict
    """
    short_name = video_key.replace("trailer_", "")
    audio_orig = audio_path(short_name, "original")

    if not audio_orig.exists():
        raise FileNotFoundError(f"{audio_orig} 없음. audio_extractor 먼저 실행")

    scene_dir_name = f"{video_key}_scene{scene_idx:02d}"
    out_dir = MUSHRA_CLIPS_DIR / scene_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 원본
    original_clip = out_dir / "original.wav"
    extract_audio_clip(audio_orig, start_sec, end_sec, original_clip)

    # 2~3. V3.1, V3.2
    probs = manual_label_to_probs(mood, prob)
    clips = {"original": original_clip.name}

    for version_name, presets in [("v3_1", EQ_PRESETS_V3_1), ("v3_2", EQ_PRESETS_V3_2)]:
        gains = compute_effective_eq(
            probs, density, alpha_d=0.5, intensity=1.0, presets=presets
        )
        eq_clip = out_dir / f"{version_name}.wav"
        apply_eq_to_file(original_clip, eq_clip, gains)
        clips[version_name] = eq_clip.name

    # 4. Anchor
    anchor_gains = make_anchor_eq()
    anchor_clip = out_dir / "anchor.wav"
    apply_eq_to_file(original_clip, anchor_clip, anchor_gains)
    clips["anchor"] = anchor_clip.name

    return {
        "scene_id": scene_dir_name,
        "scene_dir_name": scene_dir_name,
        "duration_sec": round(end_sec - start_sec, 2),
        "mood": mood,
        "density": density,
        "clips": clips,
    }


def generate_webmushra_yaml(
    scene_clips: list[dict], output_yaml, testname: str = "MoodEQ Listening Test"
) -> None:
    """webMUSHRA용 YAML config 자동 생성.

    YAML 라이브러리 없이 직접 작성 (의존성 최소화).
    """
    Path(output_yaml).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "testname: " + testname,
        "testId: mood_eq_v1",
        "bufferSize: 2048",
        "stopOnErrors: true",
        "showButtonPreviousPage: true",
        "remoteService: service/write.php",
        "",
        "pages:",
        "  - type: generic",
        "    id: first_page",
        "    name: 환영합니다",
        "    content: |",
        "      <h1>Mood-EQ 청취 평가</h1>",
        "      <p>각 씬에 대해 4가지 버전(원본/V3.1/V3.2/Anchor)을 들어보고",
        "         <b>영상의 분위기와 가장 잘 맞는다고 느낀 정도</b>를 0~100점으로 평가해주세요.</p>",
        "      <ul><li>원본은 EQ가 없는 버전입니다.</li>",
        "          <li>Anchor는 의도적으로 저음을 줄인 저품질 기준입니다.</li></ul>",
        "",
    ]

    for clip_set in scene_clips:
        scene_id = clip_set["scene_id"]
        scene_dir = clip_set["scene_dir_name"]
        mood = clip_set["mood"]
        density = clip_set["density"]
        lines.extend([
            f"  - type: mushra",
            f"    id: {scene_id}",
            f"    name: {scene_id} ({mood}, density={density})",
            f"    content: 분위기 적합도를 0~100점으로 평가하세요.",
            f"    showWaveform: true",
            f"    enableLooping: true",
            f"    reference: configs/resources/audio/{scene_dir}/original.wav",
            f"    createAnchor35: false",
            f"    createAnchor70: false",
            f"    stimuli:",
            f"      original_hidden: configs/resources/audio/{scene_dir}/original.wav",
            f"      v3_1: configs/resources/audio/{scene_dir}/v3_1.wav",
            f"      v3_2: configs/resources/audio/{scene_dir}/v3_2.wav",
            f"      anchor: configs/resources/audio/{scene_dir}/anchor.wav",
            "",
        ])

    lines.extend([
        "  - type: finish",
        "    name: 평가 완료",
        "    content: 감사합니다! 평가 결과가 저장되었습니다.",
        "    showResults: false",
        "    writeResults: true",
        "",
    ])

    Path(output_yaml).write_text("\n".join(lines), encoding="utf-8")
    print(f"  💾 webMUSHRA YAML 저장: {output_yaml}")


# ────────────────────────────────────────────────────────
# 검증
# ────────────────────────────────────────────────────────
def verify_clip_set(clip_set: dict) -> None:
    """단일 씬의 4개 클립이 모두 정상 생성됐는지."""
    from .utils import get_audio_info

    clip_dir = MUSHRA_CLIPS_DIR / clip_set["scene_dir_name"]
    clips = clip_set["clips"]

    durations = []
    for name, fname in clips.items():
        path = clip_dir / fname
        assert path.exists(), f"{path} 없음"
        info = get_audio_info(path)
        durations.append(info["duration"])
        assert info["sample_rate"] >= 32000

    # 4개 길이가 ±0.05초 이내
    for d in durations:
        assert abs(d - durations[0]) < 0.05, f"클립 길이 불일치: {durations}"

    print(
        f"  ✓ {clip_set['scene_id']}: 4클립 모두 {durations[0]:.1f}초 일치"
    )


# ────────────────────────────────────────────────────────
# 평가용 씬 선택 가이드라인
# ────────────────────────────────────────────────────────
"""
영상별 평가 씬 선택 권장:
- Top Gun: 4~6개 (예: Title, Cockpit, Dogfight, Dialogue, Emotional)
- La La Land: 4~6개 (예: Title, Dance, Piano, Romance, Highway)
- 클립 길이: 8~15초 (너무 짧으면 EQ 차이 안 들림, 너무 길면 평가 피로)
- 다양한 mood/density 조합 포함
"""

# 예시 평가 셋 (사용자가 가이드 v6 라벨링 결과로 직접 채워넣기)
EVALUATION_SET_TEMPLATE = {
    "trailer_topgun": [
        # (scene_idx, start, end, mood, prob, density)
        # (0, 0.0, 8.0, "Power", 0.75, 0.0),   # Title
        # (5, 25.3, 38.1, "Tension", 0.80, 0.2),  # Dogfight
        # ... 가이드의 MANUAL_MOOD_LABELS_v3_1에서 4~6개 선택
    ],
    "trailer_lalaland": [
        # (8, 35.2, 47.8, "Joyful Activation", 0.80, 0.3),  # Dance
        # ...
    ],
}


def generate_all_clips(evaluation_set: dict) -> list[dict]:
    """평가 셋에 정의된 모든 씬에 대해 클립 생성 + YAML 저장."""
    ensure_dirs()
    all_clip_sets = []

    for video_key, scenes_to_eval in evaluation_set.items():
        if not scenes_to_eval:
            print(f"  ⏭ {video_key}: 평가 씬 정의 없음, 스킵")
            continue

        short_name = video_key.replace("trailer_", "")
        if not trailer_path(short_name).exists():
            print(f"  ⏭ {video_key} 영상 없음, 스킵")
            continue

        print(f"\n=== {video_key} ===")
        for scene_idx, start, end, mood, prob, density in scenes_to_eval:
            print(f"  씬 {scene_idx}: {start:.1f}~{end:.1f}s ({mood}, density={density})")
            clip_set = generate_clips_for_scene(
                video_key, scene_idx, start, end, mood, prob, density
            )
            verify_clip_set(clip_set)
            all_clip_sets.append(clip_set)

    # YAML은 webMUSHRA configs/ 안에 직접 저장
    yaml_path = WEBMUSHRA_CONFIG_DIR / "mood_eq_test.yaml"
    generate_webmushra_yaml(all_clip_sets, yaml_path)

    # 메타 JSON (재생성/디버깅용)
    meta_path = MUSHRA_CLIPS_DIR / "clip_sets.json"
    meta_path.write_text(
        json.dumps(all_clip_sets, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  💾 메타 저장: {meta_path}")

    print(f"\n  ✓ 총 {len(all_clip_sets)}개 씬 × 4클립 = {len(all_clip_sets)*4}개 wav")
    print(f"\n다음 단계 — 클립과 YAML이 이미 webMUSHRA 디렉토리에 있습니다:")
    print(f"  1. webMUSHRA 서버 실행 (백엔드와 포트 분리):")
    print(f"     cd {WEBMUSHRA_DIR} && php -S localhost:8080")
    print(f"  2. 브라우저: http://localhost:8080/?config=mood_eq_test.yaml")
    print(f"  3. 평가 후 결과 CSV는 {WEBMUSHRA_DIR}/results/ 에 저장됩니다.")

    return all_clip_sets


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "template":
        # 템플릿 출력
        print("EVALUATION_SET_TEMPLATE을 채운 다음 다시 실행하세요.")
        print(json.dumps(EVALUATION_SET_TEMPLATE, indent=2, default=str))
    else:
        # 채워진 EVALUATION_SET을 import해서 사용하세요:
        # from your_eval_set import EVALUATION_SET
        # generate_all_clips(EVALUATION_SET)
        if not any(EVALUATION_SET_TEMPLATE.values()):
            print("EVALUATION_SET_TEMPLATE이 비어있습니다.")
            print("이 파일을 직접 수정하거나 generate_all_clips(your_set) 형태로 호출하세요.")
            sys.exit(1)
        generate_all_clips(EVALUATION_SET_TEMPLATE)
