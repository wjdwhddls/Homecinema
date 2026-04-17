"""run_mushra_demo.py — webMUSHRA 데모용 클립 + YAML 생성 드라이버.

전략:
- mushra_generator.py 무수정. generate_webmushra_yaml과 make_anchor_eq만 재사용.
- 원본/V3.1/V3.2 오디오는 analyzer가 이미 생성한 processed_v3_*.mp4에서 추출
  (mushra_generator의 generate_clips_for_scene는 단일 mood로 균일 EQ 적용하므로
  시변 EQ 결과 시연이 목적인 이번 데모에서는 적합하지 않음.)
- Anchor는 원본 오디오에 저역 -6dB cut.
- 트레일러 전체를 단일 "씬"으로 다루어 trial 1개 = 영상 1개.

실행: PYTHONPATH=. venv/Scripts/python.exe tools/run_mushra_demo.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from model.autoEQ.inference.mushra_generator import (
    generate_webmushra_yaml,
    make_anchor_eq,
    verify_clip_set,
)
from model.autoEQ.inference.paths import (
    MUSHRA_CLIPS_DIR,
    WEBMUSHRA_CONFIG_DIR,
    ensure_dirs,
)
from model.autoEQ.inference.playback import apply_eq_to_file


EVALUATION_SET = {
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


def extract_audio(video_path: str | Path, wav_path: Path, sample_rate: int = 48000) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", str(sample_rate),
            "-ac", "2",
            "-acodec", "pcm_s16le",
            str(wav_path),
        ],
        check=True, capture_output=True,
    )


def build_clip_set(video_key: str, videos: dict[str, str]) -> dict:
    short_name = video_key
    scene_dir_name = f"trailer_{short_name}_full"
    out_dir = MUSHRA_CLIPS_DIR / scene_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    original = out_dir / "original.wav"
    v3_1     = out_dir / "v3_1.wav"
    v3_2     = out_dir / "v3_2.wav"
    anchor   = out_dir / "anchor.wav"

    print(f"  [{video_key}] original 추출 ← {videos['trailer']}")
    extract_audio(videos["trailer"], original)

    print(f"  [{video_key}] v3_1 추출 ← {videos['v3_1']}")
    extract_audio(videos["v3_1"], v3_1)

    print(f"  [{video_key}] v3_2 추출 ← {videos['v3_2']}")
    extract_audio(videos["v3_2"], v3_2)

    print(f"  [{video_key}] anchor 생성 (저역 -6dB cut)")
    apply_eq_to_file(original, anchor, make_anchor_eq())

    from model.autoEQ.inference.utils import get_audio_info
    paths_to_align = [original, v3_1, v3_2, anchor]
    durations = [float(get_audio_info(p)["duration"]) for p in paths_to_align]
    target = min(durations)
    if max(durations) - target > 0.01:
        print(f"  [{video_key}] 길이 정렬: target={target:.3f}s (원래 {durations})")
        for p in paths_to_align:
            tmp = p.with_suffix(".trim.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-t", f"{target:.3f}",
                 "-acodec", "pcm_s16le", str(tmp)],
                check=True, capture_output=True,
            )
            tmp.replace(p)

    clips = {
        "original": original.name,
        "v3_1":     v3_1.name,
        "v3_2":     v3_2.name,
        "anchor":   anchor.name,
    }

    info = get_audio_info(original)
    duration = float(info["duration"])

    return {
        "scene_id": scene_dir_name,
        "scene_dir_name": scene_dir_name,
        "duration_sec": round(duration, 2),
        "mood": "full_trailer",
        "density": 0.0,
        "clips": clips,
    }


def main() -> None:
    ensure_dirs()
    clip_sets = []
    for key, videos in EVALUATION_SET.items():
        print(f"\n=== {key} ===")
        for v in videos.values():
            if not Path(v).exists():
                raise FileNotFoundError(f"입력 파일 없음: {v}")
        cs = build_clip_set(key, videos)
        try:
            verify_clip_set(cs)
        except AssertionError as e:
            print(f"  ⚠ verify 경고: {e}")
        clip_sets.append(cs)

    yaml_path = WEBMUSHRA_CONFIG_DIR / "mood_eq_demo.yaml"
    generate_webmushra_yaml(clip_sets, yaml_path, testname="Mood-EQ Demo (Full Trailer)")

    print("\n=== 완료 ===")
    print(f"YAML: {yaml_path}")
    for cs in clip_sets:
        print(f"  {cs['scene_dir_name']} — {cs['duration_sec']}s × 4 클립")


if __name__ == "__main__":
    main()
