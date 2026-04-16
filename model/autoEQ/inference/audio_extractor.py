"""audio_extractor.py — ffmpeg 기반 오디오 추출.

작업 1 (Day 2): 영상에서 분석용(32kHz mono) + 출력용(원본 품질) wav를 추출.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import audio_path, trailer_path
from .utils import get_audio_info, get_duration


def extract_audio(video_path, name: str) -> tuple[Path, Path]:
    """영상에서 두 종류의 wav를 추출.

    Args:
        video_path: 영상 파일 경로
        name: 키 (예: 'topgun', 'lalaland')

    Returns:
        (audio_32k, audio_original) 경로 튜플
    """
    out_32k = audio_path(name, "32k")
    out_orig = audio_path(name, "original")
    out_32k.parent.mkdir(parents=True, exist_ok=True)

    # 분석용: 32kHz mono (PANNs/Silero VAD)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "32000", "-ac", "1",
            str(out_32k),
        ],
        check=True, capture_output=True,
    )
    # 재생용: 원본 품질 유지 (EQ 적용 후 출력)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", str(out_orig)],
        check=True, capture_output=True,
    )
    return out_32k, out_orig


def verify_audio_extraction(video_path, name: str) -> None:
    """단일 영상의 오디오 추출 검증."""
    out_32k = audio_path(name, "32k")
    out_orig = audio_path(name, "original")

    assert out_32k.exists(), f"{out_32k} 없음"
    assert out_orig.exists(), f"{out_orig} 없음"

    info_32k = get_audio_info(out_32k)
    assert info_32k["sample_rate"] == 32000, \
        f"샘플레이트 {info_32k['sample_rate']} ≠ 32000"
    assert info_32k["channels"] == 1, f"채널 {info_32k['channels']} ≠ 1"

    info_orig = get_audio_info(out_orig)

    video_duration = get_duration(video_path)
    audio_duration = info_32k["duration"]
    assert abs(video_duration - audio_duration) < 0.5, \
        f"길이 불일치: 영상 {video_duration:.1f}s vs 오디오 {audio_duration:.1f}s"

    print(f"  ✓ {out_32k.name}: {info_32k['sample_rate']}Hz mono, {audio_duration:.1f}초")
    print(f"  ✓ {out_orig.name}: {info_orig['sample_rate']}Hz, {info_orig['channels']}ch")
    print(f"  ✓ 영상 길이와 오디오 길이 일치 ({video_duration:.1f}초)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 기본: 두 트레일러 모두 처리
        for name in ["topgun", "lalaland"]:
            video = trailer_path(name)
            if not video.exists():
                print(f"⏭ {video} 없음, 스킵")
                continue
            print(f"\n=== {name} ===")
            extract_audio(video, name)
            verify_audio_extraction(video, name)
    elif len(sys.argv) == 3:
        video, name = sys.argv[1], sys.argv[2]
        extract_audio(video, name)
        verify_audio_extraction(video, name)
    else:
        print("사용법:")
        print("  python -m model.autoEQ.inference.audio_extractor")
        print("  python -m model.autoEQ.inference.audio_extractor <video> <name>")
        sys.exit(1)
