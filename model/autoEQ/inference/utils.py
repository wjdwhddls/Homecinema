"""utils.py — ffprobe 기반 미디어 메타데이터 헬퍼.

영상마다 다른 길이/샘플레이트를 자동으로 감지합니다 (하드코딩 X).
모든 검증 함수가 이 모듈을 사용합니다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def get_media_info(media_path: PathLike) -> dict:
    """ffprobe로 영상/오디오 메타데이터 조회. 어떤 파일이든 동작."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def get_duration(media_path: PathLike) -> float:
    """영상 또는 오디오 파일의 길이(초)를 반환."""
    info = get_media_info(media_path)
    return float(info["format"]["duration"])


def get_audio_info(audio_path: PathLike) -> dict:
    """오디오 파일의 샘플레이트, 채널 수, 길이 반환."""
    info = get_media_info(audio_path)
    audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if not audio_streams:
        raise ValueError(f"오디오 트랙이 없는 파일: {audio_path}")
    stream = audio_streams[0]
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": stream["channels"],
        "duration": float(info["format"]["duration"]),
    }


def get_video_info(video_path: PathLike) -> dict:
    """영상 파일의 해상도, fps, 길이, 오디오 트랙 유무 반환.

    r_frame_rate 파싱 시 예외 안전 처리.
    """
    info = get_media_info(video_path)
    v_streams = [s for s in info["streams"] if s["codec_type"] == "video"]
    if not v_streams:
        raise ValueError(f"비디오 트랙이 없는 파일: {video_path}")
    v = v_streams[0]
    a_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]

    # fps 안전 파싱
    try:
        num, den = v["r_frame_rate"].split("/")
        fps = int(num) / int(den) if int(den) > 0 else 0.0
    except (ValueError, ZeroDivisionError, KeyError):
        fps = 0.0

    return {
        "width": v["width"],
        "height": v["height"],
        "fps": round(fps, 3),
        "duration": float(info["format"]["duration"]),
        "has_audio": len(a_streams) > 0,
    }


# ────────────────────────────────────────────────────────
# 직접 실행 시 빠른 점검
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m model.autoEQ.inference.utils <video_or_audio_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\n=== {path} ===")
    try:
        info = get_video_info(path)
        print(
            f"비디오: {info['width']}x{info['height']} @ {info['fps']}fps, "
            f"{info['duration']:.1f}초, audio={info['has_audio']}"
        )
        if info["has_audio"]:
            ainfo = get_audio_info(path)
            print(f"오디오: {ainfo['sample_rate']}Hz, {ainfo['channels']}ch")
    except ValueError:
        ainfo = get_audio_info(path)
        print(
            f"오디오 전용: {ainfo['sample_rate']}Hz, {ainfo['channels']}ch, "
            f"{ainfo['duration']:.1f}초"
        )
