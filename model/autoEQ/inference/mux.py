"""mux.py — ffmpeg 영상-오디오 합성.

작업 11 (Day 4~5): 원본 영상 비디오 트랙 + 새 오디오 트랙 → 새 mp4.
-c:v copy로 비디오 재인코딩 스킵 (~144초 영상 1초 이내).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def mux_video_audio(video_path, audio_path, output_path) -> None:
    """원본 영상 비디오 트랙 + 새 오디오 트랙 → 새 mp4."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("사용법: python -m model.autoEQ.inference.mux <video> <audio> <output>")
        sys.exit(1)
    mux_video_audio(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"✓ {sys.argv[3]} 생성")
