"""ffmpeg wrappers — audio extraction + video/audio remux.

Spec V3.3 §5-9 (= V3.2 §5-14):
    - Extract original audio: preserve channels + sample rate, output as WAV PCM
      for lossless intermediate.
    - Remux: video stream copied (no re-encoding, 0 quality loss), audio encoded
      to AAC 192 kbps (film-standard quality).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _ffmpeg_probe(video_path: Path) -> dict:
    """Run ffprobe and return JSON metadata dict."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(video_path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def get_audio_info(video_path: Path) -> dict:
    """Return {'sample_rate': int, 'channels': int, 'duration_sec': float} for
    the first audio stream.
    """
    probe = _ffmpeg_probe(Path(video_path))
    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise RuntimeError(f"No audio stream found in {video_path}")
    a = audio_streams[0]
    return {
        "sample_rate": int(a["sample_rate"]),
        "channels": int(a["channels"]),
        "duration_sec": float(probe["format"]["duration"]),
        "codec": a.get("codec_name", ""),
    }


def extract_audio(
    video_path: str | Path,
    output_wav: str | Path,
    *,
    target_sr: int | None = None,
    channels: int | None = None,
) -> Path:
    """Extract audio track to WAV PCM (float32 or s16 depending on target).

    Args:
        video_path: source video.
        output_wav: destination WAV path.
        target_sr: resample target (None = preserve original).
        channels: downmix to this channel count (None = preserve original).
    """
    video_path = Path(video_path)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
    ]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    if target_sr is not None:
        cmd += ["-ar", str(target_sr)]
    cmd += [
        "-c:a", "pcm_f32le",  # float32 PCM → round-trip via soundfile without quantization
        "-vn",                # no video
        str(output_wav),
    ]
    subprocess.run(cmd, check=True)
    return output_wav


def remux_video_with_audio(
    video_path: str | Path,
    processed_audio_wav: str | Path,
    output_video: str | Path,
    *,
    audio_bitrate: str = "192k",
) -> Path:
    """Combine original video track (copied) with processed audio (AAC encoded).

    Spec §5-14 reference command:
        ffmpeg -i original.mp4 -i processed_audio.wav
               -c:v copy -c:a aac -b:a 192k
               -map 0:v:0 -map 1:a:0
               output.mp4
    """
    video_path = Path(video_path)
    processed_audio_wav = Path(processed_audio_wav)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(processed_audio_wav),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-map", "0:v:0",
        "-map", "1:a:0",
        # Drop original audio entirely; avoid any cross-stream sync drift
        "-shortest",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)
    return output_video
