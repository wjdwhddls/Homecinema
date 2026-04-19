"""remux.py — validate ffmpeg binary presence + audio info probe."""

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from model.autoEQ.playback.remux import (
    extract_audio,
    get_audio_info,
    remux_video_with_audio,
)


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")
def test_get_audio_info_on_real_film(tmp_path):
    # Use caminandes_3.mp4 which we confirmed exists for E2E test
    from pathlib import Path
    film = Path("dataset/autoEQ/CCMovies/films/caminandes_3.mp4")
    if not film.is_file():
        pytest.skip("caminandes_3.mp4 not present")
    info = get_audio_info(film)
    assert info["sample_rate"] > 0
    assert info["channels"] >= 1
    assert info["duration_sec"] > 0


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_extract_audio_roundtrip(tmp_path):
    # Create tiny test video with synthetic audio via ffmpeg, then extract
    src_video = tmp_path / "src.mp4"
    # Generate 2s of 440 Hz tone + color video
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=64x64:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "2",
        str(src_video),
    ], check=True)
    out_wav = tmp_path / "out.wav"
    extract_audio(src_video, out_wav)
    assert out_wav.is_file()
    data, sr = sf.read(str(out_wav))
    assert sr == 48000
    # Extracted audio ≈ 2 seconds
    assert abs(data.shape[0] / sr - 2.0) < 0.1


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_extract_audio_can_downmix_and_resample(tmp_path):
    src_video = tmp_path / "src.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ac", "2",
        str(src_video),
    ], check=True)
    out_wav = tmp_path / "out_16k_mono.wav"
    extract_audio(src_video, out_wav, target_sr=16000, channels=1)
    data, sr = sf.read(str(out_wav))
    assert sr == 16000
    assert data.ndim == 1  # mono


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")
def test_remux_produces_valid_output(tmp_path):
    # Synthetic 1s video + synthetic 1s wav, remux, verify result has both streams
    src_video = tmp_path / "src.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=green:s=64x64:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(src_video),
    ], check=True)
    # Make a processed-audio wav (same length, different content)
    wav_path = tmp_path / "proc.wav"
    sf.write(str(wav_path),
             np.random.randn(48000).astype(np.float32) * 0.1,
             48000)
    out = tmp_path / "remuxed.mp4"
    remux_video_with_audio(src_video, wav_path, out)
    assert out.is_file()
    # Probe: both video and audio streams present
    import json
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(out)],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)
    codecs = {s["codec_type"] for s in info["streams"]}
    assert "video" in codecs and "audio" in codecs
