"""VAD module smoke test — generates synthetic audio with known speech pattern.

Skips if Silero VAD / ffmpeg not available.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from model.autoEQ.infer_pseudo.vad import SAMPLE_RATE, run_vad


def _have_silero():
    try:
        import silero_vad  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_silero(), reason="silero_vad not installed")
def test_vad_on_silent_audio_returns_no_segments(tmp_path):
    # 3 seconds of silence at 16 kHz
    silent = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    wav = tmp_path / "silent.wav"
    sf.write(str(wav), silent, SAMPLE_RATE)
    segments = run_vad(wav)
    assert segments == [], f"silent audio should produce no segments, got {segments}"


@pytest.mark.skipif(not _have_silero(), reason="silero_vad not installed")
def test_vad_rejects_wrong_sample_rate(tmp_path):
    # 8 kHz audio should raise
    data = np.zeros(8000, dtype=np.float32)
    wav = tmp_path / "lowrate.wav"
    sf.write(str(wav), data, 8000)
    with pytest.raises(ValueError):
        run_vad(wav)


def test_extract_audio_requires_ffmpeg_present():
    assert shutil.which("ffmpeg") is not None, \
        "ffmpeg binary must be on PATH for VAD pipeline"
