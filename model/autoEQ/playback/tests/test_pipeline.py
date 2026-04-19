"""Playback pipeline pure-logic tests (apply_timeline_to_audio) without ffmpeg."""

import numpy as np
import pytest

from model.autoEQ.playback.pipeline import (
    _slice_scene_audio,
    apply_timeline_to_audio,
)


def _have_pedalboard():
    try:
        import pedalboard  # noqa: F401
        return True
    except ImportError:
        return False


def _mk_flat_bands(gain_db: float = 0.0) -> list[dict]:
    return [
        {"freq_hz": 31.5, "gain_db": gain_db, "q": 0.7},
        {"freq_hz": 63.0, "gain_db": gain_db, "q": 0.7},
        {"freq_hz": 125.0, "gain_db": gain_db, "q": 1.0},
        {"freq_hz": 250.0, "gain_db": gain_db, "q": 1.2},
        {"freq_hz": 500.0, "gain_db": gain_db, "q": 1.4},
        {"freq_hz": 1000.0, "gain_db": gain_db, "q": 1.4},
        {"freq_hz": 2000.0, "gain_db": gain_db, "q": 1.4},
        {"freq_hz": 4000.0, "gain_db": gain_db, "q": 1.2},
        {"freq_hz": 8000.0, "gain_db": gain_db, "q": 0.7},
        {"freq_hz": 16000.0, "gain_db": gain_db, "q": 0.7},
    ]


def test_slice_scene_audio_basic():
    audio = np.arange(1000, dtype=np.float32)
    sliced = _slice_scene_audio(audio, sample_rate=100, start_sec=2.0, end_sec=5.0)
    assert sliced.shape[0] == 300
    assert sliced[0] == 200.0


def test_slice_scene_audio_clamps_end():
    audio = np.arange(100, dtype=np.float32)
    sliced = _slice_scene_audio(audio, sample_rate=10, start_sec=8.0, end_sec=20.0)
    # Only 2s (20 samples) left in audio (8s to 10s)
    assert sliced.shape[0] == 20


def test_slice_scene_audio_negative_start_clamps_to_zero():
    audio = np.arange(100, dtype=np.float32)
    sliced = _slice_scene_audio(audio, sample_rate=10, start_sec=-1.0, end_sec=2.0)
    assert sliced[0] == 0.0 and sliced.shape[0] == 20


def test_empty_timeline_returns_audio_unchanged():
    audio = np.ones(1000, dtype=np.float32)
    out = apply_timeline_to_audio(audio, 1000, {"scenes": []})
    assert np.array_equal(out, audio)


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_apply_timeline_preserves_total_length():
    # A/V sync requires output length == input length
    sr = 48000
    duration = 2
    audio = (np.random.randn(sr * duration).astype(np.float32) * 0.1)
    timeline = {
        "scenes": [
            {"scene_idx": 0, "start_sec": 0.0, "end_sec": 1.0,
             "eq_preset": {"effective_bands": _mk_flat_bands(0.0)}},
            {"scene_idx": 1, "start_sec": 1.0, "end_sec": 2.0,
             "eq_preset": {"effective_bands": _mk_flat_bands(0.0)}},
        ],
    }
    out = apply_timeline_to_audio(audio, sr, timeline, crossfade_ms=100)
    # Total length UNCHANGED (in-place boundary blend preserves duration)
    assert out.shape[0] == audio.shape[0]
    assert np.all(np.isfinite(out))


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_apply_timeline_boost_scene_raises_rms():
    sr = 48000
    duration = 2
    sine_2k = (np.sin(2 * np.pi * 2000.0 * np.arange(sr * duration) / sr)
               .astype(np.float32) * 0.3)
    timeline = {
        "scenes": [
            {"scene_idx": 0, "start_sec": 0.0, "end_sec": 2.0,
             "eq_preset": {"effective_bands": _mk_flat_bands(3.0)}},  # +3 dB everywhere
        ],
    }
    out = apply_timeline_to_audio(sine_2k, sr, timeline)
    rms_in = np.sqrt(np.mean(sine_2k ** 2))
    rms_out = np.sqrt(np.mean(out ** 2))
    assert rms_out > rms_in, f"boost should raise RMS ({rms_in:.3f} → {rms_out:.3f})"
