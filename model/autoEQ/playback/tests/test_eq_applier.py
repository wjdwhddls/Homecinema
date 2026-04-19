"""EQ applier — verifies pedalboard chain produces expected spectral shape."""

import numpy as np
import pytest

from model.autoEQ.playback.eq_applier import (
    apply_scene_eq_to_audio,
    build_eq_chain_from_bands,
)


def _have_pedalboard():
    try:
        import pedalboard  # noqa: F401
        return True
    except ImportError:
        return False


TENSION_BANDS = [
    {"freq_hz": 31.5, "gain_db": 2.0, "q": 0.7},
    {"freq_hz": 63.0, "gain_db": 2.0, "q": 0.7},
    {"freq_hz": 125.0, "gain_db": 1.0, "q": 1.0},
    {"freq_hz": 250.0, "gain_db": 0.0, "q": 1.2},
    {"freq_hz": 500.0, "gain_db": 0.0, "q": 1.4},
    {"freq_hz": 1000.0, "gain_db": 1.0, "q": 1.4},
    {"freq_hz": 2000.0, "gain_db": 2.5, "q": 1.4},
    {"freq_hz": 4000.0, "gain_db": 2.0, "q": 1.2},
    {"freq_hz": 8000.0, "gain_db": 0.0, "q": 0.7},
    {"freq_hz": 16000.0, "gain_db": -1.0, "q": 0.7},
]


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_build_chain_produces_pedalboard():
    from pedalboard import Pedalboard
    chain = build_eq_chain_from_bands(TENSION_BANDS)
    assert isinstance(chain, Pedalboard)
    assert len(chain) == 10


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_apply_eq_preserves_length_mono():
    sr = 48000
    audio = np.random.randn(sr * 2).astype(np.float32) * 0.1
    out = apply_scene_eq_to_audio(audio, sr, TENSION_BANDS)
    assert out.shape == audio.shape
    assert out.dtype == np.float32


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_apply_eq_preserves_length_stereo():
    sr = 48000
    audio = np.random.randn(sr * 2, 2).astype(np.float32) * 0.1
    out = apply_scene_eq_to_audio(audio, sr, TENSION_BANDS)
    assert out.shape == audio.shape


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_empty_input_returns_empty():
    out = apply_scene_eq_to_audio(np.array([], dtype=np.float32), 48000, TENSION_BANDS)
    assert out.size == 0


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_all_zero_gain_bypass_is_near_identity():
    flat_bands = [
        {"freq_hz": b["freq_hz"], "gain_db": 0.0, "q": b["q"]}
        for b in TENSION_BANDS
    ]
    sr = 48000
    audio = np.random.randn(sr).astype(np.float32) * 0.1
    out = apply_scene_eq_to_audio(audio, sr, flat_bands)
    # With all gains = 0, pedalboard biquad peaking = identity
    assert np.allclose(out, audio, atol=1e-4), \
        "zero-gain EQ should be near-identity"


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_boost_increases_rms_at_target_freq():
    # Sine wave at 2 kHz, apply Tension EQ (B7 = +2.5 dB)
    sr = 48000
    t = np.arange(sr * 2) / sr
    sine_2k = np.sin(2 * np.pi * 2000.0 * t).astype(np.float32) * 0.5
    out = apply_scene_eq_to_audio(sine_2k, sr, TENSION_BANDS)
    rms_in = np.sqrt(np.mean(sine_2k ** 2))
    rms_out = np.sqrt(np.mean(out ** 2))
    # +2.5 dB = ×1.334; expect some boost but biquad affects more than B7
    # (surrounding bands also active). Just verify output > input.
    assert rms_out > rms_in, f"2kHz boost should raise RMS; got {rms_in:.3f}→{rms_out:.3f}"


@pytest.mark.skipif(not _have_pedalboard(), reason="pedalboard not installed")
def test_cut_decreases_rms_at_target_freq():
    # Sadness preset: B7 = -2 dB, B8 = -2 dB (strong cut in voice range)
    sadness_bands = [
        {"freq_hz": 31.5, "gain_db": 0.0, "q": 0.7},
        {"freq_hz": 63.0, "gain_db": 1.0, "q": 0.7},
        {"freq_hz": 125.0, "gain_db": 1.0, "q": 1.0},
        {"freq_hz": 250.0, "gain_db": 1.0, "q": 1.2},
        {"freq_hz": 500.0, "gain_db": 0.0, "q": 1.4},
        {"freq_hz": 1000.0, "gain_db": 0.0, "q": 1.4},
        {"freq_hz": 2000.0, "gain_db": -2.0, "q": 1.4},
        {"freq_hz": 4000.0, "gain_db": -2.0, "q": 1.2},
        {"freq_hz": 8000.0, "gain_db": -1.5, "q": 0.7},
        {"freq_hz": 16000.0, "gain_db": -1.5, "q": 0.7},
    ]
    sr = 48000
    t = np.arange(sr * 2) / sr
    sine_3k = np.sin(2 * np.pi * 3000.0 * t).astype(np.float32) * 0.5
    out = apply_scene_eq_to_audio(sine_3k, sr, sadness_bands)
    rms_in = np.sqrt(np.mean(sine_3k ** 2))
    rms_out = np.sqrt(np.mean(out ** 2))
    assert rms_out < rms_in, f"3kHz cut should lower RMS; got {rms_in:.3f}→{rms_out:.3f}"
