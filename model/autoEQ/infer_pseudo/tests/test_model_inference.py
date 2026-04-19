"""model_inference module — pure-logic helpers only (no actual ckpt load)."""

import numpy as np
import pytest
import soundfile as sf
import torch

from model.autoEQ.infer_pseudo.model_inference import (
    AUDIO_SR,
    _batched,
    _load_audio_slice,
    _sample_frame_times,
)


def test_frame_times_uniform_and_inside_window():
    ts = _sample_frame_times(10.0, 14.0, num_frames=8)
    # 8 uniformly-spaced inside [10, 14], at (i+0.5)/8 offsets
    assert len(ts) == 8
    assert ts[0] > 10.0 and ts[-1] < 14.0
    # Spacing uniform
    diffs = [ts[i+1] - ts[i] for i in range(7)]
    assert max(diffs) - min(diffs) < 1e-6


def test_frame_times_span_scales_with_window():
    # 1-second window → frames packed into 1s
    ts = _sample_frame_times(0.0, 1.0, num_frames=4)
    assert ts[0] > 0.0 and ts[-1] < 1.0
    assert len(ts) == 4


def test_batched_yields_correct_groups():
    data = list(range(7))
    batches = list(_batched(data, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_batched_empty():
    assert list(_batched([], 3)) == []


def test_load_audio_slice_returns_exact_4s_length(tmp_path):
    # 10s of silence at 16 kHz mono
    wav = tmp_path / "test.wav"
    data = np.zeros(AUDIO_SR * 10, dtype=np.float32)
    sf.write(str(wav), data, AUDIO_SR)
    slice_t = _load_audio_slice(wav, start_sec=2.0, end_sec=6.0)
    assert slice_t.shape[0] == AUDIO_SR * 4


def test_load_audio_slice_pads_when_window_exceeds_audio(tmp_path):
    # 3s audio, request [1, 5] → only 2s real data + 2s zero pad
    wav = tmp_path / "short.wav"
    sig = np.ones(AUDIO_SR * 3, dtype=np.float32)
    sf.write(str(wav), sig, AUDIO_SR)
    slice_t = _load_audio_slice(wav, start_sec=1.0, end_sec=5.0)
    assert slice_t.shape[0] == AUDIO_SR * 4
    # First 2s should be ≈1 (small PCM quantization ok), last 2s zero pad
    assert abs(slice_t[0].item() - 1.0) < 1e-3
    assert slice_t[-1].item() == 0.0


def test_load_audio_slice_handles_negative_start(tmp_path):
    wav = tmp_path / "short.wav"
    sig = np.ones(AUDIO_SR * 3, dtype=np.float32)
    sf.write(str(wav), sig, AUDIO_SR)
    # Negative start is clamped to 0
    slice_t = _load_audio_slice(wav, start_sec=-1.0, end_sec=3.0)
    # expected exactly 4s (target length), pad rest with zero
    assert slice_t.shape[0] == AUDIO_SR * 4
