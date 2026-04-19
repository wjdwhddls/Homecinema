"""Unit tests for precompute.py scaffold."""

import os

import pytest
import torch

from ..config import TrainConfig
from ..precompute import (
    encode_window_batch,
    load_clip_audio,
    load_clip_frames,
    precompute_dataset,
    resample_for_panns,
    save_features,
    split_into_windows,
)


def test_split_into_windows_8s():
    assert split_into_windows(8.0) == [(0.0, 4.0), (2.0, 6.0), (4.0, 8.0)]


def test_split_into_windows_10s():
    assert split_into_windows(10.0) == [
        (0.0, 4.0),
        (2.0, 6.0),
        (4.0, 8.0),
        (6.0, 10.0),
    ]


def test_split_into_windows_12s():
    assert split_into_windows(12.0) == [
        (0.0, 4.0),
        (2.0, 6.0),
        (4.0, 8.0),
        (6.0, 10.0),
        (8.0, 12.0),
    ]


def test_split_into_windows_short_clip_returns_empty():
    assert split_into_windows(3.0) == []


def test_resample_doubles_samples_for_16_to_32khz():
    wav = torch.randn(2, 64000)
    out = resample_for_panns(wav, src_sr=16000, target_sr=32000)
    assert out.shape == (2, 128000)


def test_resample_noop_when_rates_match():
    wav = torch.randn(3, 32000)
    out = resample_for_panns(wav, src_sr=32000, target_sr=32000)
    assert out is wav or torch.equal(out, wav)


def test_resample_preserves_rank_for_3d_input():
    wav = torch.randn(2, 1, 64000)
    out = resample_for_panns(wav, src_sr=16000, target_sr=32000)
    assert out.shape == (2, 1, 128000)


def test_save_features_round_trip(tmp_path):
    visual = {
        "clip_a_w0": torch.randn(512),
        "clip_a_w1": torch.randn(512),
    }
    audio = {
        "clip_a_w0": torch.randn(2048),
        "clip_a_w1": torch.randn(2048),
    }
    metadata = {
        "clip_a_w0": {"clip_id": "a", "movie_id": 0, "valence": 0.1, "arousal": -0.2, "start": 0.0, "end": 4.0},
        "clip_a_w1": {"clip_id": "a", "movie_id": 0, "valence": 0.1, "arousal": -0.2, "start": 2.0, "end": 6.0},
    }
    save_features(visual, audio, metadata, tmp_path, split_name="toy")

    loaded_v = torch.load(tmp_path / "toy_visual.pt", weights_only=False)
    loaded_a = torch.load(tmp_path / "toy_audio.pt", weights_only=False)
    loaded_m = torch.load(tmp_path / "toy_metadata.pt", weights_only=False)

    assert set(loaded_v) == set(visual)
    for k in visual:
        assert torch.equal(loaded_v[k], visual[k])
        assert torch.equal(loaded_a[k], audio[k])
        assert loaded_m[k] == metadata[k]


def test_stubs_raise_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        load_clip_frames(tmp_path / "dummy.mp4", [0.0, 0.5])
    with pytest.raises(NotImplementedError):
        load_clip_audio(tmp_path / "dummy.wav", 0.0, 4.0)
    with pytest.raises(NotImplementedError):
        precompute_dataset(
            manifest_path=tmp_path / "m.csv",
            output_dir=tmp_path,
            xclip=None,
            panns=None,
            split_name="x",
        )


@pytest.mark.skipif(
    not os.environ.get("RUN_ENCODER_TESTS"),
    reason="set RUN_ENCODER_TESTS=1 to run (downloads pretrained weights)",
)
def test_encode_window_batch_shapes():
    from ..encoders import PANNsEncoder, XCLIPEncoder

    cfg = TrainConfig()
    xclip = XCLIPEncoder(cfg)
    panns = PANNsEncoder(cfg)

    B = 2
    frames = torch.randn(B, cfg.num_frames, 3, cfg.frame_size, cfg.frame_size)
    waveform = torch.randn(B, cfg.audio_samples) * 0.1

    visual, audio = encode_window_batch(frames, waveform, xclip, panns, src_sr=cfg.audio_sr)

    assert visual.shape == (B, cfg.visual_dim)
    assert audio.shape == (B, cfg.audio_raw_dim)
    assert torch.isfinite(visual).all()
    assert torch.isfinite(audio).all()
