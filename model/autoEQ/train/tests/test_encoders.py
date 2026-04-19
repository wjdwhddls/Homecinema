"""Dummy-input forward tests for XCLIPEncoder / PANNsEncoder.

Gated by RUN_ENCODER_TESTS=1 because loading the pretrained weights
requires network access and hundreds of MB of downloads. Keep it out
of the default test loop; run explicitly when touching encoders.py.
"""

import os

import pytest
import torch

from ..config import TrainConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_ENCODER_TESTS"),
    reason="set RUN_ENCODER_TESTS=1 to run (downloads pretrained weights)",
)


@pytest.fixture(scope="module")
def encoder_config() -> TrainConfig:
    return TrainConfig()


@pytest.fixture(scope="module")
def xclip(encoder_config: TrainConfig):
    from ..encoders import XCLIPEncoder
    return XCLIPEncoder(encoder_config)


@pytest.fixture(scope="module")
def panns(encoder_config: TrainConfig):
    from ..encoders import PANNsEncoder
    return PANNsEncoder(encoder_config)


def test_xclip_forward_shape(xclip, encoder_config):
    B = 2
    c = encoder_config
    pixel_values = torch.randn(B, c.num_frames, 3, c.frame_size, c.frame_size)
    out = xclip(pixel_values)
    assert out.shape == (B, c.visual_dim)
    assert torch.isfinite(out).all()


def test_panns_forward_shape(panns, encoder_config):
    B = 2
    waveform = torch.randn(B, encoder_config.audio_samples) * 0.1
    out = panns(waveform)
    assert out.shape == (B, encoder_config.audio_raw_dim)
    assert torch.isfinite(out).all()


def test_xclip_frozen(xclip):
    assert all(not p.requires_grad for p in xclip.parameters())
    assert not xclip.training


def test_panns_frozen(panns):
    assert all(not p.requires_grad for p in panns.tagger.model.parameters())
    assert not panns.tagger.model.training


def test_encoders_match_config_dims(xclip, panns, encoder_config):
    B = 1
    c = encoder_config
    v = xclip(torch.randn(B, c.num_frames, 3, c.frame_size, c.frame_size))
    a = panns(torch.randn(B, c.audio_samples) * 0.1)
    assert v.size(-1) == c.visual_dim
    assert a.size(-1) == c.audio_raw_dim
