"""명세 테스트 7: 오디오 길이 정렬 검증 (4초 @ 16kHz = 64000 samples)."""

import torch

from ..dataset import align_audio, SyntheticAutoEQDataset
from ..config import TrainConfig


def test_synthetic_audio_exact_length():
    """합성 데이터셋의 모든 오디오가 정확히 4초인지 확인."""
    config = TrainConfig()
    ds = SyntheticAutoEQDataset(num_clips=20, num_films=5, config=config)
    expected = config.audio_samples  # 64000

    for i in range(len(ds)):
        waveform = ds[i]["audio_waveform"]
        assert waveform.shape == (1, expected), (
            f"Sample {i}: expected (1, {expected}), got {waveform.shape}"
        )


def test_align_truncation():
    """5초 오디오 -> 4초로 truncation."""
    long_audio = torch.randn(1, 80000)  # 5 sec @ 16kHz
    aligned = align_audio(long_audio, target_samples=64000)
    assert aligned.shape == (1, 64000)
    # Content preserved: first 64000 samples should match
    assert torch.allclose(aligned, long_audio[:, :64000])


def test_align_padding():
    """3초 오디오 -> 4초로 zero-padding."""
    short_audio = torch.randn(1, 48000)  # 3 sec @ 16kHz
    aligned = align_audio(short_audio, target_samples=64000)
    assert aligned.shape == (1, 64000)
    # Original content preserved
    assert torch.allclose(aligned[:, :48000], short_audio)
    # Padding is zeros
    assert aligned[:, 48000:].abs().sum() == 0


def test_align_exact():
    """정확히 4초 오디오 -> 변경 없음."""
    exact_audio = torch.randn(1, 64000)
    aligned = align_audio(exact_audio, target_samples=64000)
    assert aligned.shape == (1, 64000)
    assert torch.allclose(aligned, exact_audio)


def test_align_1d_input():
    """1D 텐서 입력도 처리 가능."""
    audio_1d = torch.randn(48000)
    aligned = align_audio(audio_1d, target_samples=64000)
    assert aligned.shape == (1, 64000)
