"""명세 테스트 2: Negative sampling 비율 검증.

50% congruent / 25% slight / 25% strong 비율이 유지되는지 확인.
"""

import torch

from ..config import TrainConfig
from ..negative_sampler import NegativeSampler


def test_sampling_ratios():
    """대규모 샘플에서 50/25/25 비율 검증 (±5% 허용)."""
    config = TrainConfig()
    sampler = NegativeSampler(config, seed=42)

    total_counts = {0: 0, 1: 0, 2: 0}
    num_iterations = 100
    batch_size = 32

    for i in range(num_iterations):
        audio = torch.randn(batch_size, config.audio_raw_dim)
        va = torch.randn(batch_size, 2)
        # Use different movie_ids to ensure cross-film swaps are possible
        movie_ids = torch.arange(batch_size) % 10

        _, cong_labels = sampler.sample(audio, va, movie_ids)

        for label in [0, 1, 2]:
            total_counts[label] += (cong_labels == label).sum().item()

    total = sum(total_counts.values())
    ratio_cong = total_counts[0] / total
    ratio_slight = total_counts[1] / total
    ratio_strong = total_counts[2] / total

    assert 0.45 <= ratio_cong <= 0.55, f"Congruent ratio {ratio_cong:.3f} out of range"
    assert 0.20 <= ratio_slight <= 0.30, f"Slight ratio {ratio_slight:.3f} out of range"
    assert 0.20 <= ratio_strong <= 0.30, f"Strong ratio {ratio_strong:.3f} out of range"


def test_audio_actually_swapped():
    """Incongruent 샘플의 오디오가 실제로 교체되었는지 확인."""
    config = TrainConfig()
    sampler = NegativeSampler(config, seed=0)

    batch_size = 32
    audio = torch.randn(batch_size, config.audio_raw_dim)
    va = torch.randn(batch_size, 2)
    movie_ids = torch.arange(batch_size) % 10

    modified_audio, cong_labels = sampler.sample(audio, va, movie_ids)

    # Congruent samples should keep original audio
    congruent_mask = cong_labels == 0
    if congruent_mask.any():
        assert torch.allclose(
            modified_audio[congruent_mask], audio[congruent_mask]
        )

    # At least some incongruent samples should have different audio
    incongruent_mask = cong_labels > 0
    if incongruent_mask.any():
        diff = (modified_audio[incongruent_mask] - audio[incongruent_mask]).abs().sum()
        assert diff > 0, "No audio was actually swapped for incongruent samples"


def test_cross_film_swap():
    """같은 영화 내에서 스왑하지 않는지 확인."""
    config = TrainConfig()
    sampler = NegativeSampler(config, seed=42)

    # All from same film -> no candidates for swap -> should remain congruent
    batch_size = 8
    audio = torch.randn(batch_size, config.audio_raw_dim)
    va = torch.randn(batch_size, 2)
    movie_ids = torch.zeros(batch_size, dtype=torch.long)  # all same film

    modified_audio, cong_labels = sampler.sample(audio, va, movie_ids)

    # Slight/strong samples with no cross-film candidates keep original audio
    incongruent_mask = cong_labels > 0
    if incongruent_mask.any():
        assert torch.allclose(
            modified_audio[incongruent_mask], audio[incongruent_mask]
        ), "Swapped audio within same film"
