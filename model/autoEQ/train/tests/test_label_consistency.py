"""명세 테스트 4: DataLoader 라벨 일관성 검증.

- 범위: valence/arousal in [-1,1], mood in [0,6], cong_label in [0,2]
- shape/dtype 정합성
- 재현성
"""

import torch


def test_label_ranges(synthetic_dataset):
    """모든 샘플의 라벨이 정의된 범위 내에 있는지 확인."""
    for i in range(len(synthetic_dataset)):
        sample = synthetic_dataset[i]
        v = sample["valence"].item()
        a = sample["arousal"].item()
        m = sample["mood"].item()
        c = sample["cong_label"].item()

        assert -1.0 <= v <= 1.0, f"Valence {v} out of range at idx {i}"
        assert -1.0 <= a <= 1.0, f"Arousal {a} out of range at idx {i}"
        assert 0 <= m <= 6, f"Mood {m} out of range at idx {i}"
        assert 0 <= c <= 2, f"Cong label {c} out of range at idx {i}"


def test_batch_shapes(synthetic_batch, config):
    """배치의 shape과 dtype이 올바른지 확인."""
    B = config.batch_size

    assert synthetic_batch["visual_feat"].shape == (B, config.visual_dim)
    assert synthetic_batch["audio_feat"].shape == (B, config.audio_raw_dim)
    assert synthetic_batch["valence"].shape == (B,)
    assert synthetic_batch["arousal"].shape == (B,)
    assert synthetic_batch["mood"].shape == (B,)
    assert synthetic_batch["cong_label"].shape == (B,)

    assert synthetic_batch["visual_feat"].dtype == torch.float32
    assert synthetic_batch["audio_feat"].dtype == torch.float32
    assert synthetic_batch["mood"].dtype == torch.int64
    assert synthetic_batch["cong_label"].dtype == torch.int64


def test_determinism(config):
    """같은 seed로 두 번 생성하면 동일한 데이터."""
    from ..dataset import SyntheticAutoEQDataset

    ds1 = SyntheticAutoEQDataset(num_clips=20, num_films=5, config=config, seed=42)
    ds2 = SyntheticAutoEQDataset(num_clips=20, num_films=5, config=config, seed=42)

    for i in range(len(ds1)):
        s1 = ds1[i]
        s2 = ds2[i]
        assert torch.allclose(s1["visual_feat"], s2["visual_feat"])
        assert torch.allclose(s1["audio_feat"], s2["audio_feat"])
        assert s1["valence"].item() == s2["valence"].item()
        assert s1["arousal"].item() == s2["arousal"].item()
