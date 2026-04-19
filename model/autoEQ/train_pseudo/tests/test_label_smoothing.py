"""Conditional label smoothing in train collate_fn."""

import torch

from model.autoEQ.train_pseudo.config import TrainCogConfig
from model.autoEQ.train_pseudo.dataset import make_train_collate_fn


def _sample(v: float, a: float, v_std: float = 0.0, a_std: float = 0.0) -> dict:
    return {
        "visual_feat": torch.zeros(512),
        "audio_feat": torch.zeros(2048),
        "valence": torch.tensor(v, dtype=torch.float32),
        "arousal": torch.tensor(a, dtype=torch.float32),
        "mood": torch.tensor(0, dtype=torch.long),
        "movie_id": 0,
        "valence_std": torch.tensor(v_std, dtype=torch.float32),
        "arousal_std": torch.tensor(a_std, dtype=torch.float32),
    }


def test_smoothing_disabled_when_eps_zero():
    cfg = TrainCogConfig(label_smooth_eps=0.0, label_smooth_sigma_threshold=0.1,
                         mixup_prob=0.0)
    collate = make_train_collate_fn(cfg)
    out = collate([_sample(0.8, -0.6, v_std=1.0, a_std=1.0)])
    assert torch.allclose(out["valence"], torch.tensor([0.8]))
    assert torch.allclose(out["arousal"], torch.tensor([-0.6]))


def test_smoothing_disabled_when_threshold_zero():
    # threshold must be > 0 to activate per config semantics
    cfg = TrainCogConfig(label_smooth_eps=0.1, label_smooth_sigma_threshold=0.0,
                         mixup_prob=0.0)
    collate = make_train_collate_fn(cfg)
    out = collate([_sample(0.8, -0.6, v_std=1.0, a_std=1.0)])
    assert torch.allclose(out["valence"], torch.tensor([0.8]))
    assert torch.allclose(out["arousal"], torch.tensor([-0.6]))


def test_smoothing_applied_for_high_sigma():
    cfg = TrainCogConfig(label_smooth_eps=0.05, label_smooth_sigma_threshold=0.1,
                         mixup_prob=0.0)
    collate = make_train_collate_fn(cfg)
    out = collate([_sample(0.8, -0.6, v_std=0.3, a_std=0.0)])  # vsd > 0.1
    assert torch.allclose(out["valence"], torch.tensor([0.8 * 0.95]), atol=1e-6)
    assert torch.allclose(out["arousal"], torch.tensor([-0.6 * 0.95]), atol=1e-6)


def test_smoothing_skipped_for_low_sigma():
    cfg = TrainCogConfig(label_smooth_eps=0.05, label_smooth_sigma_threshold=0.1,
                         mixup_prob=0.0)
    collate = make_train_collate_fn(cfg)
    out = collate([_sample(0.8, -0.6, v_std=0.05, a_std=0.05)])  # both <= 0.1
    assert torch.allclose(out["valence"], torch.tensor([0.8]))
    assert torch.allclose(out["arousal"], torch.tensor([-0.6]))


def test_smoothing_preserves_batch_shape():
    cfg = TrainCogConfig(label_smooth_eps=0.05, label_smooth_sigma_threshold=0.1,
                         mixup_prob=0.0)
    collate = make_train_collate_fn(cfg)
    batch = [_sample(0.5, 0.5, 0.2, 0.0),   # smoothed
             _sample(-0.5, -0.5, 0.0, 0.0),  # raw
             _sample(0.3, -0.3, 0.5, 0.5)]  # smoothed
    out = collate(batch)
    assert out["valence"].shape == (3,)
    assert out["arousal"].shape == (3,)
    assert torch.allclose(out["valence"][0], torch.tensor(0.5 * 0.95), atol=1e-6)
    assert torch.allclose(out["valence"][1], torch.tensor(-0.5), atol=1e-6)
    assert torch.allclose(out["valence"][2], torch.tensor(0.3 * 0.95), atol=1e-6)
