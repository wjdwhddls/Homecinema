"""Quadrant-restricted mixup collate_fn."""

import torch

from model.autoEQ.train_pseudo.config import TrainCogConfig
from model.autoEQ.train_pseudo.dataset import (
    make_train_collate_fn,
    va_to_quadrant,
)


def _sample(v: float, a: float, visual_fill: float = 1.0, audio_fill: float = 2.0) -> dict:
    return {
        "visual_feat": torch.full((512,), float(visual_fill), dtype=torch.float32),
        "audio_feat": torch.full((2048,), float(audio_fill), dtype=torch.float32),
        "valence": torch.tensor(v, dtype=torch.float32),
        "arousal": torch.tensor(a, dtype=torch.float32),
        "mood": torch.tensor(0, dtype=torch.long),
        "movie_id": 0,
        "valence_std": torch.tensor(0.0, dtype=torch.float32),
        "arousal_std": torch.tensor(0.0, dtype=torch.float32),
    }


def test_mixup_disabled_is_identity():
    cfg = TrainCogConfig(mixup_prob=0.0, label_smooth_eps=0.0)
    collate = make_train_collate_fn(cfg)
    batch = [_sample(0.5, 0.5, i, i * 10) for i in range(4)]
    out = collate(batch)
    assert out["visual_feat"].shape == (4, 512)
    assert out["audio_feat"].shape == (4, 2048)
    # Values unchanged (identity)
    for i in range(4):
        assert torch.allclose(out["visual_feat"][i], torch.full((512,), float(i), dtype=torch.float32))


def test_mixup_same_quadrant_keeps_quadrant():
    cfg = TrainCogConfig(mixup_prob=1.0, mixup_alpha=0.4, label_smooth_eps=0.0)
    collate = make_train_collate_fn(cfg)
    # All HVHA (v>0, a>0)
    batch = [_sample(0.6, 0.6, 1.0, 2.0),
             _sample(0.4, 0.7, 3.0, 4.0),
             _sample(0.8, 0.5, 5.0, 6.0),
             _sample(0.3, 0.9, 7.0, 8.0)]
    torch.manual_seed(0)
    out = collate(batch)
    # Post-mix V/A must still lie in HVHA quadrant (same sign) because mixup
    # restricted to same-quadrant pairs and both signs preserved.
    for i in range(4):
        v = float(out["valence"][i])
        a = float(out["arousal"][i])
        assert v > 0 and a > 0, f"sample {i} escaped HVHA: v={v}, a={a}"


def test_mixup_does_not_cross_quadrants():
    # Batch of 2 samples in DIFFERENT quadrants → no valid partner, both stay raw
    cfg = TrainCogConfig(mixup_prob=1.0, mixup_alpha=0.4, label_smooth_eps=0.0)
    collate = make_train_collate_fn(cfg)
    s0 = _sample(0.7, 0.7, 1.0, 2.0)      # HVHA
    s1 = _sample(-0.7, -0.7, 3.0, 4.0)    # LVLA
    torch.manual_seed(0)
    out = collate([s0, s1])
    # s0 visual must remain 1.0 (no partner), s1 must remain 3.0
    assert torch.allclose(out["visual_feat"][0], torch.full((512,), 1.0, dtype=torch.float32))
    assert torch.allclose(out["visual_feat"][1], torch.full((512,), 3.0, dtype=torch.float32))
    assert torch.allclose(out["valence"], torch.tensor([0.7, -0.7]))


def test_mixup_lambda_inside_safe_range():
    # Verify λ shrunk to [0.1, 0.9] so no sample becomes nearly-copied of partner.
    cfg = TrainCogConfig(mixup_prob=1.0, mixup_alpha=0.1, label_smooth_eps=0.0)
    collate = make_train_collate_fn(cfg)
    s0 = _sample(0.5, 0.5, 10.0, 20.0)  # primary
    s1 = _sample(0.6, 0.6, 0.0, 0.0)    # partner, same quadrant
    mins, maxs = [], []
    for seed in range(50):
        torch.manual_seed(seed)
        out = collate([s0, s1])
        mins.append(float(out["visual_feat"][0, 0]))
        maxs.append(float(out["visual_feat"][0, 0]))
    # primary at λ=1 would give 10.0, at λ=0 would give 0.0
    # with shrink to [0.1, 0.9]: range of primary value ∈ [1.0, 9.0]
    lo = min(mins + maxs)
    hi = max(mins + maxs)
    assert 1.0 - 0.5 <= lo and hi <= 9.0 + 0.5, (
        f"mixup λ should be clamped to [0.1, 0.9] → primary∈[1,9]; got [{lo}, {hi}]"
    )
