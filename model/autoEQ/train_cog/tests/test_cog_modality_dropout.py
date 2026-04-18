import pytest
import torch

from model.autoEQ.train_cog.config import TrainCogConfig
from model.autoEQ.train_cog.model import AutoEQModelCog


def test_dropout_fires_in_training():
    """With p=1.0 every sample must have exactly one modality zeroed."""
    cfg = TrainCogConfig(modality_dropout_p=1.0)
    model = AutoEQModelCog(cfg)
    model.train()

    torch.manual_seed(0)
    B = 100
    v = torch.ones(B, cfg.visual_dim)
    a = torch.ones(B, cfg.audio_proj_dim)
    v2, a2 = model._apply_modality_dropout(v, a)
    v_zero = (v2.abs().sum(dim=-1) == 0)
    a_zero = (a2.abs().sum(dim=-1) == 0)
    # Exactly one of (v2, a2) is zero per sample
    assert (v_zero ^ a_zero).all()


def test_no_dropout_in_eval_mode():
    cfg = TrainCogConfig(modality_dropout_p=1.0)
    model = AutoEQModelCog(cfg)
    model.eval()
    v = torch.ones(10, cfg.visual_dim)
    a = torch.ones(10, cfg.audio_proj_dim) * 2.0
    v2, a2 = model._apply_modality_dropout(v, a)
    assert torch.equal(v, v2)
    assert torch.equal(a, a2)


def test_no_cong_label_dependency():
    """Signature must not accept cong_label."""
    cfg = TrainCogConfig()
    model = AutoEQModelCog(cfg)
    model.train()
    with pytest.raises(TypeError):
        model._apply_modality_dropout(
            torch.ones(2, cfg.visual_dim),
            torch.ones(2, cfg.audio_proj_dim),
            cong_label=torch.zeros(2, dtype=torch.long),
        )


def test_dropout_rate_roughly_p():
    """With p=0.5 on 1000 samples, roughly half should be dropped (±10%)."""
    cfg = TrainCogConfig(modality_dropout_p=0.5)
    model = AutoEQModelCog(cfg)
    model.train()
    torch.manual_seed(0)
    B = 2000
    v = torch.ones(B, cfg.visual_dim)
    a = torch.ones(B, cfg.audio_proj_dim)
    v2, a2 = model._apply_modality_dropout(v, a)
    v_zero = (v2.abs().sum(dim=-1) == 0).float().mean().item()
    a_zero = (a2.abs().sum(dim=-1) == 0).float().mean().item()
    total_dropped = v_zero + a_zero
    # Expected = p · 1.0 (since every drop_trigger fires drops exactly one side)
    assert abs(total_dropped - 0.5) < 0.05
