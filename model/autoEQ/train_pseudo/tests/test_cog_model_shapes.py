import pytest
import torch

from model.autoEQ.train_pseudo.config import TrainCogConfig
from model.autoEQ.train_pseudo.model import AutoEQModelCog


def test_forward_returns_expected_keys():
    cfg = TrainCogConfig()
    model = AutoEQModelCog(cfg)
    model.eval()
    B = 4
    v = torch.randn(B, cfg.visual_dim)
    a = torch.randn(B, cfg.audio_raw_dim)
    out = model(v, a)
    assert set(out.keys()) == {"va_pred", "mood_logits", "gate_weights"}
    assert "cong_logits" not in out


def test_forward_shapes_k7():
    cfg = TrainCogConfig(num_mood_classes=7)
    model = AutoEQModelCog(cfg)
    model.eval()
    B = 3
    out = model(torch.randn(B, cfg.visual_dim), torch.randn(B, cfg.audio_raw_dim))
    assert out["va_pred"].shape == (B, 2)
    assert out["mood_logits"].shape == (B, 7)
    assert out["gate_weights"].shape == (B, 2)


def test_forward_shapes_k4():
    cfg = TrainCogConfig(num_mood_classes=4)
    model = AutoEQModelCog(cfg)
    model.eval()
    B = 3
    out = model(torch.randn(B, cfg.visual_dim), torch.randn(B, cfg.audio_raw_dim))
    assert out["mood_logits"].shape == (B, 4)


def test_forward_rejects_cong_label_kwarg():
    cfg = TrainCogConfig()
    model = AutoEQModelCog(cfg)
    with pytest.raises(TypeError):
        model(
            torch.randn(2, cfg.visual_dim),
            torch.randn(2, cfg.audio_raw_dim),
            cong_label=torch.zeros(2, dtype=torch.long),
        )


def test_gate_weights_sum_to_one():
    cfg = TrainCogConfig()
    model = AutoEQModelCog(cfg)
    model.eval()
    out = model(torch.randn(5, cfg.visual_dim), torch.randn(5, cfg.audio_raw_dim))
    sums = out["gate_weights"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
