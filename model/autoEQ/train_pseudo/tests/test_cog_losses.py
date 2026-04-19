import torch

from model.autoEQ.train_pseudo.config import TrainCogConfig
from model.autoEQ.train_pseudo.losses import combined_loss_cog


def _fake_outputs(B: int = 4, K: int = 7) -> dict[str, torch.Tensor]:
    return {
        "va_pred": torch.randn(B, 2),
        "mood_logits": torch.randn(B, K),
        "gate_weights": torch.softmax(torch.randn(B, 2), dim=-1),
    }


def test_loss_keys_no_cong():
    cfg = TrainCogConfig()
    outs = _fake_outputs()
    va = torch.randn(4, 2)
    mood = torch.randint(0, cfg.num_mood_classes, (4,))
    total, ld = combined_loss_cog(outs, va, mood, cfg)
    assert set(ld.keys()) >= {"va", "mood", "gate_entropy", "total"}
    assert "cong" not in ld
    assert isinstance(total.item(), float)


def test_weighted_sum_matches():
    cfg = TrainCogConfig(use_ccc_loss=False)
    outs = _fake_outputs()
    va = torch.randn(4, 2)
    mood = torch.randint(0, cfg.num_mood_classes, (4,))
    total, ld = combined_loss_cog(outs, va, mood, cfg)
    expected = (
        cfg.lambda_va * ld["va"]
        + cfg.lambda_mood * ld["mood"]
        + cfg.lambda_gate_entropy * ld["gate_entropy"]
    )
    assert abs(total.item() - expected) < 1e-5


def test_rejects_cong_target_kwarg():
    import pytest
    cfg = TrainCogConfig()
    outs = _fake_outputs()
    va = torch.randn(4, 2)
    mood = torch.randint(0, cfg.num_mood_classes, (4,))
    with pytest.raises(TypeError):
        combined_loss_cog(
            outs, va, mood,
            cong_target=torch.zeros(4, dtype=torch.long),
            config=cfg,
        )
