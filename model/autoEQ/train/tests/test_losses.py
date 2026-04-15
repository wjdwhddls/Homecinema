"""손실함수 개별 검증: 알려진 입력에 대한 정확한 출력."""

import math

import torch

from ..losses import (
    va_mse_loss,
    va_hybrid_loss,
    mood_ce_loss,
    cong_ce_loss,
    gate_entropy_loss,
    combined_loss,
)
from ..config import TrainConfig
from ..utils import compute_ccc


def test_va_mse_perfect():
    pred = torch.tensor([[0.5, -0.3], [0.1, 0.8]])
    target = pred.clone()
    assert va_mse_loss(pred, target).item() < 1e-7


def test_va_mse_known_value():
    pred = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[0.0, 0.0]])
    # MSE = mean((1-0)^2 + (1-0)^2) = mean(1 + 1) = 1.0
    assert abs(va_mse_loss(pred, target).item() - 1.0) < 1e-6


def test_mood_ce_perfect():
    # Large logit at correct class -> near-zero loss
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
    target = torch.tensor([0])
    assert mood_ce_loss(logits, target).item() < 0.01


def test_mood_ce_uniform():
    # Uniform logits -> loss ~= log(7)
    logits = torch.zeros(1, 7)
    target = torch.tensor([0])
    expected = math.log(7)
    assert abs(mood_ce_loss(logits, target).item() - expected) < 0.01


def test_cong_ce_perfect():
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    target = torch.tensor([0])
    assert cong_ce_loss(logits, target).item() < 0.01


def test_cong_ce_uniform():
    logits = torch.zeros(1, 3)
    target = torch.tensor([0])
    expected = math.log(3)
    assert abs(cong_ce_loss(logits, target).item() - expected) < 0.01


def test_gate_entropy_uniform():
    # Uniform [0.5, 0.5] -> max entropy -> most negative loss
    weights = torch.tensor([[0.5, 0.5]])
    loss = gate_entropy_loss(weights)
    expected = math.log(0.5)  # 0.5*log(0.5) + 0.5*log(0.5) = log(0.5) = -0.693
    assert abs(loss.item() - expected) < 0.01


def test_gate_entropy_collapsed():
    # Near-collapsed [0.99, 0.01] -> low entropy -> less negative
    weights = torch.tensor([[0.99, 0.01]])
    loss = gate_entropy_loss(weights)
    # Should be closer to 0 than uniform case
    uniform_loss = gate_entropy_loss(torch.tensor([[0.5, 0.5]]))
    assert loss.item() > uniform_loss.item()


def test_combined_loss_weighted_sum(config):
    B = 4
    outputs = {
        "va_pred": torch.randn(B, 2),
        "mood_logits": torch.randn(B, 7),
        "cong_logits": torch.randn(B, 3),
        "gate_weights": torch.softmax(torch.randn(B, 2), dim=-1),
    }
    va_target = torch.randn(B, 2)
    mood_target = torch.randint(0, 7, (B,))
    cong_target = torch.randint(0, 3, (B,))

    total, loss_dict = combined_loss(outputs, va_target, mood_target, cong_target, config)

    expected = (
        config.lambda_va * loss_dict["va"]
        + config.lambda_mood * loss_dict["mood"]
        + config.lambda_cong * loss_dict["cong"]
        + config.lambda_gate_entropy * loss_dict["gate_entropy"]
    )
    assert abs(total.item() - expected) < 1e-4


def test_ccc_perfect():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_ccc(x, x).item() > 0.999


def test_ccc_anti_correlation():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    assert compute_ccc(x, y).item() < -0.9


# --- Hybrid CCC loss tests ---


def test_hybrid_loss_perfect_prediction():
    """Perfect prediction -> hybrid loss near 0."""
    pred = torch.tensor([[0.5, -0.3], [0.1, 0.8], [-0.5, 0.2], [0.3, -0.7]])
    target = pred.clone()
    loss, mse_val, ccc_val = va_hybrid_loss(pred, target)
    assert loss.item() < 1e-5
    assert mse_val < 1e-7
    assert ccc_val > 0.999


def test_hybrid_loss_between_mse_and_ccc():
    """Hybrid loss is a weighted blend of MSE and (1-CCC)."""
    torch.manual_seed(42)
    pred = torch.randn(16, 2)
    target = torch.randn(16, 2)
    w = 0.3
    loss, mse_val, ccc_val = va_hybrid_loss(pred, target, ccc_weight=w)
    expected = (1 - w) * mse_val + w * (1 - ccc_val)
    assert abs(loss.item() - expected) < 1e-5


def test_hybrid_loss_gradient_finite():
    """Hybrid loss gradients are finite (no NaN/Inf from CCC denominator)."""
    pred = torch.randn(8, 2, requires_grad=True)
    target = torch.randn(8, 2)
    loss, _, _ = va_hybrid_loss(pred, target)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_hybrid_loss_gradient_finite_low_variance():
    """Gradient stability when targets have near-zero variance."""
    pred = torch.randn(8, 2, requires_grad=True)
    target = torch.full((8, 2), 0.5) + torch.randn(8, 2) * 1e-4
    loss, _, _ = va_hybrid_loss(pred, target)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_combined_loss_with_ccc_enabled(config):
    """combined_loss includes CCC components when use_ccc_loss=True."""
    config.use_ccc_loss = True
    config.ccc_loss_weight = 0.3
    B = 8
    outputs = {
        "va_pred": torch.randn(B, 2),
        "mood_logits": torch.randn(B, 7),
        "cong_logits": torch.randn(B, 3),
        "gate_weights": torch.softmax(torch.randn(B, 2), dim=-1),
    }
    va_target = torch.randn(B, 2)
    mood_target = torch.randint(0, 7, (B,))
    cong_target = torch.randint(0, 3, (B,))

    total, loss_dict = combined_loss(outputs, va_target, mood_target, cong_target, config)
    assert "va_mse" in loss_dict
    assert "va_ccc" in loss_dict
    assert total.item() > 0


def test_combined_loss_with_ccc_disabled(config):
    """combined_loss uses pure MSE when use_ccc_loss=False."""
    config.use_ccc_loss = False
    B = 8
    outputs = {
        "va_pred": torch.randn(B, 2),
        "mood_logits": torch.randn(B, 7),
        "cong_logits": torch.randn(B, 3),
        "gate_weights": torch.softmax(torch.randn(B, 2), dim=-1),
    }
    va_target = torch.randn(B, 2)
    mood_target = torch.randint(0, 7, (B,))
    cong_target = torch.randint(0, 3, (B,))

    total, loss_dict = combined_loss(outputs, va_target, mood_target, cong_target, config)
    assert "va_mse" not in loss_dict
    assert "va_ccc" not in loss_dict
