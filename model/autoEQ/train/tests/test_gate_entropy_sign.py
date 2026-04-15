"""명세 테스트 1: Gate entropy loss 부호 검증.

균등 분포 입력 시 손실이 작고, 극단 분포 입력 시 손실이 큰지 확인.
"""

import torch

from ..losses import gate_entropy_loss


def test_uniform_has_lowest_loss():
    """균등 분포 [0.5, 0.5]가 가장 낮은 (most negative) 손실을 생성."""
    uniform = torch.tensor([[0.5, 0.5]])
    collapsed = torch.tensor([[0.99, 0.01]])
    moderate = torch.tensor([[0.7, 0.3]])

    loss_uniform = gate_entropy_loss(uniform).item()
    loss_collapsed = gate_entropy_loss(collapsed).item()
    loss_moderate = gate_entropy_loss(moderate).item()

    # Uniform should be most negative (lowest)
    assert loss_uniform < loss_moderate < loss_collapsed


def test_total_loss_rewards_high_entropy(config):
    """total_loss에 lambda * gate_entropy를 더할 때, entropy 높을수록 total loss 감소."""
    base_loss = torch.tensor(1.0)

    uniform_gate = torch.tensor([[0.5, 0.5]])
    collapsed_gate = torch.tensor([[0.99, 0.01]])

    total_uniform = base_loss + config.lambda_gate_entropy * gate_entropy_loss(uniform_gate)
    total_collapsed = base_loss + config.lambda_gate_entropy * gate_entropy_loss(collapsed_gate)

    # Higher entropy (uniform) should yield lower total loss
    assert total_uniform.item() < total_collapsed.item()


def test_gate_entropy_always_non_positive():
    """Gate entropy loss는 항상 <= 0 (w*log(w)의 합)."""
    for _ in range(100):
        weights = torch.softmax(torch.randn(8, 2), dim=-1)
        loss = gate_entropy_loss(weights)
        assert loss.item() <= 1e-7  # should be <= 0 (with eps tolerance)
