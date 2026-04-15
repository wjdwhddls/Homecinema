"""학습 sanity check: overfit-one-batch + loss 감소 테스트."""

import torch
import torch.nn.functional as F

from ..config import TrainConfig
from ..model import AutoEQModel
from ..losses import combined_loss


def test_overfit_one_batch():
    """단일 배치를 200 step 학습 -> 거의 완벽하게 암기 가능한지 확인."""
    config = TrainConfig()
    model = AutoEQModel(config)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 8
    torch.manual_seed(0)
    v = torch.randn(B, config.visual_dim)
    a = torch.randn(B, config.audio_raw_dim)
    va_target = torch.rand(B, 2) * 2 - 1
    mood_target = torch.randint(0, 7, (B,))
    cong_target = torch.randint(0, 3, (B,))

    for step in range(200):
        outputs = model(v, a, cong_label=cong_target)
        loss, _ = combined_loss(outputs, va_target, mood_target, cong_target, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Verify near-perfect memorization
    model.eval()
    with torch.no_grad():
        outputs = model(v, a, cong_label=cong_target)

    va_mse = F.mse_loss(outputs["va_pred"], va_target).item()
    mood_pred = outputs["mood_logits"].argmax(dim=-1)
    mood_acc = (mood_pred == mood_target).float().mean().item()
    cong_pred = outputs["cong_logits"].argmax(dim=-1)
    cong_acc = (cong_pred == cong_target).float().mean().item()

    assert va_mse < 0.05, f"VA MSE {va_mse} too high for memorization"
    assert mood_acc >= 0.75, f"Mood accuracy {mood_acc} too low for memorization"
    assert cong_acc >= 0.75, f"Cong accuracy {cong_acc} too low for memorization"


def test_loss_decreases():
    """50 step 학습 -> 평균 loss가 감소하는지 확인."""
    config = TrainConfig()
    model = AutoEQModel(config)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B = 16
    torch.manual_seed(42)
    v = torch.randn(B, config.visual_dim)
    a = torch.randn(B, config.audio_raw_dim)
    va_target = torch.rand(B, 2) * 2 - 1
    mood_target = torch.randint(0, 7, (B,))
    cong_target = torch.randint(0, 3, (B,))

    losses = []
    for step in range(50):
        outputs = model(v, a, cong_label=cong_target)
        loss, _ = combined_loss(outputs, va_target, mood_target, cong_target, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_10_mean = sum(losses[:10]) / 10
    last_10_mean = sum(losses[-10:]) / 10

    assert last_10_mean < first_10_mean, (
        f"Loss did not decrease: first 10 mean={first_10_mean:.4f}, "
        f"last 10 mean={last_10_mean:.4f}"
    )
