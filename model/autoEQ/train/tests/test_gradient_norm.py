"""명세 테스트 6: Gradient norm 측정 및 clipping 검증.

- 첫 batch backward 후 각 head의 grad norm > 0, NaN 아님
- clipping 후 total norm <= 1.0
"""

import torch
import torch.nn as nn

from ..losses import combined_loss
from ..utils import compute_head_grad_norms


def test_grad_norms_nonzero_and_finite(model, config):
    """각 head의 gradient norm이 0이 아니고 NaN이 아닌지 확인."""
    model.train()
    v = torch.randn(4, config.visual_dim)
    a = torch.randn(4, config.audio_raw_dim)
    cong = torch.zeros(4, dtype=torch.long)

    outputs = model(v, a, cong_label=cong)
    va_target = torch.randn(4, 2)
    mood_target = torch.randint(0, 7, (4,))

    loss, _ = combined_loss(outputs, va_target, mood_target, cong, config)
    loss.backward()

    heads = {
        "va": model.va_head,
        "mood": model.mood_head,
        "cong": model.cong_head,
    }
    grad_norms = compute_head_grad_norms(heads)

    for name, norm in grad_norms.items():
        assert norm > 0, f"Head '{name}' has zero gradient norm"
        assert not (norm != norm), f"Head '{name}' has NaN gradient norm"  # NaN check


def test_grad_clipping(model, config):
    """Gradient clipping 후 total norm이 max_norm 이하인지 확인."""
    model.train()
    v = torch.randn(4, config.visual_dim)
    a = torch.randn(4, config.audio_raw_dim)
    cong = torch.zeros(4, dtype=torch.long)

    outputs = model(v, a, cong_label=cong)
    va_target = torch.randn(4, 2)
    mood_target = torch.randint(0, 7, (4,))

    loss, _ = combined_loss(outputs, va_target, mood_target, cong, config)
    loss.backward()

    # Clip
    total_norm = nn.utils.clip_grad_norm_(
        model.parameters(), config.grad_clip_norm
    )

    # Verify post-clip norm
    post_clip_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            post_clip_norm += p.grad.data.norm(2).item() ** 2
    post_clip_norm = post_clip_norm**0.5

    assert post_clip_norm <= config.grad_clip_norm + 1e-6, (
        f"Post-clip norm {post_clip_norm} exceeds max {config.grad_clip_norm}"
    )
