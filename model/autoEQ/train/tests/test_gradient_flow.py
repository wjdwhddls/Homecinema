"""Gradient flow 검증: trainable 파라미터만 gradient를 받는지 확인."""

import torch

from ..losses import combined_loss
from ..utils import count_parameters


def test_trainable_params_receive_gradients(model, config):
    v = torch.randn(4, config.visual_dim)
    a = torch.randn(4, config.audio_raw_dim)
    cong = torch.zeros(4, dtype=torch.long)

    model.train()
    outputs = model(v, a, cong_label=cong)

    va_target = torch.randn(4, 2)
    mood_target = torch.randint(0, 7, (4,))

    loss, _ = combined_loss(outputs, va_target, mood_target, cong, config)
    loss.backward()

    trainable_modules = [
        model.audio_projection,
        model.gate_network,
        model.va_head,
        model.mood_head,
        model.cong_head,
    ]
    for module in trainable_modules:
        for name, p in module.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"


def test_frozen_inputs_no_gradient(model, config):
    """Input feature tensors (simulating frozen encoder outputs) should not have gradients."""
    v = torch.randn(4, config.visual_dim, requires_grad=False)
    a = torch.randn(4, config.audio_raw_dim, requires_grad=False)

    model.train()
    outputs = model(v, a)

    va_target = torch.randn(4, 2)
    mood_target = torch.randint(0, 7, (4,))
    cong_target = torch.zeros(4, dtype=torch.long)

    loss, _ = combined_loss(outputs, va_target, mood_target, cong_target, config)
    loss.backward()

    assert v.grad is None, "Visual input should not accumulate gradient"
    assert a.grad is None, "Audio input should not accumulate gradient"


def test_parameter_count(model):
    trainable, total = count_parameters(model)
    # All parameters in AutoEQModel are trainable (frozen encoders are external)
    assert trainable == total
    assert trainable > 0
