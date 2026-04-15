"""모델 아키텍처 forward pass shape 검증."""

import torch


def test_audio_projection_shape(model, config):
    x = torch.randn(4, config.audio_raw_dim)
    out = model.audio_projection(x)
    assert out.shape == (4, config.audio_proj_dim)


def test_gate_network_shape_and_sum(model, config):
    v = torch.randn(4, config.visual_dim)
    a = torch.randn(4, config.audio_proj_dim)
    gate = model.gate_network(v, a)
    assert gate.shape == (4, 2)
    # Softmax guarantees sum = 1
    assert torch.allclose(gate.sum(dim=-1), torch.ones(4), atol=1e-5)
    # All values non-negative
    assert (gate >= 0).all()


def test_va_head_shape(model, config):
    fused = torch.randn(4, config.fused_dim)
    out = model.va_head(fused)
    assert out.shape == (4, 2)


def test_mood_head_shape(model, config):
    fused = torch.randn(4, config.fused_dim)
    out = model.mood_head(fused)
    assert out.shape == (4, config.num_mood_classes)


def test_cong_head_shape(model, config):
    x = torch.randn(4, config.cong_head_input_dim)
    out = model.cong_head(x)
    assert out.shape == (4, config.num_cong_classes)


def test_full_forward_shapes(model, config):
    B = 4
    v = torch.randn(B, config.visual_dim)
    a = torch.randn(B, config.audio_raw_dim)
    cong = torch.zeros(B, dtype=torch.long)

    outputs = model(v, a, cong_label=cong)

    assert outputs["va_pred"].shape == (B, 2)
    assert outputs["mood_logits"].shape == (B, config.num_mood_classes)
    assert outputs["cong_logits"].shape == (B, config.num_cong_classes)
    assert outputs["gate_weights"].shape == (B, 2)


def test_no_nan_inf_in_outputs(model, config):
    B = 4
    v = torch.randn(B, config.visual_dim)
    a = torch.randn(B, config.audio_raw_dim)

    outputs = model(v, a)
    for key, tensor in outputs.items():
        assert not torch.isnan(tensor).any(), f"NaN in {key}"
        assert not torch.isinf(tensor).any(), f"Inf in {key}"
