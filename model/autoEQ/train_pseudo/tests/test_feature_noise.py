"""Gaussian noise hook on frozen features."""

import torch

from model.autoEQ.train_pseudo.config import TrainCogConfig
from model.autoEQ.train_pseudo.model import AutoEQModelCog


def _make_batch(B: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(B, 512, generator=g)
    a = torch.randn(B, 2048, generator=g)
    return v, a


def test_noise_default_off_is_identity_for_va_branch():
    cfg = TrainCogConfig(feature_noise_std=0.0, modality_dropout_p=0.0)
    model = AutoEQModelCog(cfg)
    v, a = _make_batch()

    torch.manual_seed(0)
    out_a = model(v, a)["va_pred"]
    torch.manual_seed(0)
    out_b = model(v, a)["va_pred"]

    assert torch.allclose(out_a, out_b, atol=0.0), (
        "With noise_std=0 and dropout=0, forward must be deterministic"
    )


def test_noise_off_in_eval_mode_regardless_of_std():
    cfg = TrainCogConfig(feature_noise_std=0.5, modality_dropout_p=0.0)
    model = AutoEQModelCog(cfg).eval()
    v, a = _make_batch()

    out_a = model(v, a)["va_pred"]
    out_b = model(v, a)["va_pred"]

    assert torch.allclose(out_a, out_b), (
        "eval() must disable feature noise"
    )


def test_noise_changes_output_in_training():
    cfg = TrainCogConfig(feature_noise_std=0.5, modality_dropout_p=0.0)
    model = AutoEQModelCog(cfg).train()
    v, a = _make_batch()

    torch.manual_seed(1)
    out_a = model(v, a)["va_pred"]
    torch.manual_seed(2)
    out_b = model(v, a)["va_pred"]

    assert not torch.allclose(out_a, out_b), (
        "Large noise_std should produce different va_pred across RNG states"
    )


def test_noise_bypasses_dropped_modality():
    # With modality_dropout_p=1.0 and noise std > 0: dropped samples stay zero
    cfg = TrainCogConfig(feature_noise_std=0.5, modality_dropout_p=1.0)
    model = AutoEQModelCog(cfg).train()
    v, a = _make_batch(B=8)

    # Hook into _apply_feature_noise by mocking dropout output
    v_drop, a_drop = model._apply_modality_dropout(v, model.audio_projection(a))
    v_after, a_after = model._apply_feature_noise(v_drop, a_drop)

    for i in range(v.size(0)):
        if v_drop[i].abs().sum().item() == 0:
            assert v_after[i].abs().sum().item() == 0, (
                f"sample {i}: dropped visual must stay zero after noise"
            )
        if a_drop[i].abs().sum().item() == 0:
            assert a_after[i].abs().sum().item() == 0, (
                f"sample {i}: dropped audio must stay zero after noise"
            )
