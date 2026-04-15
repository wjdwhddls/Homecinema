"""명세 테스트 3: Modality dropout 조건부 동작 검증.

- cong_label=0 + train mode: dropout 발동
- cong_label!=0: dropout 미발동
- eval mode: dropout 미발동
"""

import torch

from ..config import TrainConfig
from ..model import AutoEQModel


def test_dropout_triggers_on_congruent_train():
    """Train mode + cong_label=0 + p=1.0: 반드시 한 modality가 zeroed."""
    config = TrainConfig(modality_dropout_p=1.0)  # force 100% trigger
    model = AutoEQModel(config)
    model.train()

    B = 16
    v = torch.randn(B, config.visual_dim)
    a = torch.randn(B, config.audio_raw_dim)
    cong = torch.zeros(B, dtype=torch.long)  # all congruent

    # Run multiple times; with p=1.0 every congruent sample should be affected
    dropout_detected = False
    for _ in range(10):
        outputs = model(v, a, cong_label=cong)
        # We can't directly observe the internal dropout, but we test via the model's
        # _apply_modality_dropout method
        v_test = v.clone()
        a_proj = model.audio_projection(a)
        v_dropped, a_dropped = model._apply_modality_dropout(v_test, a_proj, cong)

        # At least one modality should be zeroed for each sample
        v_zeroed = (v_dropped.abs().sum(dim=-1) == 0)
        a_zeroed = (a_dropped.abs().sum(dim=-1) == 0)
        either_zeroed = v_zeroed | a_zeroed

        if either_zeroed.all():
            dropout_detected = True
            break

    assert dropout_detected, "With p=1.0, all congruent samples should have dropout"


def test_no_dropout_on_incongruent():
    """cong_label != 0: 절대 dropout 미발동."""
    config = TrainConfig(modality_dropout_p=1.0)  # force trigger
    model = AutoEQModel(config)
    model.train()

    B = 16
    v = torch.randn(B, config.visual_dim)
    a_proj = torch.randn(B, config.audio_proj_dim)
    cong = torch.ones(B, dtype=torch.long)  # all slight incongruent

    for _ in range(10):
        v_out, a_out = model._apply_modality_dropout(v.clone(), a_proj.clone(), cong)
        # Nothing should be zeroed
        v_zeroed = (v_out.abs().sum(dim=-1) == 0)
        a_zeroed = (a_out.abs().sum(dim=-1) == 0)
        assert not v_zeroed.any(), "Visual zeroed for incongruent sample"
        assert not a_zeroed.any(), "Audio zeroed for incongruent sample"


def test_no_dropout_in_eval():
    """Eval mode: 어떤 cong_label이든 dropout 미발동."""
    config = TrainConfig(modality_dropout_p=1.0)
    model = AutoEQModel(config)
    model.eval()

    B = 16
    v = torch.randn(B, config.visual_dim)
    a_proj = torch.randn(B, config.audio_proj_dim)
    cong = torch.zeros(B, dtype=torch.long)

    v_out, a_out = model._apply_modality_dropout(v.clone(), a_proj.clone(), cong)
    # In eval mode, nothing should change
    assert torch.allclose(v_out, v)
    assert torch.allclose(a_out, a_proj)


def test_never_both_zeroed():
    """Dropout은 visual 또는 audio 중 하나만 zero, 둘 다 zero는 불가."""
    config = TrainConfig(modality_dropout_p=1.0)
    model = AutoEQModel(config)
    model.train()

    B = 32
    v = torch.randn(B, config.visual_dim)
    a_proj = torch.randn(B, config.audio_proj_dim)
    cong = torch.zeros(B, dtype=torch.long)

    for _ in range(20):
        v_out, a_out = model._apply_modality_dropout(v.clone(), a_proj.clone(), cong)
        v_zeroed = (v_out.abs().sum(dim=-1) == 0)
        a_zeroed = (a_out.abs().sum(dim=-1) == 0)
        both_zeroed = v_zeroed & a_zeroed
        assert not both_zeroed.any(), "Both modalities zeroed simultaneously"
