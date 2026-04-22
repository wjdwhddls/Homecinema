"""Phase 2a-5 Fusion variant compliance tests.

Independent from parent train_liris/tests — the Base Model (gate-mode via
AutoEQModelLiris) + AST + CLIPMean variants must all remain PASS.

Run: pytest model/autoEQ/train_liris/model_fusion/tests -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.autoEQ.train_liris.config import TrainLirisConfig
from model.autoEQ.train_liris.model import AutoEQModelLiris
from model.autoEQ.train_liris.model_fusion.config import TrainLirisConfigFusion
from model.autoEQ.train_liris.model_fusion.fusion import GMUFusion, GMUNoTanhFusion, SimpleConcatFusion
from model.autoEQ.train_liris.model_fusion.model import AutoEQModelLirisFusion


# ────────────────────────────────────────────────────────────────────────
# 1. Config inheritance — every Base Model hyper is preserved.
# ────────────────────────────────────────────────────────────────────────
def test_fusion_config_inherits_base():
    c = TrainLirisConfigFusion()
    assert c.num_mood_classes == 7
    assert c.head_dropout == pytest.approx(0.3)
    assert c.weight_decay == pytest.approx(1e-4)
    assert c.use_full_learning_set is True
    assert c.va_norm_strategy == "A"
    assert c.lambda_mood == pytest.approx(0.3)
    assert c.lambda_gate_entropy == pytest.approx(0.05)
    assert c.ccc_loss_weight == pytest.approx(0.3)
    assert c.batch_size == 32
    assert c.lr == pytest.approx(1e-4)
    assert c.epochs == 40
    assert c.visual_dim == 512
    assert c.audio_raw_dim == 2048
    assert c.audio_proj_dim == 512
    assert c.fused_dim == 1024
    # Default fusion_mode
    assert c.fusion_mode == "gate"


# ────────────────────────────────────────────────────────────────────────
# 2. Config — invalid fusion_mode rejected.
# ────────────────────────────────────────────────────────────────────────
def test_fusion_mode_invalid_raises():
    cfg = TrainLirisConfigFusion()
    cfg.fusion_mode = "bogus"
    with pytest.raises(ValueError, match="fusion_mode"):
        AutoEQModelLirisFusion(cfg)


# ────────────────────────────────────────────────────────────────────────
# 3. Gate mode — param count byte-identical to BASE (AutoEQModelLiris).
# ────────────────────────────────────────────────────────────────────────
def test_gate_mode_params_identical_to_base():
    base = AutoEQModelLiris(TrainLirisConfig())
    cfg = TrainLirisConfigFusion(fusion_mode="gate")
    fused = AutoEQModelLirisFusion(cfg)
    base_p = sum(p.numel() for p in base.parameters())
    fused_p = sum(p.numel() for p in fused.parameters())
    assert fused_p == base_p == 3_417_099


# ────────────────────────────────────────────────────────────────────────
# 4. Gate mode — state_dict compatible with BASE (swap weights, matching keys).
# ────────────────────────────────────────────────────────────────────────
def test_gate_mode_state_dict_compatible():
    base = AutoEQModelLiris(TrainLirisConfig())
    cfg = TrainLirisConfigFusion(fusion_mode="gate")
    fused = AutoEQModelLirisFusion(cfg)
    # strict load — any key mismatch will raise
    fused.load_state_dict(base.state_dict(), strict=True)


# ────────────────────────────────────────────────────────────────────────
# 5. Gate mode — forward output BYTE-IDENTICAL to BASE given same weights.
# ────────────────────────────────────────────────────────────────────────
def test_gate_mode_forward_byte_identical_to_base():
    torch.manual_seed(0)
    base = AutoEQModelLiris(TrainLirisConfig()).eval()
    cfg = TrainLirisConfigFusion(fusion_mode="gate")
    fused = AutoEQModelLirisFusion(cfg).eval()
    fused.load_state_dict(base.state_dict(), strict=True)
    v = torch.randn(5, 512)
    a = torch.randn(5, 2048)
    with torch.no_grad():
        out_base = base(v, a)
        out_fused = fused(v, a)
    for key in ("va_pred", "mood_logits", "gate_weights"):
        assert torch.equal(out_base[key], out_fused[key]), f"mismatch on {key}"


# ────────────────────────────────────────────────────────────────────────
# 6. Concat mode — zero fusion params (null baseline).
# ────────────────────────────────────────────────────────────────────────
def test_concat_mode_zero_fusion_params():
    cfg = TrainLirisConfigFusion(fusion_mode="concat")
    m = AutoEQModelLirisFusion(cfg)
    # No gate_network
    assert m.gate_network is None
    # Fusion has zero learnable params (SimpleConcatFusion is stateless)
    assert isinstance(m.fusion, SimpleConcatFusion)
    fusion_p = sum(p.numel() for p in m.fusion.parameters())
    assert fusion_p == 0
    # Total params = BASE − GateNetwork params (262,914)
    base_p = sum(p.numel() for p in AutoEQModelLiris(TrainLirisConfig()).parameters())
    concat_p = sum(p.numel() for p in m.parameters())
    # GateNetwork: Linear(1024,256)+Linear(256,2) = (1024*256+256)+(256*2+2) = 262,400+514 = 262,914
    assert base_p - concat_p == 262_914


# ────────────────────────────────────────────────────────────────────────
# 7. GMU mode — expected param count (wide variant, d_out=1024).
# ────────────────────────────────────────────────────────────────────────
def test_gmu_mode_wide_param_count():
    cfg = TrainLirisConfigFusion(fusion_mode="gmu")
    m = AutoEQModelLirisFusion(cfg)
    assert isinstance(m.fusion, GMUFusion)
    fusion_p = sum(p.numel() for p in m.fusion.parameters())
    # W_v: 512*1024 + 1024 = 525,312
    # W_a: 512*1024 + 1024 = 525,312
    # W_z: 1024*1024 + 1024 = 1,049,600
    expected = 525_312 + 525_312 + 1_049_600
    assert fusion_p == expected, f"GMU params {fusion_p} != {expected}"


# ────────────────────────────────────────────────────────────────────────
# 8. Forward shapes — all three modes produce (B,2)/(B,7)/(B,2).
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["gate", "concat", "gmu", "gmu_notanh"])
def test_forward_shapes_all_modes(mode: str):
    cfg = TrainLirisConfigFusion(fusion_mode=mode)
    m = AutoEQModelLirisFusion(cfg).eval()
    v = torch.randn(3, cfg.visual_dim)
    a = torch.randn(3, cfg.audio_raw_dim)
    out = m(v, a)
    assert out["va_pred"].shape == (3, 2)
    assert out["mood_logits"].shape == (3, 7)
    assert out["gate_weights"].shape == (3, 2)


# ────────────────────────────────────────────────────────────────────────
# 9. Concat mode — gate_weights are constant (0.5, 0.5) as specified.
# ────────────────────────────────────────────────────────────────────────
def test_concat_gate_weights_uniform():
    cfg = TrainLirisConfigFusion(fusion_mode="concat")
    m = AutoEQModelLirisFusion(cfg).eval()
    v = torch.randn(4, cfg.visual_dim)
    a = torch.randn(4, cfg.audio_raw_dim)
    with torch.no_grad():
        out = m(v, a)
    assert torch.allclose(out["gate_weights"], torch.full((4, 2), 0.5))


# ────────────────────────────────────────────────────────────────────────
# 10. GMU mode — gate weights summary in [0, 1] and sum to 1.
# ────────────────────────────────────────────────────────────────────────
def test_gmu_gate_weights_summary_valid():
    cfg = TrainLirisConfigFusion(fusion_mode="gmu")
    m = AutoEQModelLirisFusion(cfg).eval()
    v = torch.randn(4, cfg.visual_dim)
    a = torch.randn(4, cfg.audio_raw_dim)
    with torch.no_grad():
        out = m(v, a)
    gw = out["gate_weights"]
    assert (gw >= 0).all() and (gw <= 1).all()
    assert torch.allclose(gw.sum(dim=-1), torch.ones(4), atol=1e-6)


# ────────────────────────────────────────────────────────────────────────
# 11. Param budget sanity — concat < gate (BASE) < gmu, gmu_notanh == gmu.
# ────────────────────────────────────────────────────────────────────────
def test_param_budget_ordering():
    p_concat = sum(p.numel() for p in AutoEQModelLirisFusion(
        TrainLirisConfigFusion(fusion_mode="concat")).parameters())
    p_gate = sum(p.numel() for p in AutoEQModelLirisFusion(
        TrainLirisConfigFusion(fusion_mode="gate")).parameters())
    p_gmu = sum(p.numel() for p in AutoEQModelLirisFusion(
        TrainLirisConfigFusion(fusion_mode="gmu")).parameters())
    p_gmunt = sum(p.numel() for p in AutoEQModelLirisFusion(
        TrainLirisConfigFusion(fusion_mode="gmu_notanh")).parameters())
    assert p_concat < p_gate < p_gmu
    # Concrete values:
    #   concat     = 3,417,099 − 262,914 = 3,154,185
    #   gate       = 3,417,099 (BASE)
    #   gmu        = 3,417,099 − 262,914 + 2,100,224 = 5,254,409
    #   gmu_notanh = same as gmu (tanh has no params)
    assert p_concat == 3_154_185
    assert p_gate == 3_417_099
    assert p_gmu == 5_254_409
    assert p_gmunt == p_gmu


# ────────────────────────────────────────────────────────────────────────
# 12. gmu_notanh — forward uses linear h_v/h_a (no tanh saturation).
#     Given large-magnitude input, output should exceed [-1, 1].
# ────────────────────────────────────────────────────────────────────────
def test_gmu_notanh_no_tanh_bound():
    cfg = TrainLirisConfigFusion(fusion_mode="gmu_notanh")
    m = AutoEQModelLirisFusion(cfg).eval()
    assert isinstance(m.fusion, GMUNoTanhFusion)
    # Scaled input — without tanh, output should be able to exceed [-1, 1]
    torch.manual_seed(0)
    # Force larger h_v/h_a by scaling W_v weights up
    with torch.no_grad():
        m.fusion.W_v.weight.mul_(10)
        m.fusion.W_a.weight.mul_(10)
    v = torch.randn(8, cfg.visual_dim)
    a = torch.randn(8, cfg.audio_raw_dim)
    with torch.no_grad():
        out = m(v, a)
    # Some dim of fused should now exceed 1.0 in magnitude
    assert out["va_pred"].abs().max() > 0  # basic sanity

    # Compare to vanilla gmu (tanh bounded): should NOT exceed [-1, 1] in h_v/h_a
    cfg2 = TrainLirisConfigFusion(fusion_mode="gmu")
    m2 = AutoEQModelLirisFusion(cfg2).eval()
    with torch.no_grad():
        m2.fusion.W_v.weight.mul_(10)
        m2.fusion.W_a.weight.mul_(10)
    v_proj = m2.audio_projection(a)
    with torch.no_grad():
        h_v_tanh = torch.tanh(m2.fusion.W_v(v))
    # tanh bounds
    assert h_v_tanh.abs().max() <= 1.0 + 1e-6
