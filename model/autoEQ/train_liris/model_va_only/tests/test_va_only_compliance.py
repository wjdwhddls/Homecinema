"""Phase 2a-7 VA-only (no MoodHead) compliance tests.

Verification goals:
    1. Config inherits BASE correctly with lambda_mood=0 override
    2. MoodHead actually REMOVED (not present as module)
    3. Parameter count exactly 3,152,388 (BASE 3,417,099 − MoodHead 264,711)
    4. Forward output interface preserved (va_pred, mood_logits, gate_weights)
    5. mood_logits is truly all zeros (dummy, trainer-compat)
    6. Other components (AudioProjection / GateNetwork / VAHead) byte-identical
    7. Non-va_head/mood_head components state_dict keys match BASE
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.autoEQ.train_liris.config import TrainLirisConfig
from model.autoEQ.train_liris.model import AutoEQModelLiris
from model.autoEQ.train_liris.model_va_only.config import TrainLirisConfigVAOnly
from model.autoEQ.train_liris.model_va_only.model import AutoEQModelLirisVAOnly


# ────────────────────────────────────────────────────────────────────────
# 1. Config inheritance — BASE hyperparameters preserved except lambda_mood
# ────────────────────────────────────────────────────────────────────────
def test_config_inherits_base_except_lambda_mood():
    c = TrainLirisConfigVAOnly()
    # From TrainLirisConfig (BASE) — preserved
    assert c.num_mood_classes == 7
    assert c.head_dropout == pytest.approx(0.3)
    assert c.weight_decay == pytest.approx(1e-4)
    assert c.use_full_learning_set is True
    assert c.va_norm_strategy == "A"
    assert c.visual_dim == 512
    assert c.audio_raw_dim == 2048
    assert c.fused_dim == 1024
    assert c.batch_size == 32
    assert c.lr == pytest.approx(1e-4)
    assert c.epochs == 40
    assert c.lambda_va == pytest.approx(1.0)
    assert c.lambda_gate_entropy == pytest.approx(0.05)
    assert c.ccc_loss_weight == pytest.approx(0.3)
    # Override: lambda_mood = 0
    assert c.lambda_mood == pytest.approx(0.0)
    # Run identity override
    assert c.run_name == "2a7_va_only"


# ────────────────────────────────────────────────────────────────────────
# 2. MoodHead is REMOVED from the model (not present as a module)
# ────────────────────────────────────────────────────────────────────────
def test_mood_head_removed():
    cfg = TrainLirisConfigVAOnly()
    m = AutoEQModelLirisVAOnly(cfg)
    # BASE has self.mood_head; VA-only should NOT have it
    assert not hasattr(m, "mood_head") or m.mood_head is None, (
        "VA-only should not have MoodHead module"
    )
    # state_dict should contain NO keys starting with 'mood_head.'
    mood_keys = [k for k in m.state_dict() if k.startswith("mood_head.")]
    assert mood_keys == [], f"Unexpected mood_head keys in state_dict: {mood_keys}"


# ────────────────────────────────────────────────────────────────────────
# 3. Parameter count — exactly BASE − MoodHead
# ────────────────────────────────────────────────────────────────────────
def test_param_count_base_minus_mood_head():
    base = AutoEQModelLiris(TrainLirisConfig())
    va_only = AutoEQModelLirisVAOnly(TrainLirisConfigVAOnly())

    base_p = sum(p.numel() for p in base.parameters())
    va_only_p = sum(p.numel() for p in va_only.parameters())
    mood_p = sum(p.numel() for p in base.mood_head.parameters())

    assert base_p == 3_417_099
    assert mood_p == 264_711
    assert va_only_p == base_p - mood_p == 3_152_388


# ────────────────────────────────────────────────────────────────────────
# 4. Forward output interface — same 3 keys as BASE, same shapes
# ────────────────────────────────────────────────────────────────────────
def test_forward_output_interface():
    cfg = TrainLirisConfigVAOnly()
    m = AutoEQModelLirisVAOnly(cfg).eval()
    v = torch.randn(4, cfg.visual_dim)
    a = torch.randn(4, cfg.audio_raw_dim)
    with torch.no_grad():
        out = m(v, a)
    # Exact 3 keys expected — matches BASE trainer expectations
    assert set(out.keys()) == {"va_pred", "mood_logits", "gate_weights"}
    assert out["va_pred"].shape == (4, 2)
    assert out["mood_logits"].shape == (4, 7)
    assert out["gate_weights"].shape == (4, 2)


# ────────────────────────────────────────────────────────────────────────
# 5. mood_logits MUST be exactly zeros (dummy for trainer compat)
# ────────────────────────────────────────────────────────────────────────
def test_mood_logits_are_dummy_zeros():
    cfg = TrainLirisConfigVAOnly()
    m = AutoEQModelLirisVAOnly(cfg).eval()
    v = torch.randn(8, cfg.visual_dim)
    a = torch.randn(8, cfg.audio_raw_dim)
    with torch.no_grad():
        out = m(v, a)
    # All zeros — no gradient path from mood, no meaningful output
    assert torch.equal(out["mood_logits"], torch.zeros(8, 7)), (
        "mood_logits must be exactly zeros (dummy trainer-compat output)"
    )


# ────────────────────────────────────────────────────────────────────────
# 6. VAHead forward byte-identical to BASE given same weights
# ────────────────────────────────────────────────────────────────────────
def test_va_head_byte_identical_to_base():
    torch.manual_seed(0)
    base = AutoEQModelLiris(TrainLirisConfig()).eval()
    va_only = AutoEQModelLirisVAOnly(TrainLirisConfigVAOnly()).eval()
    # Copy the shared components from base to va_only (they have matching keys)
    va_only_sd = va_only.state_dict()
    for k, v in base.state_dict().items():
        if k in va_only_sd:
            va_only_sd[k] = v
    va_only.load_state_dict(va_only_sd)

    vis = torch.randn(5, 512)
    aud = torch.randn(5, 2048)
    with torch.no_grad():
        o_base = base(vis, aud)
        o_vo = va_only(vis, aud)
    # va_pred and gate_weights must be identical (shared components, same weights)
    assert torch.equal(o_base["va_pred"], o_vo["va_pred"]), \
        "va_pred must be byte-identical when shared components use same weights"
    assert torch.equal(o_base["gate_weights"], o_vo["gate_weights"])


# ────────────────────────────────────────────────────────────────────────
# 7. state_dict key comparison — VA-only keys ⊂ BASE keys (minus mood_head)
# ────────────────────────────────────────────────────────────────────────
def test_state_dict_subset_of_base():
    base = AutoEQModelLiris(TrainLirisConfig())
    va_only = AutoEQModelLirisVAOnly(TrainLirisConfigVAOnly())
    base_keys = set(base.state_dict().keys())
    va_only_keys = set(va_only.state_dict().keys())

    # va_only keys should be base keys minus mood_head.*
    expected_va_only_keys = {k for k in base_keys if not k.startswith("mood_head.")}
    assert va_only_keys == expected_va_only_keys, (
        f"Mismatch:\n"
        f"  in VA-only but not BASE-minus-mood: {va_only_keys - expected_va_only_keys}\n"
        f"  in BASE-minus-mood but not VA-only: {expected_va_only_keys - va_only_keys}"
    )


# ────────────────────────────────────────────────────────────────────────
# 8. Non-mood components param counts byte-identical to BASE
# ────────────────────────────────────────────────────────────────────────
def test_non_mood_components_param_counts_match():
    base = AutoEQModelLiris(TrainLirisConfig())
    va_only = AutoEQModelLirisVAOnly(TrainLirisConfigVAOnly())
    for block in ("audio_projection", "gate_network", "va_head"):
        b_block = getattr(base, block)
        v_block = getattr(va_only, block)
        b_p = sum(p.numel() for p in b_block.parameters())
        v_p = sum(p.numel() for p in v_block.parameters())
        assert b_p == v_p, f"{block}: BASE {b_p} vs VA-only {v_p}"


# ────────────────────────────────────────────────────────────────────────
# 9. lambda_mood=0 hard-guard — ensure trainer's L_mood contribution is 0
#    Simulate: loss = lambda_mood * L_mood → must be 0
# ────────────────────────────────────────────────────────────────────────
def test_lambda_mood_zero_makes_mood_contribution_zero():
    cfg = TrainLirisConfigVAOnly()
    assert cfg.lambda_mood == 0.0
    # Even with nonzero L_mood, λ * L_mood = 0
    fake_L_mood = torch.tensor(1.9459)  # ≈ ln(7)
    contribution = cfg.lambda_mood * fake_L_mood
    assert contribution.item() == 0.0


# ────────────────────────────────────────────────────────────────────────
# 10. Forward gradient — va_pred has gradient path, mood_logits does NOT
# ────────────────────────────────────────────────────────────────────────
def test_mood_logits_have_no_gradient_path():
    """Dummy mood_logits (torch.zeros) must be disconnected from the
    computational graph — no grad_fn, so it contributes no gradient to any
    trainable parameter. This is the correctness guarantee for Option B'
    (zero-loss auxiliary task without trainer.py modification).
    """
    cfg = TrainLirisConfigVAOnly()
    m = AutoEQModelLirisVAOnly(cfg).train()
    vis = torch.randn(3, cfg.visual_dim)
    aud = torch.randn(3, cfg.audio_raw_dim)
    out = m(vis, aud)
    # Dummy zero tensor → no grad_fn (not part of any computation graph)
    assert out["mood_logits"].grad_fn is None, (
        "mood_logits should have no grad_fn (pure disconnected zeros). "
        "Got grad_fn={}".format(out["mood_logits"].grad_fn)
    )
    # Sanity: va_pred and gate_weights SHOULD have grad_fn (live graph)
    assert out["va_pred"].grad_fn is not None, \
        "va_pred must have grad_fn (connected to trainable VAHead)"
    assert out["gate_weights"].grad_fn is not None, \
        "gate_weights must have grad_fn (connected to trainable GateNetwork)"
