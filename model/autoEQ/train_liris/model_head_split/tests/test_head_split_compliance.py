"""Phase 2a-6 Head-Split compliance tests.

Independent from parent train_liris/tests and sibling model_ast / model_clipmean /
model_fusion tests. All previous test suites must remain PASS after adding
this subpackage.

Run: pytest model/autoEQ/train_liris/model_head_split/tests -v

Verification goals:
    1. Config inheritance chain intact
    2. SeparateVAHead architecture correct (two independent paths)
    3. Parameter accounting matches sizing table (BASE_MODEL.md-style)
    4. V path and A path are truly independent (perturbation invariance)
    5. Each fusion mode instantiates correctly with SeparateVAHead
    6. Output interface byte-compatible with downstream (loss, metrics)
    7. Non-va_head components byte-identical to BASE / Phase 2a-5
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.autoEQ.train_liris.config import TrainLirisConfig
from model.autoEQ.train_liris.model import AutoEQModelLiris
from model.autoEQ.train_liris.model_fusion.model import AutoEQModelLirisFusion
from model.autoEQ.train_liris.model_fusion.config import TrainLirisConfigFusion
from model.autoEQ.train_liris.model_head_split.config import TrainLirisConfigHeadSplit
from model.autoEQ.train_liris.model_head_split.heads import SeparateVAHead
from model.autoEQ.train_liris.model_head_split.model import AutoEQModelLirisHeadSplit


# ────────────────────────────────────────────────────────────────────────
# 1. Config inheritance chain — TrainLirisConfig → Fusion → HeadSplit
# ────────────────────────────────────────────────────────────────────────
def test_config_inherits_full_chain():
    c = TrainLirisConfigHeadSplit()
    # From TrainLirisConfig (BASE)
    assert c.num_mood_classes == 7
    assert c.head_dropout == pytest.approx(0.3)
    assert c.weight_decay == pytest.approx(1e-4)
    assert c.use_full_learning_set is True
    assert c.va_norm_strategy == "A"
    assert c.visual_dim == 512
    assert c.audio_raw_dim == 2048
    assert c.audio_proj_dim == 512
    assert c.fused_dim == 1024
    assert c.head_hidden_dim == 256
    assert c.batch_size == 32
    assert c.lr == pytest.approx(1e-4)
    assert c.epochs == 40
    assert c.lambda_va == pytest.approx(1.0)
    assert c.lambda_mood == pytest.approx(0.3)
    # From TrainLirisConfigFusion (Phase 2a-5)
    assert c.fusion_mode == "gate"  # default
    # Own override
    assert c.run_name == "2a6_head_split"


# ────────────────────────────────────────────────────────────────────────
# 2. Invalid fusion_mode → ValueError at instantiation
# ────────────────────────────────────────────────────────────────────────
def test_invalid_fusion_mode_raises():
    cfg = TrainLirisConfigHeadSplit()
    cfg.fusion_mode = "bogus"
    with pytest.raises(ValueError, match="fusion_mode"):
        AutoEQModelLirisHeadSplit(cfg)


# ────────────────────────────────────────────────────────────────────────
# 3. SeparateVAHead architecture — two independent 5-layer Sequential stacks
# ────────────────────────────────────────────────────────────────────────
def test_separate_va_head_architecture():
    cfg = TrainLirisConfig()
    head = SeparateVAHead(cfg)

    # Must have exactly two paths named v_head and a_head
    assert hasattr(head, "v_head")
    assert hasattr(head, "a_head")
    assert isinstance(head.v_head, nn.Sequential)
    assert isinstance(head.a_head, nn.Sequential)

    # Each path: Linear → LN → ReLU → Dropout → Linear (5 layers)
    for path_name, path in (("v_head", head.v_head), ("a_head", head.a_head)):
        assert len(path) == 5, f"{path_name} has {len(path)} layers, expected 5"
        assert isinstance(path[0], nn.Linear), f"{path_name}[0] not Linear"
        assert path[0].in_features == cfg.fused_dim
        assert path[0].out_features == cfg.head_hidden_dim
        assert isinstance(path[1], nn.LayerNorm)
        assert path[1].normalized_shape == (cfg.head_hidden_dim,)
        assert isinstance(path[2], nn.ReLU)
        assert isinstance(path[3], nn.Dropout)
        assert path[3].p == pytest.approx(cfg.head_dropout)
        assert isinstance(path[4], nn.Linear)
        assert path[4].in_features == cfg.head_hidden_dim
        assert path[4].out_features == 1


# ────────────────────────────────────────────────────────────────────────
# 4. SeparateVAHead parameter count — exact arithmetic
# ────────────────────────────────────────────────────────────────────────
def test_separate_va_head_param_count():
    cfg = TrainLirisConfig()
    head = SeparateVAHead(cfg)
    actual = sum(p.numel() for p in head.parameters())
    # per head:
    #   Linear(1024→256).weight = 1024 * 256 = 262,144
    #   Linear(1024→256).bias   = 256
    #   LayerNorm(256).weight   = 256
    #   LayerNorm(256).bias     = 256
    #   Linear(256→1).weight    = 256
    #   Linear(256→1).bias      = 1
    #   per-head subtotal       = 263,169
    # total: 2 × 263,169 = 526,338
    expected_per_head = (1024 * 256 + 256) + (256 + 256) + (256 + 1)
    expected_total = 2 * expected_per_head
    assert expected_per_head == 263_169
    assert expected_total == 526_338
    assert actual == expected_total, (
        f"SeparateVAHead params {actual} != expected {expected_total}"
    )


# ────────────────────────────────────────────────────────────────────────
# 5. SeparateVAHead forward shape (B, 2) — matches BASE VAHead interface
# ────────────────────────────────────────────────────────────────────────
def test_separate_va_head_forward_shape():
    cfg = TrainLirisConfig()
    head = SeparateVAHead(cfg).eval()
    fused = torch.randn(7, cfg.fused_dim)
    with torch.no_grad():
        out = head(fused)
    assert out.shape == (7, 2)
    assert out.dtype == torch.float32


# ────────────────────────────────────────────────────────────────────────
# 6. V path and A path are INDEPENDENT — perturbing one does not affect the other
# ────────────────────────────────────────────────────────────────────────
def test_v_and_a_paths_independent():
    torch.manual_seed(42)
    cfg = TrainLirisConfig()
    head = SeparateVAHead(cfg).eval()
    fused = torch.randn(4, cfg.fused_dim)

    with torch.no_grad():
        out_baseline = head(fused).clone()  # (4, 2)

        # Perturb ONLY v_head weights
        head.v_head[0].weight.data += 1.0
        out_after_v_perturb = head(fused).clone()

        # Reset v_head, perturb ONLY a_head
        head.v_head[0].weight.data -= 1.0
        head.a_head[0].weight.data += 1.0
        out_after_a_perturb = head(fused).clone()

    # V-perturbation: V column changes, A column unchanged
    assert not torch.allclose(out_baseline[:, 0], out_after_v_perturb[:, 0]), \
        "V-perturbation should change V column"
    assert torch.allclose(out_baseline[:, 1], out_after_v_perturb[:, 1]), \
        "V-perturbation must NOT change A column (paths should be independent)"

    # A-perturbation: A column changes, V column unchanged
    assert torch.allclose(out_baseline[:, 0], out_after_a_perturb[:, 0]), \
        "A-perturbation must NOT change V column (paths should be independent)"
    assert not torch.allclose(out_baseline[:, 1], out_after_a_perturb[:, 1]), \
        "A-perturbation should change A column"


# ────────────────────────────────────────────────────────────────────────
# 7. HeadSplit model — total param counts per fusion mode
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode,expected_total", [
    # BASE breakdown: AudioProj 2,626,048 + VAHead_joint 263,426 + MoodHead 264,711 = 3,154,185
    # HeadSplit VAHead: 526,338 (vs 263,426 joint)
    # + gate fusion   (262,914): 2,626,048 + 262,914 + 526,338 + 264,711 = 3,680,011
    ("gate", 3_680_011),
    # + concat fusion (      0): 2,626,048 +       0 + 526,338 + 264,711 = 3,417,097
    ("concat", 3_417_097),
    # + gmu fusion  (2,100,224): 2,626,048 + 2,100,224 + 526,338 + 264,711 = 5,517,321
    ("gmu", 5_517_321),
    # + gmu_notanh  (2,100,224): same as gmu
    ("gmu_notanh", 5_517_321),
])
def test_model_param_counts_per_fusion_mode(mode: str, expected_total: int):
    cfg = TrainLirisConfigHeadSplit(fusion_mode=mode)
    m = AutoEQModelLirisHeadSplit(cfg)
    actual = sum(p.numel() for p in m.parameters())
    assert actual == expected_total, (
        f"fusion_mode={mode}: got {actual:,}, expected {expected_total:,}"
    )


# ────────────────────────────────────────────────────────────────────────
# 8. HeadSplit adds exactly +262,912 params vs Phase 2a-5 joint-head counterpart
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["gate", "concat", "gmu", "gmu_notanh"])
def test_headsplit_delta_vs_joint(mode: str):
    joint = AutoEQModelLirisFusion(TrainLirisConfigFusion(fusion_mode=mode))
    split = AutoEQModelLirisHeadSplit(TrainLirisConfigHeadSplit(fusion_mode=mode))
    joint_p = sum(p.numel() for p in joint.parameters())
    split_p = sum(p.numel() for p in split.parameters())
    delta = split_p - joint_p
    # SeparateVAHead (526,338) − joint VAHead (263,426) = +262,912
    assert delta == 262_912, (
        f"fusion_mode={mode}: delta {delta:,} != expected +262,912"
    )


# ────────────────────────────────────────────────────────────────────────
# 9. Forward output shapes for all fusion modes — downstream compatibility
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["gate", "concat", "gmu", "gmu_notanh"])
def test_forward_output_shapes(mode: str):
    cfg = TrainLirisConfigHeadSplit(fusion_mode=mode)
    m = AutoEQModelLirisHeadSplit(cfg).eval()
    v = torch.randn(3, cfg.visual_dim)
    a = torch.randn(3, cfg.audio_raw_dim)
    with torch.no_grad():
        out = m(v, a)
    assert out["va_pred"].shape == (3, 2)
    assert out["mood_logits"].shape == (3, 7)
    assert out["gate_weights"].shape == (3, 2)
    # dtype sanity
    assert out["va_pred"].dtype == torch.float32
    assert out["mood_logits"].dtype == torch.float32
    assert out["gate_weights"].dtype == torch.float32


# ────────────────────────────────────────────────────────────────────────
# 10. State dict: va_head keys differ (intended), others match BASE byte-identical
# ────────────────────────────────────────────────────────────────────────
def test_state_dict_structure():
    base = AutoEQModelLiris(TrainLirisConfig())
    split = AutoEQModelLirisHeadSplit(TrainLirisConfigHeadSplit(fusion_mode="gate"))

    base_keys = set(base.state_dict().keys())
    split_keys = set(split.state_dict().keys())

    # All non-va_head keys match exactly
    base_non_va = {k for k in base_keys if not k.startswith("va_head.")}
    split_non_va = {k for k in split_keys if not k.startswith("va_head.")}
    assert base_non_va == split_non_va, (
        "non-va_head keys should match BASE byte-identical\n"
        f"in BASE but not HeadSplit: {base_non_va - split_non_va}\n"
        f"in HeadSplit but not BASE: {split_non_va - base_non_va}"
    )

    # va_head keys MUST differ (intentional structural change)
    base_va = {k for k in base_keys if k.startswith("va_head.")}
    split_va = {k for k in split_keys if k.startswith("va_head.")}
    assert base_va != split_va, "va_head structure should differ"
    # split should have v_head.* and a_head.* subtrees
    assert any(k.startswith("va_head.v_head.") for k in split_va), \
        "split va_head must contain v_head subtree"
    assert any(k.startswith("va_head.a_head.") for k in split_va), \
        "split va_head must contain a_head subtree"


# ────────────────────────────────────────────────────────────────────────
# 11. Non-va_head components byte-identical to BASE (param count match per block)
# ────────────────────────────────────────────────────────────────────────
def test_non_va_head_components_identical():
    base = AutoEQModelLiris(TrainLirisConfig())
    split = AutoEQModelLirisHeadSplit(TrainLirisConfigHeadSplit(fusion_mode="gate"))

    for block in ("audio_projection", "gate_network", "mood_head"):
        base_block = getattr(base, block, None)
        split_block = getattr(split, block, None)
        assert base_block is not None and split_block is not None, \
            f"{block} missing on one of BASE/HeadSplit"
        base_p = sum(p.numel() for p in base_block.parameters())
        split_p = sum(p.numel() for p in split_block.parameters())
        assert base_p == split_p, f"{block}: BASE {base_p} vs HeadSplit {split_p}"


# ────────────────────────────────────────────────────────────────────────
# 12. Forward VA output column 0 corresponds to V (matches dataset va_target order)
# ────────────────────────────────────────────────────────────────────────
def test_va_pred_column_order():
    """va_target is [v_norm, a_norm] in dataset.py — SeparateVAHead output must
    match this ordering: column 0 = V (from v_head), column 1 = A (from a_head)."""
    torch.manual_seed(0)
    cfg = TrainLirisConfig()
    head = SeparateVAHead(cfg).eval()
    # Force v_head to produce a constant unique value, a_head a different one.
    with torch.no_grad():
        for p in head.v_head.parameters():
            p.data.zero_()
        for p in head.a_head.parameters():
            p.data.zero_()
        head.v_head[-1].bias.data.fill_(0.111)  # V prediction = 0.111
        head.a_head[-1].bias.data.fill_(0.999)  # A prediction = 0.999
        fused = torch.randn(2, cfg.fused_dim)
        out = head(fused)
    assert torch.allclose(out[:, 0], torch.full((2,), 0.111)), \
        f"column 0 should be V (from v_head), got {out[:, 0]}"
    assert torch.allclose(out[:, 1], torch.full((2,), 0.999)), \
        f"column 1 should be A (from a_head), got {out[:, 1]}"


# ────────────────────────────────────────────────────────────────────────
# 13. Gate-mode HeadSplit forward differs from BASE (confirms HeadSplit has effect)
# ────────────────────────────────────────────────────────────────────────
def test_headsplit_gate_differs_from_base_under_random_init():
    """With random init, HeadSplit and BASE produce different outputs (different arch)."""
    torch.manual_seed(42)
    base = AutoEQModelLiris(TrainLirisConfig()).eval()
    torch.manual_seed(42)
    split = AutoEQModelLirisHeadSplit(
        TrainLirisConfigHeadSplit(fusion_mode="gate")).eval()
    v = torch.randn(3, 512)
    a = torch.randn(3, 2048)
    with torch.no_grad():
        o_base = base(v, a)
        o_split = split(v, a)
    # va_pred shapes match but values differ (different VAHead architecture)
    assert o_base["va_pred"].shape == o_split["va_pred"].shape
    assert not torch.allclose(o_base["va_pred"], o_split["va_pred"]), \
        "HeadSplit should produce different va_pred from BASE under same init"
