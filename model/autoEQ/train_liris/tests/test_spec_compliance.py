"""V5-FINAL §9-1 + §14-1 compliance tests (9 required).

Run: pytest model/autoEQ/train_liris/tests -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from model.autoEQ.train_liris.config import TrainLirisConfig
from model.autoEQ.train_liris.dataset import (
    MixupTargetShrinkageCollator, official_split,
)
from model.autoEQ.train_liris.losses import combined_loss_liris
from model.autoEQ.train_liris.model import AutoEQModelLiris

REPO = Path(__file__).resolve().parents[4]
META_CSV = REPO / "dataset" / "autoEQ" / "liris" / "liris_metadata.csv"


# ────────────────────────────────────────────────────────────────────────
# 1. Base model (FROZEN 2026-04-21): V5-FINAL §9-1 with engineering fixes.
#    Fields that remain spec-literal are labelled §9-1;
#    fields with an engineering override are labelled BASE.
# ────────────────────────────────────────────────────────────────────────
def test_base_model_config():
    c = TrainLirisConfig()
    # --- §9-1 literal ---
    assert c.num_mood_classes == 7, "BASE 2026-04-21 (Phase 2a-2): K=7 (+0.014 CCC, p<0.05)"
    assert c.lambda_mood == pytest.approx(0.3), "§9-1: λ_mood=0.3"
    assert c.lambda_gate_entropy == pytest.approx(0.05), "§9-1: λ_gate_entropy=0.05"
    assert c.ccc_loss_weight == pytest.approx(0.3), "§9-1: ccc_hybrid_w=0.3"
    assert c.modality_dropout_p == pytest.approx(0.05), "§9-1: modality_dropout_p=0.05"
    assert c.feature_noise_std == pytest.approx(0.03), "§9-1: feature_noise_std=0.03"
    assert c.mixup_prob == pytest.approx(0.5), "§9-1: mixup_prob=0.5"
    assert c.mixup_alpha == pytest.approx(0.4), "§9-1: mixup_alpha=0.4"
    assert c.target_shrinkage_eps == pytest.approx(0.05), "§9-1: ε=0.05"
    assert c.v_var_threshold == pytest.approx(0.117), "§2-2: v p75"
    assert c.a_var_threshold == pytest.approx(0.164), "§2-2: a p75"
    assert c.shrinkage_logic == "AND"
    assert c.batch_size == 32, "§9-1: batch_size=32"
    assert c.lr == pytest.approx(1e-4), "§9-1: lr=1e-4"
    assert c.epochs == 40
    assert c.early_stop_patience == 10
    assert c.grad_clip_norm == pytest.approx(1.0)
    assert c.warmup_steps == 500
    assert c.seed == 42
    assert c.use_official_split is True
    # §9-1: pad_audio_to_10s=auto
    assert c.audio_pad_to_sec == pytest.approx(10.0)
    # §3 line 119: stride=2s
    assert c.audio_stride_sec == pytest.approx(2.0)
    # §6-1: window=4s
    assert c.audio_crop_sec == pytest.approx(4.0)

    # --- BASE 2026-04-21 engineering overrides ---
    assert c.weight_decay == pytest.approx(1e-4), "BASE: wd 1e-5 → 1e-4"
    assert c.head_dropout == pytest.approx(0.3), "BASE: head_dropout 0.0 → 0.3"
    assert c.use_full_learning_set is True, "BASE: use_full_learning_set False → True (LIRIS paper protocol)"
    # Phase 2a-1 winner: A (default), Phase 2a-2 winner: K=7
    assert c.va_norm_strategy == "A", "Phase 2a-1 winner: Strategy A"
    assert c.num_mood_classes == 7, "Phase 2a-2 winner: K=7"


def test_base_model_architecture():
    """BASE 2026-04-21 model structure must contain LayerNorm in 4 places:
    AudioProjection (mid + output) + VAHead hidden + MoodHead hidden.
    """
    import torch.nn as nn
    c = TrainLirisConfig()
    m = AutoEQModelLiris(c)
    # AudioProjection: 2-layer MLP with 2 LayerNorm + GELU
    ap_modules = list(m.audio_projection.net)
    assert len(ap_modules) == 6, f"AudioProjection should be 6-layer Sequential, got {len(ap_modules)}"
    assert isinstance(ap_modules[0], nn.Linear) and ap_modules[0].out_features == 1024
    assert isinstance(ap_modules[1], nn.LayerNorm)
    assert isinstance(ap_modules[2], nn.GELU)
    assert isinstance(ap_modules[3], nn.Dropout)
    assert isinstance(ap_modules[4], nn.Linear) and ap_modules[4].out_features == 512
    assert isinstance(ap_modules[5], nn.LayerNorm)
    # VA head: hidden LayerNorm before ReLU
    va = list(m.va_head.mlp)
    assert isinstance(va[0], nn.Linear)
    assert isinstance(va[1], nn.LayerNorm)
    assert isinstance(va[2], nn.ReLU)
    # Mood head: same pattern
    mh = list(m.mood_head.mlp)
    assert isinstance(mh[1], nn.LayerNorm)
    # Param count sanity (K=7 adds 256*3 + 3 = 771 params over K=4 mood head output)
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 3_400_000 < total < 3_500_000, f"BASE model (K=7) param count should be ~3.42M, got {total}"


# ────────────────────────────────────────────────────────────────────────
# 2. V/A (v_raw−3)/2 round-trip (§2-1)
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not META_CSV.is_file(), reason="metadata not present")
def test_va_norm_roundtrip():
    df = pd.read_csv(META_CSV)
    assert {"v_raw", "a_raw", "v_norm", "a_norm"}.issubset(df.columns)
    vback = df.v_norm * 2 + 3
    aback = df.a_norm * 2 + 3
    assert (vback - df.v_raw).abs().max() < 1e-6
    assert (aback - df.a_raw).abs().max() < 1e-6
    # §2-1 실측 범위
    assert df.v_norm.min() >= -1.0 and df.v_norm.max() <= 1.0
    assert df.a_norm.min() >= -1.0 and df.a_norm.max() <= 1.0


# ────────────────────────────────────────────────────────────────────────
# 3. Official split integrity (§2-4)
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not META_CSV.is_file(), reason="metadata not present")
def test_official_split_integrity():
    splits = official_split(META_CSV)
    assert len(splits["train"]) == 2450
    assert len(splits["val"]) == 2450
    assert len(splits["test"]) == 4900
    assert splits["train"].film_id.nunique() == 40
    assert splits["val"].film_id.nunique() == 40
    assert splits["test"].film_id.nunique() == 80
    # zero film overlap
    tr = set(splits["train"].film_id.unique())
    va = set(splits["val"].film_id.unique())
    te = set(splits["test"].film_id.unique())
    assert tr & va == set()
    assert tr & te == set()
    assert va & te == set()


# ────────────────────────────────────────────────────────────────────────
# 4. Variance threshold fire rate ≈ 6–10 % per §2-2
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not META_CSV.is_file(), reason="metadata not present")
def test_variance_threshold_fire_rate():
    df = pd.read_csv(META_CSV)
    c = TrainLirisConfig()
    v_high = df.v_var > c.v_var_threshold
    a_high = df.a_var > c.a_var_threshold
    rate_and = float((v_high & a_high).mean())
    assert 0.03 < rate_and < 0.15, f"AND fire rate {rate_and:.3f} outside §2-2 range"


# ────────────────────────────────────────────────────────────────────────
# 5. §2-5 K=7 gate expected outcome — JA=0%, strict FAIL → K=4
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not META_CSV.is_file(), reason="metadata not present")
def test_k7_gate_fails_as_expected():
    df = pd.read_csv(META_CSV)
    n = len(df)
    # All 7 classes (missing = 0 count). JA is expected at 0%.
    present = df.mood_k7.value_counts()
    under = 0
    for k in range(7):
        ratio = present.get(k, 0) / n
        if ratio < 0.01:
            under += 1
    assert under >= 1, f"§2-5: expected at least one class < 1% (got {under})"
    # JA is index 3 in shared MOOD_CENTERS (train/dataset.py)
    assert present.get(3, 0) == 0, "§2-5: JoyfulActivation must be 0-count"


# ────────────────────────────────────────────────────────────────────────
# 6. Mood head is K=4 / K=7 switchable (§8 + §21)
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("k", [4, 7])
def test_mood_head_k4_k7_switchable(k: int):
    c = TrainLirisConfig()
    c.num_mood_classes = k
    model = AutoEQModelLiris(c)
    B = 3
    v = torch.randn(B, c.visual_dim)
    a = torch.randn(B, c.audio_raw_dim)
    out = model(v, a)
    assert out["mood_logits"].shape == (B, k)
    assert out["va_pred"].shape == (B, 2)
    assert out["gate_weights"].shape[-1] == 2


# ────────────────────────────────────────────────────────────────────────
# 7. §10-3 Loss terms decompose into independent logs
# ────────────────────────────────────────────────────────────────────────
def test_loss_terms_decompose():
    c = TrainLirisConfig()
    B = 4
    outputs = {
        "va_pred": torch.randn(B, 2, requires_grad=True),
        "mood_logits": torch.randn(B, c.num_mood_classes, requires_grad=True),
        "gate_weights": torch.softmax(torch.randn(B, 2), dim=-1),
    }
    va_target = torch.randn(B, 2)
    mood_target = torch.randint(0, c.num_mood_classes, (B,))
    _, log = combined_loss_liris(outputs, va_target, mood_target, c)
    required = {"loss_total", "loss_va", "loss_va_mse", "loss_va_ccc",
                "loss_mood", "loss_gate_entropy"}
    assert required.issubset(log.keys())
    # L_va_mse + L_va_ccc ≈ L_va (hybrid decomposition invariant)
    assert log["loss_va"] == pytest.approx(
        log["loss_va_mse"] + log["loss_va_ccc"], rel=1e-5
    )


# ────────────────────────────────────────────────────────────────────────
# 8. Overfit monitor threshold (§10-2)
# ────────────────────────────────────────────────────────────────────────
def test_overfit_monitor_threshold_matches_spec():
    c = TrainLirisConfig()
    assert c.overfit_gap_threshold == pytest.approx(0.10)


# ────────────────────────────────────────────────────────────────────────
# 9. Target shrinkage: AND logic shrinks only when v>p75 AND a>p75
# ────────────────────────────────────────────────────────────────────────
def test_target_shrinkage_and_logic():
    c = TrainLirisConfig()
    collator = MixupTargetShrinkageCollator(c, active=True)
    c.mixup_prob = 0.0  # disable mixup for isolation

    def mk(v_var: float, a_var: float, v: float = 0.5, a: float = 0.5) -> dict:
        return {
            "visual": torch.zeros(c.visual_dim),
            "audio": torch.zeros(c.audio_raw_dim),
            "va_target": torch.tensor([v, a]),
            "mood_k4": torch.tensor(0),
            "mood_k7": torch.tensor(0),
            "v_var": torch.tensor(v_var),
            "a_var": torch.tensor(a_var),
            "quadrant": torch.tensor(0),
            "name": "fake.mp4",
        }

    # Case A: both above p75 → should shrink by (1 - ε)
    both_high = mk(c.v_var_threshold + 0.01, c.a_var_threshold + 0.01)
    out = collator([both_high])
    expected = 0.5 * (1.0 - c.target_shrinkage_eps)
    assert out["va_target"][0, 0].item() == pytest.approx(expected, abs=1e-5)

    # Case B: only one above (AND logic) → no shrink
    only_v = mk(c.v_var_threshold + 0.01, c.a_var_threshold - 0.01)
    out = collator([only_v])
    assert out["va_target"][0, 0].item() == pytest.approx(0.5, abs=1e-5)

    # Case C: both below → no shrink
    both_low = mk(c.v_var_threshold - 0.01, c.a_var_threshold - 0.01)
    out = collator([both_low])
    assert out["va_target"][0, 0].item() == pytest.approx(0.5, abs=1e-5)
