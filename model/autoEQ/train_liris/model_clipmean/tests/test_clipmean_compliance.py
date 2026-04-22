"""Phase 2a-4 CLIP frame-mean variant compliance tests.

Independent from train_liris/tests/test_spec_compliance.py and
model_ast/tests/test_ast_compliance.py — the Base Model (PANNs/X-CLIP)
and AST variant tests must both remain PASS after this subpackage is added.

Run: pytest model/autoEQ/train_liris/model_clipmean/tests -v
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from model.autoEQ.train_liris.config import TrainLirisConfig
from model.autoEQ.train_liris.model import AutoEQModelLiris
from model.autoEQ.train_liris.model_clipmean.config import TrainLirisConfigCLIPMean
from model.autoEQ.train_liris.model_clipmean.encoders import CLIPFrameMeanEncoder
from model.autoEQ.train_liris.model_clipmean.model import AutoEQModelLirisCLIPMean


# ────────────────────────────────────────────────────────────────────────
# 1. Config inheritance — every Base Model hyper is preserved.
# ────────────────────────────────────────────────────────────────────────
def test_clipmean_config_inherits_base():
    c = TrainLirisConfigCLIPMean()
    # BASE 2026-04-21 (Phase 2a-2 winner) — these must match TrainLirisConfig.
    assert c.num_mood_classes == 7
    assert c.head_dropout == pytest.approx(0.3)
    assert c.weight_decay == pytest.approx(1e-4)
    assert c.use_full_learning_set is True
    assert c.va_norm_strategy == "A"
    assert c.lambda_mood == pytest.approx(0.3)
    assert c.lambda_gate_entropy == pytest.approx(0.05)
    assert c.ccc_loss_weight == pytest.approx(0.3)
    assert c.modality_dropout_p == pytest.approx(0.05)
    assert c.feature_noise_std == pytest.approx(0.03)
    assert c.mixup_prob == pytest.approx(0.5)
    assert c.mixup_alpha == pytest.approx(0.4)
    assert c.target_shrinkage_eps == pytest.approx(0.05)
    assert c.v_var_threshold == pytest.approx(0.117)
    assert c.a_var_threshold == pytest.approx(0.164)
    assert c.shrinkage_logic == "AND"
    assert c.batch_size == 32
    assert c.lr == pytest.approx(1e-4)
    assert c.epochs == 40
    assert c.early_stop_patience == 10
    assert c.grad_clip_norm == pytest.approx(1.0)
    assert c.warmup_steps == 500
    assert c.seed == 42
    assert c.audio_crop_sec == pytest.approx(4.0)
    assert c.audio_stride_sec == pytest.approx(2.0)
    assert c.audio_pad_to_sec == pytest.approx(10.0)
    # Audio side UNCHANGED — PANNs 2048-d reused from BASE cache.
    assert c.audio_raw_dim == 2048
    # Visual dim UNCHANGED — CLIP ViT-B/32 projection dim == X-CLIP video dim.
    assert c.visual_dim == 512


# ────────────────────────────────────────────────────────────────────────
# 2. Config — CLIPMean-specific overrides present & correct.
# ────────────────────────────────────────────────────────────────────────
def test_clipmean_config_specific_overrides():
    c = TrainLirisConfigCLIPMean()
    assert c.clip_model_name == "openai/clip-vit-base-patch32"
    assert "liris_clipmean_v5spec" in c.feature_file
    assert c.run_name == "2a4_clipmean"


# ────────────────────────────────────────────────────────────────────────
# 3. AudioProjection UNCHANGED from BASE (only visual encoder differs).
# ────────────────────────────────────────────────────────────────────────
def test_clipmean_model_audio_projection_unchanged():
    import torch.nn as nn
    c = TrainLirisConfigCLIPMean()
    m = AutoEQModelLirisCLIPMean(c)
    ap = list(m.audio_projection.net)
    # Same 6-layer Sequential as BASE, same dims (2048 → 1024 → 512).
    assert len(ap) == 6
    assert isinstance(ap[0], nn.Linear)
    assert ap[0].in_features == 2048
    assert ap[0].out_features == 1024
    assert isinstance(ap[1], nn.LayerNorm)
    assert isinstance(ap[2], nn.GELU)
    assert isinstance(ap[3], nn.Dropout)
    assert isinstance(ap[4], nn.Linear)
    assert ap[4].in_features == 1024
    assert ap[4].out_features == 512
    assert isinstance(ap[5], nn.LayerNorm)


# ────────────────────────────────────────────────────────────────────────
# 4. Full forward shape — heads unaffected by visual-encoder swap.
# ────────────────────────────────────────────────────────────────────────
def test_clipmean_model_full_forward_shape():
    c = TrainLirisConfigCLIPMean()
    m = AutoEQModelLirisCLIPMean(c).eval()
    visual = torch.randn(3, c.visual_dim)
    audio = torch.randn(3, c.audio_raw_dim)
    out = m(visual, audio)
    assert out["va_pred"].shape == (3, 2)
    assert out["mood_logits"].shape == (3, 7)       # K=7 inherited
    assert out["gate_weights"].shape == (3, 2)


# ────────────────────────────────────────────────────────────────────────
# 5. Param count — CLIPMean variant must be BYTE-IDENTICAL to BASE.
#    (unlike AST which changed AudioProjection dims; here both visual_dim
#     and audio_raw_dim match BASE → identical trainable-param count.)
# ────────────────────────────────────────────────────────────────────────
def test_clipmean_model_parameter_count_identical_to_base():
    base = AutoEQModelLiris(TrainLirisConfig())
    cm = AutoEQModelLirisCLIPMean(TrainLirisConfigCLIPMean())
    base_p = sum(p.numel() for p in base.parameters())
    cm_p = sum(p.numel() for p in cm.parameters())
    assert cm_p == base_p, (
        f"CLIPMean ({cm_p}) should be IDENTICAL to BASE ({base_p}) — OAT "
        f"keeps architecture fixed, only upstream visual representation differs"
    )
    # Phase 2a-2 BASE fingerprint (see BASE_MODEL.md §2).
    assert base_p == 3_417_099


# ────────────────────────────────────────────────────────────────────────
# 6. Dataset — "clipmean" key must be present in expected feature dict.
# ────────────────────────────────────────────────────────────────────────
def test_clipmean_dataset_visual_key():
    """PrecomputedLirisDatasetCLIPMean must read feat['clipmean'] (smoke — uses synthetic data)."""
    import pandas as pd
    from model.autoEQ.train_liris.model_clipmean.dataset import PrecomputedLirisDatasetCLIPMean

    meta = pd.DataFrame(
        [
            {
                "name": "fake.mp4",
                "v_norm": 0.1, "a_norm": -0.2,
                "v_var": 0.05, "a_var": 0.05,
                "mood_k7": 2, "quadrant_k4": 1,
            }
        ]
    )
    features = {
        "fake.mp4": {
            "clipmean": torch.randn(512),
            "panns": torch.randn(2048),
        }
    }
    ds = PrecomputedLirisDatasetCLIPMean(meta, features)
    item = ds[0]
    assert item["visual"].shape == (512,)
    assert item["audio"].shape == (2048,)
    assert item["va_target"].shape == (2,)


# ────────────────────────────────────────────────────────────────────────
# 7. (skip if transformers not installed) CLIPFrameMeanEncoder import + class surface smoke.
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed",
)
def test_clipmean_encoder_class_interface():
    """Class is importable and declares expected attributes (without network)."""
    # Don't actually instantiate (requires model download) — just verify
    # class surface so unit-test layer stays offline-safe.
    assert hasattr(CLIPFrameMeanEncoder, "forward")
    assert hasattr(CLIPFrameMeanEncoder, "__init__")
