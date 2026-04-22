"""Phase 2a-3 AST variant compliance tests.

Independent from train_liris/tests/test_spec_compliance.py — the Base Model
(PANNs) tests must remain PASS after this subpackage is added.

Run: pytest model/autoEQ/train_liris/model_ast/tests -v
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from model.autoEQ.train_liris.config import TrainLirisConfig
from model.autoEQ.train_liris.model import AutoEQModelLiris
from model.autoEQ.train_liris.model_ast.config import TrainLirisConfigAST
from model.autoEQ.train_liris.model_ast.encoders import ASTEncoder
from model.autoEQ.train_liris.model_ast.model import AutoEQModelLirisAST


# ────────────────────────────────────────────────────────────────────────
# 1. Config inheritance — every Base Model hyper is preserved.
# ────────────────────────────────────────────────────────────────────────
def test_ast_config_inherits_base():
    c = TrainLirisConfigAST()
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


# ────────────────────────────────────────────────────────────────────────
# 2. Config — AST-specific overrides present & correct.
# ────────────────────────────────────────────────────────────────────────
def test_ast_config_ast_specific_overrides():
    c = TrainLirisConfigAST()
    assert c.audio_raw_dim == 768, "AST CLS embedding dim"
    assert c.ast_model_name.startswith("MIT/ast-finetuned-audioset")
    assert c.audio_sample_rate_hz == 16000, "AST requires 16 kHz"
    assert "liris_ast_v5spec" in c.feature_file
    assert c.run_name == "2a3_ast"


# ────────────────────────────────────────────────────────────────────────
# 3. AudioProjection auto-configures for 768-dim AST input.
# ────────────────────────────────────────────────────────────────────────
def test_ast_model_audio_projection_shape():
    import torch.nn as nn
    c = TrainLirisConfigAST()
    m = AutoEQModelLirisAST(c)
    ap = list(m.audio_projection.net)
    # Same 6-layer Sequential as Base Model, only dims change.
    assert len(ap) == 6
    assert isinstance(ap[0], nn.Linear)
    assert ap[0].in_features == 768
    assert ap[0].out_features == 768 // 2          # mid = 384
    assert isinstance(ap[1], nn.LayerNorm)
    assert isinstance(ap[2], nn.GELU)
    assert isinstance(ap[3], nn.Dropout)
    assert isinstance(ap[4], nn.Linear)
    assert ap[4].in_features == 384
    assert ap[4].out_features == 512               # audio_proj_dim
    assert isinstance(ap[5], nn.LayerNorm)

    x = torch.randn(4, 768)
    y = m.audio_projection(x)
    assert y.shape == (4, 512)


# ────────────────────────────────────────────────────────────────────────
# 4. Full forward shape — gate / va / mood heads unaffected by encoder swap.
# ────────────────────────────────────────────────────────────────────────
def test_ast_model_full_forward_shape():
    c = TrainLirisConfigAST()
    m = AutoEQModelLirisAST(c).eval()
    visual = torch.randn(3, c.visual_dim)
    audio = torch.randn(3, c.audio_raw_dim)
    out = m(visual, audio)
    assert out["va_pred"].shape == (3, 2)
    assert out["mood_logits"].shape == (3, 7)       # K=7 inherited
    assert out["gate_weights"].shape == (3, 2)


# ────────────────────────────────────────────────────────────────────────
# 5. Param count — AST variant should be strictly smaller than BASE
#    (AudioProjection: 2,626,048 → ~492,032, i.e. −81%).
# ────────────────────────────────────────────────────────────────────────
def test_ast_model_parameter_count_decreases():
    base = AutoEQModelLiris(TrainLirisConfig())
    ast = AutoEQModelLirisAST(TrainLirisConfigAST())
    base_p = sum(p.numel() for p in base.parameters())
    ast_p = sum(p.numel() for p in ast.parameters())
    assert ast_p < base_p, f"AST ({ast_p}) should be smaller than BASE ({base_p})"
    # Sanity: AudioProjection reduction of ~2.1M params → total delta ≈ 2.1M.
    assert base_p - ast_p > 1_500_000


# ────────────────────────────────────────────────────────────────────────
# 6. Dataset — "ast" key must be present in expected feature dict.
# ────────────────────────────────────────────────────────────────────────
def test_ast_dataset_audio_key():
    """PrecomputedLirisDatasetAST must read feat['ast'] (smoke — uses synthetic data)."""
    import pandas as pd
    from model.autoEQ.train_liris.model_ast.dataset import PrecomputedLirisDatasetAST

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
            "xclip": torch.randn(512),
            "ast": torch.randn(768),
        }
    }
    ds = PrecomputedLirisDatasetAST(meta, features)
    item = ds[0]
    assert item["audio"].shape == (768,)
    assert item["visual"].shape == (512,)
    assert item["va_target"].shape == (2,)


# ────────────────────────────────────────────────────────────────────────
# 7. (skip if transformers not installed) ASTEncoder import + construction smoke.
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed",
)
def test_ast_encoder_class_interface():
    """Class is importable and declares expected attributes (without network)."""
    # Don't actually instantiate (requires model download) — just verify
    # class surface so unit-test layer stays offline-safe.
    assert hasattr(ASTEncoder, "forward")
    assert hasattr(ASTEncoder, "__init__")
