"""combined_loss_liris — per-term independent logging (V5-FINAL §10-3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..train.utils import compute_ccc
from .config import TrainLirisConfig


def va_hybrid_loss_terms(
    va_pred: Tensor, va_target: Tensor, ccc_w: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (L_va, mse_term, ccc_term) as **loss contributions**.

        mse_term = (1 - ccc_w) * mse
        ccc_term = ccc_w * (1 - mean_CCC)
        L_va     = mse_term + ccc_term

    Each term is differentiable (no .item()).
    """
    mse = F.mse_loss(va_pred, va_target)
    ccc_v = compute_ccc(va_pred[:, 0], va_target[:, 0])
    ccc_a = compute_ccc(va_pred[:, 1], va_target[:, 1])
    mean_ccc = (ccc_v + ccc_a) / 2.0
    mse_term = (1.0 - ccc_w) * mse
    ccc_term = ccc_w * (1.0 - mean_ccc)
    return mse_term + ccc_term, mse_term, ccc_term


def combined_loss_liris(
    outputs: dict[str, Tensor],
    va_target: Tensor,
    mood_target: Tensor,
    cfg: TrainLirisConfig,
) -> tuple[Tensor, dict[str, float]]:
    """L_total = λ_va·L_va + λ_mood·L_mood + λ_gate·L_gate.

    Returns (loss_tensor, per-term scalar dict) for independent logging.
    """
    if cfg.use_ccc_loss:
        l_va, mse_term, ccc_term = va_hybrid_loss_terms(
            outputs["va_pred"], va_target, cfg.ccc_loss_weight
        )
    else:
        l_va = F.mse_loss(outputs["va_pred"], va_target)
        mse_term = l_va.detach()
        ccc_term = torch.zeros_like(l_va)

    l_mood = F.cross_entropy(outputs["mood_logits"], mood_target)

    gate = outputs["gate_weights"]
    gate_entropy = (gate * torch.log(gate + 1e-8)).sum(dim=-1).mean()
    # Note: gate_entropy is <=0; adding λ·gate_entropy pulls gate toward uniform
    # when λ>0 (entropy becomes less negative). Matches train_pseudo convention.

    total = (
        cfg.lambda_va * l_va
        + cfg.lambda_mood * l_mood
        + cfg.lambda_gate_entropy * gate_entropy
    )

    return total, {
        "loss_total": total.item(),
        "loss_va": l_va.item(),
        "loss_va_mse": mse_term.item(),
        "loss_va_ccc": ccc_term.item(),
        "loss_mood": l_mood.item(),
        "loss_gate_entropy": gate_entropy.item(),
    }
