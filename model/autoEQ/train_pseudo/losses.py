import torch
import torch.nn.functional as F
from torch import Tensor

from ..train.utils import compute_ccc  # reuse from train/
from .config import TrainCogConfig


def va_mse_loss(va_pred: Tensor, va_target: Tensor) -> Tensor:
    return F.mse_loss(va_pred, va_target)


def va_hybrid_loss(
    va_pred: Tensor, va_target: Tensor, ccc_weight: float = 0.3
) -> tuple[Tensor, float, float]:
    """L = (1 - ccc_weight) * MSE + ccc_weight * (1 - mean_CCC).

    Returns (loss, mse_value, ccc_value) for logging.
    """
    mse = F.mse_loss(va_pred, va_target)
    ccc_v = compute_ccc(va_pred[:, 0], va_target[:, 0])
    ccc_a = compute_ccc(va_pred[:, 1], va_target[:, 1])
    mean_ccc = (ccc_v + ccc_a) / 2.0
    loss = (1.0 - ccc_weight) * mse + ccc_weight * (1.0 - mean_ccc)
    return loss, mse.item(), mean_ccc.item()


def mood_ce_loss(mood_logits: Tensor, mood_target: Tensor) -> Tensor:
    return F.cross_entropy(mood_logits, mood_target)


def gate_entropy_loss(gate_weights: Tensor) -> Tensor:
    """Mean of sum(w * log(w)). Always <= 0; adding lambda * loss rewards
    higher entropy (more uniform weights).
    """
    eps = 1e-8
    entropy = (gate_weights * torch.log(gate_weights + eps)).sum(dim=-1)
    return entropy.mean()


def combined_loss_cog(
    outputs: dict[str, Tensor],
    va_target: Tensor,
    mood_target: Tensor,
    config: TrainCogConfig,
) -> tuple[Tensor, dict[str, float]]:
    """L = lambda_va * L_va + lambda_mood * L_mood + lambda_gate * L_gate.

    No cong term. Signature drops cong_target compared to train.combined_loss.
    """
    if config.use_ccc_loss:
        l_va, mse_val, ccc_val = va_hybrid_loss(
            outputs["va_pred"], va_target, config.ccc_loss_weight
        )
    else:
        l_va = va_mse_loss(outputs["va_pred"], va_target)
        mse_val = None
        ccc_val = None

    l_mood = mood_ce_loss(outputs["mood_logits"], mood_target)
    l_gate = gate_entropy_loss(outputs["gate_weights"])

    total = (
        config.lambda_va * l_va
        + config.lambda_mood * l_mood
        + config.lambda_gate_entropy * l_gate
    )

    loss_dict: dict[str, float] = {
        "va": l_va.item(),
        "mood": l_mood.item(),
        "gate_entropy": l_gate.item(),
        "total": total.item(),
    }
    if config.use_ccc_loss:
        loss_dict["va_mse"] = mse_val  # type: ignore[assignment]
        loss_dict["va_ccc"] = ccc_val  # type: ignore[assignment]
    return total, loss_dict
