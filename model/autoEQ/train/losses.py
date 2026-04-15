import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TrainConfig
from .utils import compute_ccc


def va_mse_loss(va_pred: Tensor, va_target: Tensor) -> Tensor:
    """MSE loss for V/A regression. Both tensors: (B, 2)."""
    return F.mse_loss(va_pred, va_target)


def va_hybrid_loss(
    va_pred: Tensor, va_target: Tensor, ccc_weight: float = 0.3
) -> tuple[Tensor, float, float]:
    """Hybrid MSE + CCC loss for V/A regression.

    L = (1 - ccc_weight) * MSE + ccc_weight * (1 - mean_CCC)

    Returns (loss, mse_value, ccc_value) for logging.
    """
    mse = F.mse_loss(va_pred, va_target)
    ccc_v = compute_ccc(va_pred[:, 0], va_target[:, 0])
    ccc_a = compute_ccc(va_pred[:, 1], va_target[:, 1])
    mean_ccc = (ccc_v + ccc_a) / 2.0
    loss = (1.0 - ccc_weight) * mse + ccc_weight * (1.0 - mean_ccc)
    return loss, mse.item(), mean_ccc.item()


def mood_ce_loss(mood_logits: Tensor, mood_target: Tensor) -> Tensor:
    """CrossEntropy loss for mood classification.

    mood_logits: (B, 7), mood_target: (B,) long.
    """
    return F.cross_entropy(mood_logits, mood_target)


def cong_ce_loss(cong_logits: Tensor, cong_target: Tensor) -> Tensor:
    """CrossEntropy loss for congruence classification.

    cong_logits: (B, 3), cong_target: (B,) long.
    """
    return F.cross_entropy(cong_logits, cong_target)


def gate_entropy_loss(gate_weights: Tensor) -> Tensor:
    """Gate entropy loss for degeneracy prevention.

    Encourages uniform gate weights by maximizing entropy.
    Returns mean(sum(w * log(w + eps))), which is always <= 0.
    Adding lambda * gate_entropy_loss to total loss rewards high entropy
    (more negative = lower total loss when entropy is high).

    gate_weights: (B, 2) with values in (0, 1), summing to 1 per row.
    """
    eps = 1e-8
    entropy = (gate_weights * torch.log(gate_weights + eps)).sum(dim=-1)  # (B,)
    return entropy.mean()  # scalar <= 0


def combined_loss(
    outputs: dict[str, Tensor],
    va_target: Tensor,
    mood_target: Tensor,
    cong_target: Tensor,
    config: TrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Compute weighted sum of all losses.

    Returns:
        (total_loss, loss_dict) where loss_dict contains individual loss values.
    """
    if config.use_ccc_loss:
        l_va, mse_val, ccc_val = va_hybrid_loss(
            outputs["va_pred"], va_target, config.ccc_loss_weight
        )
    else:
        l_va = va_mse_loss(outputs["va_pred"], va_target)
    l_mood = mood_ce_loss(outputs["mood_logits"], mood_target)
    l_cong = cong_ce_loss(outputs["cong_logits"], cong_target)
    l_gate = gate_entropy_loss(outputs["gate_weights"])

    total = (
        config.lambda_va * l_va
        + config.lambda_mood * l_mood
        + config.lambda_cong * l_cong
        + config.lambda_gate_entropy * l_gate
    )

    loss_dict = {
        "va": l_va.item(),
        "mood": l_mood.item(),
        "cong": l_cong.item(),
        "gate_entropy": l_gate.item(),
        "total": total.item(),
    }
    if config.use_ccc_loss:
        loss_dict["va_mse"] = mse_val
        loss_dict["va_ccc"] = ccc_val

    return total, loss_dict
