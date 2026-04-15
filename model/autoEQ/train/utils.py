import torch
from torch import Tensor, nn


def compute_ccc(pred: Tensor, target: Tensor) -> Tensor:
    """Concordance Correlation Coefficient (CCC).

    CCC = 2 * rho * sigma_x * sigma_y
          / (sigma_x^2 + sigma_y^2 + (mu_x - mu_y)^2)

    Returns scalar tensor in [-1, 1].
    """
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var(correction=0)
    target_var = target.var(correction=0)
    covar = ((pred - pred_mean) * (target - target_mean)).mean()

    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    ccc = 2.0 * covar / (denominator + 1e-8)
    return ccc


def compute_mean_ccc(
    va_pred: Tensor, va_target: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute CCC for valence and arousal, return (mean_ccc, ccc_v, ccc_a)."""
    ccc_v = compute_ccc(va_pred[:, 0], va_target[:, 0])
    ccc_a = compute_ccc(va_pred[:, 1], va_target[:, 1])
    mean_ccc = (ccc_v + ccc_a) / 2.0
    return mean_ccc, ccc_v, ccc_a


def compute_head_grad_norms(
    heads: dict[str, nn.Module],
) -> dict[str, float]:
    """Compute L2 gradient norm for each head's parameters.

    Args:
        heads: dict mapping head name to nn.Module, e.g.
               {'va': va_head, 'mood': mood_head, 'cong': cong_head}

    Returns:
        dict mapping head name to gradient L2 norm.
    """
    grad_norms = {}
    for name, head in heads.items():
        total_norm = 0.0
        for p in head.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms[name] = total_norm**0.5
    return grad_norms


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
