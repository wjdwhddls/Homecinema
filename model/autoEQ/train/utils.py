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


# ---------- Evaluation metrics (spec 4-2) ----------


def compute_mae(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error. Same shape on both sides."""
    return (pred - target).abs().mean()


def compute_pearson(pred: Tensor, target: Tensor) -> Tensor:
    """Pearson correlation coefficient. Scalar in [-1, 1]."""
    p = pred - pred.mean()
    t = target - target.mean()
    denom = (p.pow(2).sum().sqrt() * t.pow(2).sum().sqrt()).clamp(min=1e-8)
    return (p * t).sum() / denom


def compute_rmse(pred: Tensor, target: Tensor) -> Tensor:
    """Root mean squared error. Scalar tensor."""
    return torch.sqrt(((pred - target) ** 2).mean())


def compute_va_regression_metrics(
    va_pred: Tensor, va_target: Tensor
) -> dict[str, float]:
    """MAE / RMSE / Pearson for valence & arousal. Complements existing CCC.

    RMSE added per train_cog spec (AVEC/MediaEval reporting convention).
    Backward compatible — existing callers see new rmse_* keys but other
    keys unchanged.
    """
    return {
        "mae_valence": compute_mae(va_pred[:, 0], va_target[:, 0]).item(),
        "mae_arousal": compute_mae(va_pred[:, 1], va_target[:, 1]).item(),
        "rmse_valence": compute_rmse(va_pred[:, 0], va_target[:, 0]).item(),
        "rmse_arousal": compute_rmse(va_pred[:, 1], va_target[:, 1]).item(),
        "pearson_valence": compute_pearson(va_pred[:, 0], va_target[:, 0]).item(),
        "pearson_arousal": compute_pearson(va_pred[:, 1], va_target[:, 1]).item(),
    }


def compute_mood_metrics(
    mood_logits: Tensor, mood_target: Tensor, num_classes: int = 7
) -> dict:
    """Accuracy, F1 (macro/weighted), Cohen's kappa, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )

    preds = mood_logits.argmax(dim=-1).cpu().numpy()
    targets = mood_target.cpu().numpy()
    labels = list(range(num_classes))
    return {
        "mood_accuracy": float(accuracy_score(targets, preds)),
        "mood_f1_macro": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "mood_f1_weighted": float(f1_score(targets, preds, labels=labels, average="weighted", zero_division=0)),
        "mood_kappa": float(cohen_kappa_score(targets, preds, labels=labels)),
        "mood_confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),
    }


def compute_cong_accuracy(cong_logits: Tensor, cong_target: Tensor) -> float:
    """Plain accuracy for 3-class congruence head."""
    preds = cong_logits.argmax(dim=-1)
    return (preds == cong_target).float().mean().item()
