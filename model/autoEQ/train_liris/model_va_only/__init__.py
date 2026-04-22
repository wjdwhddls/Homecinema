"""Phase 2a-7 Multi-task Regularization ablation — VA-only (no Mood head).

Tests whether the auxiliary Mood classification task (MoodHead with K=7 GEMS
classes) provides regularization benefit to the primary V/A regression task,
or whether it adds noise/interference.

Design (Option B'):
    - MoodHead is REMOVED from the model (clean param count, true VA-only
      architecture: 3,152,388 vs BASE 3,417,099, −264,711 params).
    - Forward returns a dummy torch.zeros((B, K)) tensor for `mood_logits` so
      the BASE trainer/loss code runs unchanged (no trainer.py modification).
    - Config overrides `lambda_mood = 0.0` so the zero-logit mood loss
      (constant ≈ ln(K) = 1.945) contributes nothing to the total loss.
    - Net effect: L_mood has zero gradient, zero contribution to optimization
      — equivalent to single-task V/A training.

OAT axis: "with vs without multi-task Mood auxiliary loss."
All other hyperparameters (K=7 architectural, BASE-inherited) preserved.

Usage:
    from model.autoEQ.train_liris.model_va_only.config import TrainLirisConfigVAOnly
    from model.autoEQ.train_liris.model_va_only.model import AutoEQModelLirisVAOnly
"""
