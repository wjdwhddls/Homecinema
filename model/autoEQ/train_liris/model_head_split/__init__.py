"""Phase 2a-6 Head Structure Ablation — Separate V-head + A-head.

Built on top of Phase 2a-5 (model_fusion). Replaces the BASE's joint VAHead
(1024 → 256 → 2) with SeparateVAHead (two parallel 1024 → 256 → 1 heads)
while keeping AudioProjection / MoodHead / fusion mechanisms byte-identical
to their Phase 2a-5 counterparts.

Motivation (diagnosed in Phase 2a-5 Gate analysis, 2026-04-21):
    The joint VAHead forced a single 1024-d fused vector to serve both V and
    A regression simultaneously. The Gate Network therefore learned a
    compromise modality weighting (w_v ≈ 0.76) that's slightly optimal for V
    and slightly suboptimal for A, resulting in the observed weak adaptive
    behavior (per-sample std = 0.06). Separating V/A heads gives each
    dimension its own prediction path and lets the shared fused vector be
    processed through dimension-specialized transforms.

Scope:
    - Head structure is ALWAYS 'separate' in this subpackage (subpackage
      identity == axis-3 treatment).
    - Fusion axis is re-swept: {concat, gate (BASE), gmu}.
    - 3 new variants × 3 seeds = 9 new runs for the 2×3 factorial.

Usage:
    from model.autoEQ.train_liris.model_head_split.config import TrainLirisConfigHeadSplit
    from model.autoEQ.train_liris.model_head_split.model import AutoEQModelLirisHeadSplit
    from model.autoEQ.train_liris.model_head_split.heads import SeparateVAHead

No change to Phase 2a-5 or BASE code (zero-modification subpackage pattern).
"""
