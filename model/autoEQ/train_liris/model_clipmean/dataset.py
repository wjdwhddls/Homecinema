"""CLIPMean-variant dataset.

PrecomputedLirisDatasetCLIPMean reads `features[name]["clipmean"]` instead of
`features[name]["xclip"]`. Audio key stays `panns` (same feature reused
bit-identically from the BASE PANNs cache — see precompute_clipmean.py).
All other behavior (official_split, MixupTargetShrinkageCollator, V/A norm
A/B, K=4/7 labels, quadrant mixup, target shrinkage) is inherited from the
Base Model's dataset module.
"""

from __future__ import annotations

import torch

from ..dataset import (
    MixupTargetShrinkageCollator,
    PrecomputedLirisDataset,
    apply_va_norm_strategy_B,
    official_split,
    va_to_quadrant_k4,
)


class PrecomputedLirisDatasetCLIPMean(PrecomputedLirisDataset):
    """Same as parent — only the visual feature dict key differs ('xclip' → 'clipmean')."""

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        feat = self.features[row["name"]]
        return {
            "visual": feat["clipmean"].float(),
            "audio": feat["panns"].float(),
            "va_target": torch.tensor(
                [row["v_norm"], row["a_norm"]], dtype=torch.float32
            ),
            "mood_k4": torch.tensor(int(row["quadrant_k4"]), dtype=torch.long),
            "mood_k7": torch.tensor(int(row["mood_k7"]), dtype=torch.long),
            "v_var": torch.tensor(float(row["v_var"]), dtype=torch.float32),
            "a_var": torch.tensor(float(row["a_var"]), dtype=torch.float32),
            "quadrant": torch.tensor(int(row["quadrant_k4"]), dtype=torch.long),
            "name": row["name"],
        }


__all__ = [
    "MixupTargetShrinkageCollator",
    "PrecomputedLirisDatasetCLIPMean",
    "apply_va_norm_strategy_B",
    "official_split",
    "va_to_quadrant_k4",
]
