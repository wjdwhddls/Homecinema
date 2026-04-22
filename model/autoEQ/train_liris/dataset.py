"""PrecomputedLirisDataset + official split + Quadrant Mixup + Target Shrinkage.

V5-FINAL §9-1, §18.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..train.dataset import MOOD_CENTERS, va_to_mood
from .config import TrainLirisConfig

SPLIT_NAMES = ("train", "val", "test")


def va_to_quadrant_k4(v: float, a: float) -> int:
    """HVHA=0, HVLA=1, LVHA=2, LVLA=3  (matches liris_preprocess convention)."""
    v_high = v >= 0
    a_high = a >= 0
    if v_high and a_high: return 0
    if v_high and not a_high: return 1
    if not v_high and a_high: return 2
    return 3


def apply_va_norm_strategy_B(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Re-normalize v_raw / a_raw to [-1, +1] using per-axis min-max computed
    from `reference_df` (typically train split only, to avoid leakage).

    Also recomputes mood_k7 and quadrant_k4 under the new v/a scale.
    """
    v_min = float(reference_df["v_raw"].min())
    v_max = float(reference_df["v_raw"].max())
    a_min = float(reference_df["a_raw"].min())
    a_max = float(reference_df["a_raw"].max())
    # Stretch + clip (val/test v_raw may slightly exceed train min/max)
    v_range = max(v_max - v_min, 1e-9)
    a_range = max(a_max - a_min, 1e-9)
    df = df.copy()
    df["v_norm"] = ((df["v_raw"] - v_min) / v_range * 2 - 1).clip(-1, 1)
    df["a_norm"] = ((df["a_raw"] - a_min) / a_range * 2 - 1).clip(-1, 1)
    # Recompute mood labels under the new scale
    df["mood_k7"] = df.apply(
        lambda r: va_to_mood(float(r["v_norm"]), float(r["a_norm"])), axis=1
    )
    df["quadrant_k4"] = df.apply(
        lambda r: va_to_quadrant_k4(float(r["v_norm"]), float(r["a_norm"])), axis=1
    )
    return df


class PrecomputedLirisDataset(Dataset):
    """Wraps `features.pt` (dict[name → {xclip, panns}]) with LIRIS metadata.

    Returns per-sample dict:
        visual:    (512,)  float32
        audio:     (2048,) float32
        va_target: (2,)    float32, [v_norm, a_norm]
        mood_k4:   int64
        v_var, a_var: float32 (for Target Shrinkage)
        quadrant:  int64 (0..3 — Mixup grouping)
        name:      str
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        features: dict,
    ):
        self.meta = metadata.reset_index(drop=True)
        self.features = features
        missing = [
            n for n in self.meta["name"].tolist() if n not in features
        ]
        if missing:
            raise ValueError(f"{len(missing)} clips missing from features.pt (first: {missing[:3]})")

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        feat = self.features[row["name"]]
        return {
            "visual": feat["xclip"].float(),
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


def official_split(
    metadata_csv: Path,
    use_full_learning_set: bool = False,
    internal_val_films: int = 16,
    split_seed: int = 42,
    va_norm_strategy: str = "A",
) -> dict[str, pd.DataFrame]:
    """LIRIS official 40/40/80 film split via `liris_metadata.csv::split`.

    Default (use_full_learning_set=False):
        train = 40 films (2450 clips)  — set=1 learning
        val   = 40 films (2450 clips)  — set=2 validation
        test  = 80 films (4900 clips)  — set=0 hold-out
        MediaEval 2015-2018 benchmark protocol.

    If use_full_learning_set=True (V5-FINAL §15 #7, fixed 2026-04-21):
        Follows LIRIS original paper (Baveye 2015) protocol — merge
        learning + validation into 80 films, then carve out a film-level
        `internal_val_films` subset (default 16) for early-stopping.
        Disjoint at film level, deterministic via `split_seed`.

        train = 80 - internal_val_films films  (default 64 films)
        val   = internal_val_films films       (default 16 films)
        test  = 80 films (unchanged)
    """
    df = pd.read_csv(metadata_csv)
    needed = {"name", "film_id", "split", "v_norm", "a_norm", "v_var", "a_var",
              "mood_k7", "quadrant_k4"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"metadata csv missing columns: {missing}")
    splits = {name: df[df["split"] == name].reset_index(drop=True) for name in SPLIT_NAMES}
    if use_full_learning_set:
        merged = pd.concat([splits["train"], splits["val"]], ignore_index=True)
        all_films = sorted(merged.film_id.unique())
        if len(all_films) != 80:
            raise ValueError(f"expected 80 learning films, got {len(all_films)}")
        rng = np.random.RandomState(split_seed)
        val_film_list = list(rng.choice(all_films, size=internal_val_films, replace=False))
        val_film_set = set(val_film_list)
        new_val = merged[merged.film_id.isin(val_film_set)].reset_index(drop=True)
        new_train = merged[~merged.film_id.isin(val_film_set)].reset_index(drop=True)
        # Enforce zero leakage
        assert set(new_train.film_id.unique()).isdisjoint(set(new_val.film_id.unique())), \
            "train/val film leakage in use_full_learning_set"
        splits = {"train": new_train, "val": new_val, "test": splits["test"]}

    # V/A normalization strategy (Phase 2a-1)
    if va_norm_strategy.upper() == "B":
        ref = splits["train"]
        splits = {k: apply_va_norm_strategy_B(v, ref) for k, v in splits.items()}
    elif va_norm_strategy.upper() != "A":
        raise ValueError(f"unknown va_norm_strategy: {va_norm_strategy}")

    return splits


# --- Augmentations (collate-time) ---------------------------------------------


class MixupTargetShrinkageCollator:
    """Collate-fn wrapper applying:
       1. Quadrant Mixup  (prob=p, α=alpha, pair only same quadrant)
       2. Target Shrinkage (per-axis variance gate → ε shrink toward 0)

    Both no-op when `active=False` (val/test).

    Order is fixed per V5-FINAL §18: Mixup → Target Shrinkage.
    """

    def __init__(self, cfg: TrainLirisConfig, active: bool):
        self.cfg = cfg
        self.active = active

    def _stack(self, items: list[dict], key: str) -> Tensor:
        return torch.stack([it[key] for it in items], dim=0)

    def __call__(self, batch: list[dict]) -> dict:
        out: dict = {
            "visual": self._stack(batch, "visual"),
            "audio": self._stack(batch, "audio"),
            "va_target": self._stack(batch, "va_target"),
            "mood_k4": self._stack(batch, "mood_k4"),
            "mood_k7": self._stack(batch, "mood_k7"),
            "v_var": self._stack(batch, "v_var"),
            "a_var": self._stack(batch, "a_var"),
            "quadrant": self._stack(batch, "quadrant"),
            "name": [it["name"] for it in batch],
        }

        if not self.active:
            return out

        out = self._quadrant_mixup(out)
        out = self._target_shrinkage(out)
        return out

    def _quadrant_mixup(self, batch: dict) -> dict:
        if self.cfg.mixup_prob <= 0 or self.cfg.mixup_alpha <= 0:
            return batch
        if random.random() >= self.cfg.mixup_prob:
            return batch

        quadrant = batch["quadrant"]
        B = quadrant.size(0)
        # Pair indices within each quadrant. Unpaired samples left as-is.
        pairs: list[tuple[int, int]] = []
        used = set()
        for q in quadrant.unique().tolist():
            idxs = (quadrant == q).nonzero(as_tuple=True)[0].tolist()
            random.shuffle(idxs)
            for i in range(0, len(idxs) - 1, 2):
                a, b = idxs[i], idxs[i + 1]
                pairs.append((a, b))
                used.update((a, b))

        if not pairs:
            return batch

        lam = float(np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha))
        lam_shrink = max(lam, 1.0 - lam)  # shrink toward strong sample — §18

        for (a, b) in pairs:
            batch["visual"][a] = lam_shrink * batch["visual"][a] + (1 - lam_shrink) * batch["visual"][b]
            batch["visual"][b] = lam_shrink * batch["visual"][b] + (1 - lam_shrink) * batch["visual"][a]
            batch["audio"][a] = lam_shrink * batch["audio"][a] + (1 - lam_shrink) * batch["audio"][b]
            batch["audio"][b] = lam_shrink * batch["audio"][b] + (1 - lam_shrink) * batch["audio"][a]
            batch["va_target"][a] = lam_shrink * batch["va_target"][a] + (1 - lam_shrink) * batch["va_target"][b]
            batch["va_target"][b] = lam_shrink * batch["va_target"][b] + (1 - lam_shrink) * batch["va_target"][a]
            # Mood label stays (same quadrant already).

        return batch

    def _target_shrinkage(self, batch: dict) -> dict:
        eps = self.cfg.target_shrinkage_eps
        if eps <= 0:
            return batch
        v_high = batch["v_var"] > self.cfg.v_var_threshold
        a_high = batch["a_var"] > self.cfg.a_var_threshold
        if self.cfg.shrinkage_logic.upper() == "AND":
            mask = v_high & a_high
        else:  # OR
            mask = v_high | a_high
        if not mask.any():
            return batch
        mask_f = mask.float().unsqueeze(-1)
        batch["va_target"] = batch["va_target"] * (1.0 - eps * mask_f)
        return batch
