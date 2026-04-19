"""Shared fixtures for train_pseudo tests."""
from __future__ import annotations

import torch


def make_synthetic_metadata(
    n_per_movie: int = 40,
    stride_sec: float = 2.0,
    window_sec: float = 4.0,
    seed: int = 0,
) -> dict[str, dict]:
    """Build a synthetic cognimuse-style metadata dict for 7 movies."""
    from model.autoEQ.train_pseudo.dataset import COGNIMUSE_MOVIES

    g = torch.Generator().manual_seed(seed)
    metadata: dict[str, dict] = {}
    for mid, code in enumerate(COGNIMUSE_MOVIES):
        for idx in range(n_per_movie):
            t0 = idx * stride_sec
            t1 = t0 + window_sec
            v = float((torch.rand(1, generator=g).item() * 2) - 1)
            a = float((torch.rand(1, generator=g).item() * 2) - 1)
            wid = f"{code}_{idx:05d}"
            metadata[wid] = {
                "movie_id": mid,
                "movie_code": code,
                "valence": v,
                "arousal": a,
                "valence_std": float(torch.rand(1, generator=g).item() * 0.3),
                "arousal_std": float(torch.rand(1, generator=g).item() * 0.3),
                "t0": float(t0),
                "t1": float(t1),
                "annotation_source": "experienced",
            }
    return metadata
