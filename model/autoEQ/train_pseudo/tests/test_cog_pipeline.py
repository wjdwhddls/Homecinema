"""End-to-end smoke test using synthetic features.

Bypasses cognimuse_preprocess (which requires real MP4 files) by building
the three .pt files directly, then running a 2-epoch training loop.
"""
from __future__ import annotations

from pathlib import Path

import torch

from model.autoEQ.train_pseudo.config import TrainCogConfig
from model.autoEQ.train_pseudo.dataset import (
    COGNIMUSE_MOVIES,
    PrecomputedCogDataset,
    apply_sigma_filter,
    create_dataloaders_cog,
    lomo_splits_with_time_val,
)
from model.autoEQ.train_pseudo.model_base.model import AutoEQModelCog
from model.autoEQ.train_pseudo.trainer import TrainerCog


def _write_synthetic_features(
    out: Path,
    n_per_movie: int = 40,
    visual_dim: int = 512,
    audio_dim: int = 2048,
    seed: int = 0,
) -> None:
    g = torch.Generator().manual_seed(seed)
    visual: dict[str, torch.Tensor] = {}
    audio: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict] = {}
    for mid, code in enumerate(COGNIMUSE_MOVIES):
        for idx in range(n_per_movie):
            wid = f"{code}_{idx:05d}"
            visual[wid] = torch.randn(visual_dim, generator=g)
            audio[wid] = torch.randn(audio_dim, generator=g)
            v = float(torch.rand(1, generator=g).item() * 2 - 1)
            a = float(torch.rand(1, generator=g).item() * 2 - 1)
            metadata[wid] = {
                "movie_id": mid,
                "movie_code": code,
                "valence": v,
                "arousal": a,
                "valence_std": float(torch.rand(1, generator=g).item() * 0.2),
                "arousal_std": float(torch.rand(1, generator=g).item() * 0.2),
                "t0": idx * 2.0,
                "t1": idx * 2.0 + 4.0,
                "annotation_source": "experienced",
            }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(visual, out / "cognimuse_visual.pt")
    torch.save(audio, out / "cognimuse_audio.pt")
    torch.save(metadata, out / "cognimuse_metadata.pt")


def test_two_epoch_smoke(tmp_path):
    _write_synthetic_features(tmp_path, n_per_movie=40)

    cfg = TrainCogConfig(epochs=2, batch_size=8, warmup_steps=5, early_stop_patience=10)
    dataset = PrecomputedCogDataset(str(tmp_path), "cognimuse",
                                    num_mood_classes=cfg.num_mood_classes)

    train_ids, val_ids, test_ids = lomo_splits_with_time_val(
        dataset.metadata, fold=0,
        val_tail_ratio=cfg.val_tail_ratio,
        gap_windows=cfg.val_gap_windows,
    )
    assert train_ids and val_ids and test_ids

    train_ids = apply_sigma_filter(train_ids, dataset.metadata, cfg.sigma_filter_threshold)
    train_loader, val_loader = create_dataloaders_cog(dataset, train_ids, val_ids, cfg)

    model = AutoEQModelCog(cfg)
    trainer = TrainerCog(model, train_loader, val_loader, cfg, device=torch.device("cpu"))

    history = trainer.fit()
    assert len(history) == 2
    # validation recorded expected keys
    val0 = history[0]["val"]
    for key in (
        "mean_ccc", "ccc_valence", "ccc_arousal",
        "mae_valence", "mae_arousal", "mean_mae",
        "rmse_valence", "rmse_arousal", "mean_rmse",
        "mood_f1_macro", "mood_kappa",
    ):
        assert key in val0, f"missing val metric: {key}"
    assert "cong" not in val0


def test_early_stopping_tuple_update(tmp_path):
    _write_synthetic_features(tmp_path, n_per_movie=30)
    cfg = TrainCogConfig(epochs=1, batch_size=8, warmup_steps=1, early_stop_patience=10)
    dataset = PrecomputedCogDataset(str(tmp_path), "cognimuse")
    t, v, _ = lomo_splits_with_time_val(dataset.metadata, 0, cfg.val_tail_ratio, cfg.val_gap_windows)
    loaders = create_dataloaders_cog(dataset, t, v, cfg)
    trainer = TrainerCog(AutoEQModelCog(cfg), *loaders, cfg, device=torch.device("cpu"))

    # inject two metric dicts: second has identical CCC but better MAE → should still update
    should_stop1 = trainer.check_early_stopping(
        {"mean_ccc": 0.1, "mean_mae": 0.4, "mae_valence": 0.4, "mae_arousal": 0.4}
    )
    assert should_stop1 is False
    assert trainer.best_mean_ccc == 0.1

    should_stop2 = trainer.check_early_stopping(
        {"mean_ccc": 0.1, "mean_mae": 0.3, "mae_valence": 0.3, "mae_arousal": 0.3}
    )
    assert should_stop2 is False
    assert trainer.best_mean_mae == 0.3  # MAE improved under identical CCC
