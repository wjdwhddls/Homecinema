"""Wandb logging hook tests — covers on/off, fallback, payload shape."""

import builtins
from unittest.mock import MagicMock

import pytest
import torch

from ..config import TrainConfig
from ..dataset import SyntheticAutoEQDataset, create_dataloaders, film_level_split
from ..model import AutoEQModel
from ..trainer import Trainer


def _make_trainer(config: TrainConfig, wandb_run=None) -> Trainer:
    dataset = SyntheticAutoEQDataset(num_clips=40, num_films=5, config=config, seed=0)
    train_ids, val_ids, _ = film_level_split(dataset.movie_ids)
    train_loader, val_loader = create_dataloaders(dataset, train_ids, val_ids, config)
    model = AutoEQModel(config)
    return Trainer(model, train_loader, val_loader, config, wandb_run=wandb_run)


def test_trainer_works_without_wandb():
    config = TrainConfig(batch_size=8, epochs=2, warmup_steps=1, use_wandb=False)
    trainer = _make_trainer(config)
    assert trainer.wandb_run is None
    history = trainer.fit(max_epochs=2)
    assert len(history) == 2


def test_wandb_disabled_when_not_installed(monkeypatch):
    config = TrainConfig(batch_size=8, epochs=1, warmup_steps=1, use_wandb=True)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("simulated: wandb not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    trainer = _make_trainer(config)
    assert trainer.wandb_run is None
    history = trainer.fit(max_epochs=1)
    assert len(history) == 1


def test_wandb_log_called_per_epoch():
    config = TrainConfig(batch_size=8, epochs=2, warmup_steps=1)
    mock_run = MagicMock()
    trainer = _make_trainer(config, wandb_run=mock_run)

    trainer.fit(max_epochs=2)

    assert mock_run.log.call_count == 2
    mock_run.finish.assert_called_once()


def test_wandb_log_payload_keys():
    config = TrainConfig(batch_size=8, epochs=1, warmup_steps=1)
    mock_run = MagicMock()
    trainer = _make_trainer(config, wandb_run=mock_run)

    trainer.fit(max_epochs=1)

    payload = mock_run.log.call_args[0][0]
    for expected in (
        "epoch",
        "lr",
        "train/total",
        "val/total",
        "val/mean_ccc",
        "train/gate_w_v",
        "train/gate_w_a",
        "train/grad_norm/va",
        "train/grad_norm/mood",
        "train/grad_norm/cong",
    ):
        assert expected in payload, f"missing key: {expected}"
    for v in payload.values():
        if isinstance(v, float):
            assert torch.isfinite(torch.tensor(v))
