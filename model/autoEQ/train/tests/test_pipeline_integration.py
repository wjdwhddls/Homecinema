"""전체 파이프라인 통합 테스트: 합성 데이터 -> film split -> 3 epoch 학습."""

import torch

from ..config import TrainConfig
from ..dataset import SyntheticAutoEQDataset, film_level_split, create_dataloaders
from ..model import AutoEQModel
from ..trainer import Trainer


def test_full_pipeline():
    """End-to-end: 합성 데이터로 3 epoch 학습 -> 예외 없음, loss finite, CCC in [-1,1]."""
    config = TrainConfig(batch_size=8, epochs=3, warmup_steps=2)
    dataset = SyntheticAutoEQDataset(num_clips=120, num_films=10, config=config, seed=42)

    # Film-level split
    train_ids, val_ids, test_ids = film_level_split(dataset.movie_ids)
    train_loader, val_loader = create_dataloaders(dataset, train_ids, val_ids, config)

    # Model
    model = AutoEQModel(config)
    initial_params = {
        name: p.clone() for name, p in model.named_parameters()
    }

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Train
    history = trainer.fit(max_epochs=3)

    # Verify: no exceptions (implicit), history exists
    assert len(history) == 3

    # Verify: all losses are finite
    for record in history:
        for key in ["va", "mood", "cong", "gate_entropy", "total"]:
            train_loss = record["train"][key]
            val_loss = record["val"][key]
            assert torch.isfinite(torch.tensor(train_loss)), f"Train {key} not finite"
            assert torch.isfinite(torch.tensor(val_loss)), f"Val {key} not finite"

    # Verify: CCC in valid range
    for record in history:
        ccc = record["val"]["mean_ccc"]
        assert -1.0 <= ccc <= 1.0, f"CCC {ccc} out of range [-1, 1]"

    # Verify: model weights changed from initialization
    changed = False
    for name, p in model.named_parameters():
        if not torch.allclose(p, initial_params[name]):
            changed = True
            break
    assert changed, "Model weights did not change during training"


def test_early_stopping_triggers():
    """Early stopping patience 테스트: patience=1이면 빠르게 중단."""
    config = TrainConfig(batch_size=8, epochs=100, warmup_steps=1, early_stop_patience=1)
    dataset = SyntheticAutoEQDataset(num_clips=40, num_films=5, config=config, seed=0)

    train_ids, val_ids, _ = film_level_split(dataset.movie_ids)
    train_loader, val_loader = create_dataloaders(dataset, train_ids, val_ids, config)

    model = AutoEQModel(config)
    trainer = Trainer(model, train_loader, val_loader, config)

    history = trainer.fit()

    # Should stop well before 100 epochs
    assert len(history) < 100, f"Trained {len(history)} epochs, expected early stop"


def test_best_model_restored_after_fit():
    """fit() 종료 후 best epoch의 모델 가중치가 복원되는지 확인."""
    config = TrainConfig(batch_size=8, epochs=5, warmup_steps=2)
    dataset = SyntheticAutoEQDataset(num_clips=120, num_films=10, config=config, seed=42)

    train_ids, val_ids, _ = film_level_split(dataset.movie_ids)
    train_loader, val_loader = create_dataloaders(dataset, train_ids, val_ids, config)

    model = AutoEQModel(config)
    trainer = Trainer(model, train_loader, val_loader, config)

    trainer.fit(max_epochs=5)

    # best_state_dict should have been saved
    assert trainer.best_state_dict is not None, "best_state_dict was never saved"

    # Model should now hold best weights (verify by checking state matches)
    current_state = model.state_dict()
    for key in trainer.best_state_dict:
        assert torch.equal(current_state[key], trainer.best_state_dict[key]), (
            f"Model param '{key}' does not match best_state_dict after fit()"
        )


def test_gate_weights_tracked():
    """Train/val 메트릭에 gate 가중치 통계가 포함되는지 확인."""
    config = TrainConfig(batch_size=8, epochs=2, warmup_steps=2)
    dataset = SyntheticAutoEQDataset(num_clips=120, num_films=10, config=config, seed=42)

    train_ids, val_ids, _ = film_level_split(dataset.movie_ids)
    train_loader, val_loader = create_dataloaders(dataset, train_ids, val_ids, config)

    model = AutoEQModel(config)
    trainer = Trainer(model, train_loader, val_loader, config)

    history = trainer.fit(max_epochs=2)

    for record in history:
        # Train gate stats
        assert "gate_w_v" in record["train"], "train missing gate_w_v"
        assert "gate_w_a" in record["train"], "train missing gate_w_a"
        w_v = record["train"]["gate_w_v"]
        w_a = record["train"]["gate_w_a"]
        assert 0.0 < w_v < 1.0, f"gate_w_v={w_v} out of (0,1)"
        assert 0.0 < w_a < 1.0, f"gate_w_a={w_a} out of (0,1)"

        # Val gate stats
        assert "gate_w_v" in record["val"], "val missing gate_w_v"
        assert "gate_w_a" in record["val"], "val missing gate_w_a"
