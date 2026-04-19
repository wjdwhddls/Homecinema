"""평가 지표 확장 테스트 (spec 4-2): MAE, Pearson, F1, kappa, confusion, cong accuracy."""

import pytest
import torch

from ..utils import (
    compute_cong_accuracy,
    compute_mae,
    compute_mood_metrics,
    compute_pearson,
    compute_va_regression_metrics,
)


def test_mae_zero_for_identical():
    x = torch.randn(20)
    assert compute_mae(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_mae_known_value():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 2.0, 4.0])
    # |1-2| + |2-2| + |3-4| = 2, mean = 2/3
    assert compute_mae(pred, target).item() == pytest.approx(2.0 / 3.0)


def test_pearson_perfect_correlation():
    x = torch.randn(50)
    y = 2 * x + 1.0
    assert compute_pearson(x, y).item() == pytest.approx(1.0, abs=1e-5)


def test_pearson_anti_correlation():
    x = torch.randn(50)
    y = -3 * x
    assert compute_pearson(x, y).item() == pytest.approx(-1.0, abs=1e-5)


def test_va_regression_metrics_dict_keys():
    pred = torch.randn(30, 2)
    target = torch.randn(30, 2)
    out = compute_va_regression_metrics(pred, target)
    assert set(out) == {
        "mae_valence", "mae_arousal",
        "rmse_valence", "rmse_arousal",
        "pearson_valence", "pearson_arousal",
    }
    for v in out.values():
        assert torch.isfinite(torch.tensor(v))


def test_mood_metrics_perfect_prediction():
    # 7 samples, one per class, prediction perfectly matches
    logits = torch.eye(7) * 10.0  # argmax = diagonal index
    target = torch.arange(7)
    m = compute_mood_metrics(logits, target, num_classes=7)
    assert m["mood_accuracy"] == pytest.approx(1.0)
    assert m["mood_f1_macro"] == pytest.approx(1.0)
    assert m["mood_f1_weighted"] == pytest.approx(1.0)
    assert m["mood_kappa"] == pytest.approx(1.0)
    cm = m["mood_confusion_matrix"]
    assert len(cm) == 7 and len(cm[0]) == 7
    for i in range(7):
        assert cm[i][i] == 1


def test_mood_metrics_random_bounds():
    torch.manual_seed(0)
    logits = torch.randn(50, 7)
    target = torch.randint(0, 7, (50,))
    m = compute_mood_metrics(logits, target, num_classes=7)
    assert 0.0 <= m["mood_accuracy"] <= 1.0
    assert 0.0 <= m["mood_f1_macro"] <= 1.0
    assert 0.0 <= m["mood_f1_weighted"] <= 1.0
    assert -1.0 <= m["mood_kappa"] <= 1.0


def test_cong_accuracy_perfect():
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    target = torch.tensor([0, 1, 2])
    assert compute_cong_accuracy(logits, target) == pytest.approx(1.0)


def test_cong_accuracy_partial():
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    target = torch.tensor([0, 1, 2])  # 2/3 correct
    assert compute_cong_accuracy(logits, target) == pytest.approx(2.0 / 3.0)


def test_validate_returns_new_metrics():
    """Trainer.validate() 가 확장된 지표들을 실제로 반환하는지 통합 검증."""
    from ..config import TrainConfig
    from ..dataset import SyntheticAutoEQDataset, create_dataloaders, film_level_split
    from ..model import AutoEQModel
    from ..trainer import Trainer

    config = TrainConfig(batch_size=8, epochs=1, warmup_steps=1)
    dataset = SyntheticAutoEQDataset(num_clips=40, num_films=5, config=config, seed=0)
    train_ids, val_ids, _ = film_level_split(dataset.movie_ids)
    train_loader, val_loader = create_dataloaders(dataset, train_ids, val_ids, config)
    model = AutoEQModel(config)
    trainer = Trainer(model, train_loader, val_loader, config)

    val = trainer.validate()
    for k in (
        "mae_valence", "mae_arousal", "pearson_valence", "pearson_arousal",
        "mood_accuracy", "mood_f1_macro", "mood_f1_weighted", "mood_kappa",
        "mood_confusion_matrix", "cong_accuracy",
    ):
        assert k in val, f"missing: {k}"
