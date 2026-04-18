"""명세 테스트 5: Film-level split 검증.

- train/val/test의 movie_id 집합이 서로 disjoint
- union = 전체 집합
- Stratified split 은 V/A 4분면 밸런스 보존
"""

import pytest
import torch

from ..dataset import (
    _va_quadrant,
    compute_movie_va,
    film_level_split,
    stratified_film_level_split,
)


def test_disjoint_splits(synthetic_dataset):
    """train/val/test 영화 ID 집합이 서로 겹치지 않는지 확인."""
    movie_ids = synthetic_dataset.movie_ids
    train_ids, val_ids, test_ids = film_level_split(movie_ids)

    assert train_ids & val_ids == set(), "Train and val overlap"
    assert train_ids & test_ids == set(), "Train and test overlap"
    assert val_ids & test_ids == set(), "Val and test overlap"


def test_union_equals_all(synthetic_dataset):
    """split 합집합이 전체 영화 집합과 동일."""
    movie_ids = synthetic_dataset.movie_ids
    all_ids = set(movie_ids)
    train_ids, val_ids, test_ids = film_level_split(movie_ids)

    assert train_ids | val_ids | test_ids == all_ids


def test_split_sizes(synthetic_dataset):
    """10편 영화 기준 대략적 split 크기 검증."""
    movie_ids = synthetic_dataset.movie_ids
    train_ids, val_ids, test_ids = film_level_split(movie_ids)

    total_films = len(set(movie_ids))
    # Train should have the majority
    assert len(train_ids) >= total_films // 2
    # Val and test should each have at least 1
    assert len(val_ids) >= 1
    assert len(test_ids) >= 1


def test_no_clip_crosses_split(synthetic_dataset):
    """각 클립의 movie_id가 정확히 하나의 split에만 속하는지 확인."""
    movie_ids = synthetic_dataset.movie_ids
    train_ids, val_ids, test_ids = film_level_split(movie_ids)

    for mid in movie_ids:
        in_train = mid in train_ids
        in_val = mid in val_ids
        in_test = mid in test_ids
        assert sum([in_train, in_val, in_test]) == 1, (
            f"Movie {mid} appears in {sum([in_train, in_val, in_test])} splits"
        )


# ---------- Stratified split ----------


def _toy_movie_va() -> dict[int, tuple[float, float]]:
    """8 movies, 2 per quadrant (balanced)."""
    return {
        0: (0.5, 0.5),   # HVHA
        1: (0.3, 0.7),   # HVHA
        2: (0.6, -0.3),  # HVLA
        3: (0.4, -0.5),  # HVLA
        4: (-0.5, 0.5),  # LVHA
        5: (-0.3, 0.7),  # LVHA
        6: (-0.6, -0.3), # LVLA
        7: (-0.4, -0.5), # LVLA
    }


def test_stratified_disjoint():
    movie_va = _toy_movie_va()
    train, val, test = stratified_film_level_split(movie_va, seed=0)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_stratified_union_equals_all():
    movie_va = _toy_movie_va()
    train, val, test = stratified_film_level_split(movie_va, seed=0)
    assert train | val | test == set(movie_va)


def test_stratified_every_quadrant_in_train():
    movie_va = _toy_movie_va()
    train, _, _ = stratified_film_level_split(movie_va, seed=0)
    seen = {_va_quadrant(*movie_va[m]) for m in train}
    assert seen == {"HVHA", "HVLA", "LVHA", "LVLA"}


def test_stratified_quadrant_balance_vs_random():
    """Stratified 는 train 내 분면 다양성이 단순 랜덤보다 (평균적으로) 좋다."""
    movie_va = _toy_movie_va()

    # Stratified: 모든 seed 에서 4분면 전부 등장해야 함
    for seed in range(5):
        train, _, _ = stratified_film_level_split(movie_va, seed=seed)
        quads = {_va_quadrant(*movie_va[m]) for m in train}
        assert quads == {"HVHA", "HVLA", "LVHA", "LVLA"}, f"seed={seed}: {quads}"


def test_compute_movie_va_simple():
    movie_ids = [0, 0, 1, 1, 1]
    valences = [0.2, 0.4, -0.1, -0.3, -0.2]
    arousals = [0.5, 0.5, 0.0, 0.2, 0.1]
    mv = compute_movie_va(movie_ids, valences, arousals)
    assert mv[0] == pytest.approx((0.3, 0.5))
    assert mv[1] == pytest.approx((-0.2, 0.1))


def test_compute_movie_va_accepts_tensors():
    movie_ids = [0, 0, 1]
    valences = torch.tensor([0.2, 0.4, -0.1])
    arousals = torch.tensor([0.5, 0.5, 0.0])
    mv = compute_movie_va(movie_ids, valences, arousals)
    assert mv[0][0] == pytest.approx(0.3)
