"""명세 테스트 5: Film-level split 검증.

- train/val/test의 movie_id 집합이 서로 disjoint
- union = 전체 집합
"""

from ..dataset import film_level_split


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
