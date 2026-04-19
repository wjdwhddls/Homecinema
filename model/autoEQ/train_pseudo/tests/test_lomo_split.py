import pytest

from model.autoEQ.train_pseudo.dataset import (
    COGNIMUSE_MOVIES,
    lomo_splits_with_time_val,
)
from model.autoEQ.train_pseudo.tests.conftest import make_synthetic_metadata


def test_test_is_full_movie():
    meta = make_synthetic_metadata(n_per_movie=30)
    for fold in range(7):
        train, val, test = lomo_splits_with_time_val(meta, fold, 0.15, 2)
        test_code = COGNIMUSE_MOVIES[fold]
        test_from_meta = [w for w, m in meta.items() if m["movie_code"] == test_code]
        assert sorted(test) == sorted(test_from_meta)


def test_train_val_test_disjoint():
    meta = make_synthetic_metadata(n_per_movie=30)
    train, val, test = lomo_splits_with_time_val(meta, 0, 0.15, 2)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_gap_windows_dropped_no_overlap():
    """With window=4s, stride=2s and gap_windows=2, adjacent train/val windows
    have zero overlap **and** a 2-second time buffer.

    Derivation: last_train.t1 = (N_train-1)*stride + window
                first_val.t0 = (N_train + gap) * stride
                diff = stride*(gap + 1) - window = 2*(2+1) - 4 = 2s
    """
    meta = make_synthetic_metadata(n_per_movie=30)
    train, val, _ = lomo_splits_with_time_val(meta, 0, 0.15, 2)
    for code in COGNIMUSE_MOVIES:
        if code == COGNIMUSE_MOVIES[0]:
            continue
        train_here = sorted(
            [(w, meta[w]["t0"], meta[w]["t1"]) for w in train if meta[w]["movie_code"] == code],
            key=lambda x: x[1],
        )
        val_here = sorted(
            [(w, meta[w]["t0"], meta[w]["t1"]) for w in val if meta[w]["movie_code"] == code],
            key=lambda x: x[1],
        )
        if not train_here or not val_here:
            continue
        last_train_t1 = train_here[-1][2]
        first_val_t0 = val_here[0][1]
        # gap_windows=2 at stride=2 → buffer = 2s; the critical property is
        # that the buffer is > 0 (no temporal overlap between train and val).
        assert first_val_t0 - last_train_t1 >= 2.0 - 1e-6, (
            f"Movie {code}: last_train_t1={last_train_t1}, first_val_t0={first_val_t0}"
        )


def test_gap_exceeding_movie_raises():
    """N=2, gap=2 → N_train = 2 - 0 - 2 = 0, assertion should fire."""
    meta = make_synthetic_metadata(n_per_movie=2)
    with pytest.raises(AssertionError):
        lomo_splits_with_time_val(meta, 0, 0.15, 2)


def test_window_counts_sensible():
    meta = make_synthetic_metadata(n_per_movie=40)
    train, val, test = lomo_splits_with_time_val(meta, 0, 0.15, 2)
    # test movie: 40 windows
    assert len(test) == 40
    # other 6 movies: N=40, N_val=round(40*0.15)=6, N_train=40-6-2=32
    assert len(val) == 6 * 6
    assert len(train) == 6 * 32
