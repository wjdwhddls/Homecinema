import numpy as np

from model.autoEQ.train_pseudo.cognimuse_preprocess import (
    _normalize_va,
    aggregate_window_va,
)


def test_normalize_va_passthrough():
    raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    out = _normalize_va(raw, src_range=(-1.0, 1.0))
    np.testing.assert_allclose(out, raw.astype(np.float32))


def test_normalize_va_linear_transform():
    raw = np.array([1.0, 4.0, 7.0])  # Likert 1..7
    out = _normalize_va(raw, src_range=(1.0, 7.0))
    # 1 → -1, 4 → 0, 7 → 1
    np.testing.assert_allclose(out, np.array([-1.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)


def test_normalize_va_oob_assertion():
    import pytest
    raw = np.array([0.0, 10.0, 20.0])  # way outside declared range
    with pytest.raises(AssertionError):
        _normalize_va(raw, src_range=(0.0, 1.0))


def test_aggregate_mean_matches_numpy():
    sr = 25.0
    v = np.sin(np.linspace(0, 2 * np.pi, 1000, dtype=np.float32))  # 40 seconds @ 25 Hz
    a = np.cos(np.linspace(0, 2 * np.pi, 1000, dtype=np.float32))
    mean_v, mean_a, std_v, std_a = aggregate_window_va(v, a, sr, 0.0, 4.0)

    expected_v = v[0 : int(4 * sr)]
    assert abs(mean_v - float(expected_v.mean())) < 1e-5
    assert abs(std_v - float(expected_v.std())) < 1e-5


def test_aggregate_empty_slice_safe():
    v = np.array([0.1, 0.2], dtype=np.float32)
    a = np.array([0.3, 0.4], dtype=np.float32)
    mv, ma, sv, sa = aggregate_window_va(v, a, 25.0, 100.0, 104.0)
    assert (mv, ma, sv, sa) == (0.0, 0.0, 0.0, 0.0)
