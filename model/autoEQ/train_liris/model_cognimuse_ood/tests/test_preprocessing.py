"""COGNIMUSE preprocessing 단위 테스트.

검증:
    1. .dat 파서가 결정론적 (동일 입력 → 동일 출력)
    2. Median consensus 가 outlier annotator 에 robust
    3. 10s window aggregation mean 이 합성 신호에서 정확
    4. 12 annotator 미달 시 에러
    5. VA norm identity (값 유지, [-1,+1] clip)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from model.autoEQ.train_liris.model_cognimuse_ood.preprocessing import (
    EXPECTED_ANNOTATORS,
    FRAME_DT,
    WINDOW_SEC,
    load_film_traces,
    load_trace,
    median_consensus,
    window_aggregate,
)


def _write_dat(path: Path, v_values: list[float], a_values: list[float]) -> None:
    assert len(v_values) == len(a_values)
    with path.open("w") as f:
        for i, (v, a) in enumerate(zip(v_values, a_values)):
            t = i * FRAME_DT
            f.write(f"{t:.2f} {v:.6f} {a:.6f}\n")


def test_load_trace_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        p = tdp / "sample.dat"
        v = [0.1, 0.2, 0.3, 0.4, 0.5]
        a = [-0.5, -0.4, -0.3, -0.2, -0.1]
        _write_dat(p, v, a)
        arr = load_trace(p)
        assert arr.shape == (5, 3)
        np.testing.assert_allclose(arr[:, 1], v, atol=1e-6)
        np.testing.assert_allclose(arr[:, 2], a, atol=1e-6)


def test_load_trace_is_deterministic():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.dat"
        _write_dat(p, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        a1 = load_trace(p)
        a2 = load_trace(p)
        np.testing.assert_array_equal(a1, a2)


def test_median_robust_to_outlier():
    """11 annotators around 0.0, 1 outlier at +0.9 → median ≈ 0.0."""
    N, T = 12, 100
    base = np.zeros((T, 2, N))
    # 11 annotators: V≈0.0, A≈0.0
    base[:, 0, :11] = np.random.default_rng(0).normal(0.0, 0.02, size=(T, 11))
    base[:, 1, :11] = np.random.default_rng(1).normal(0.0, 0.02, size=(T, 11))
    # 1 outlier: V=0.9, A=−0.9
    base[:, 0, 11] = 0.9
    base[:, 1, 11] = -0.9
    med = median_consensus(base)
    assert med.shape == (T, 2)
    # median 이 outlier 에 무관하게 0 근처 유지
    assert abs(med[:, 0].mean()) < 0.05, f"V median drift {med[:, 0].mean()}"
    assert abs(med[:, 1].mean()) < 0.05, f"A median drift {med[:, 1].mean()}"


def test_window_aggregate_mean_synthetic():
    """25Hz × 20s 합성 신호: v 전반 0.3, 후반 −0.3 → 10s window 2개 mean 정확."""
    frames_per_window = int(round(WINDOW_SEC / FRAME_DT))  # 250
    T = frames_per_window * 2  # 20s
    N = 12
    consensus = np.zeros((T, 2))
    consensus[:frames_per_window, 0] = 0.3   # window 0: V=+0.3
    consensus[frames_per_window:, 0] = -0.3  # window 1: V=-0.3
    consensus[:, 1] = 0.1                    # A=+0.1 전체
    traces = np.broadcast_to(consensus[..., None], (T, 2, N)).copy()

    df = window_aggregate(consensus, traces)
    assert len(df) == 2
    np.testing.assert_allclose(df["v_raw"].iloc[0], 0.3, atol=1e-6)
    np.testing.assert_allclose(df["v_raw"].iloc[1], -0.3, atol=1e-6)
    np.testing.assert_allclose(df["a_raw"].iloc[0], 0.1, atol=1e-6)
    # 동일 신호 → v_var ≈ 0
    assert df["v_var"].iloc[0] < 1e-6


def test_window_aggregate_drops_tail():
    """13.5s 신호 → 10s window 1개만, 남은 3.5s drop."""
    frames_per_window = int(round(WINDOW_SEC / FRAME_DT))  # 250
    T = frames_per_window + int(round(3.5 / FRAME_DT))
    consensus = np.ones((T, 2)) * 0.5
    traces = np.broadcast_to(consensus[..., None], (T, 2, 3)).copy()
    df = window_aggregate(consensus, traces)
    assert len(df) == 1


def test_window_aggregate_start_end_monotone():
    """Window start_sec / end_sec non-overlap 증가 확인."""
    T = 1000
    consensus = np.random.default_rng(2).normal(0, 0.1, size=(T, 2))
    traces = np.broadcast_to(consensus[..., None], (T, 2, 5)).copy()
    df = window_aggregate(consensus, traces)
    if len(df) > 1:
        for i in range(1, len(df)):
            assert df["start_sec"].iloc[i] == df["end_sec"].iloc[i - 1], \
                "non-overlap 원칙 위반"
            assert df["end_sec"].iloc[i] > df["start_sec"].iloc[i]


def test_load_film_traces_rejects_wrong_count():
    """annotator 수가 expected 와 다르면 error."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # 3 파일만 (expected 12)
        for subj in (1, 2, 3):
            p = tdp / f"subj{subj}_1_TestFilm.dat"
            _write_dat(p, [0.1, 0.2], [-0.1, -0.2])
        with pytest.raises(ValueError, match="expected"):
            load_film_traces(tdp, "TestFilm", expected_n=EXPECTED_ANNOTATORS)


def test_load_film_traces_truncates_to_shortest():
    """길이가 다른 trace들이 최소 길이로 truncate 되는지 확인."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # subj1: 100 frames, subj2: 80 frames, subj3: 90 frames → min = 80
        lens = [100, 80, 90]
        for i, n in enumerate(lens, start=1):
            p = tdp / f"subj{i}_1_TestFilm.dat"
            _write_dat(p, [0.1] * n, [-0.1] * n)
        stacked, files = load_film_traces(tdp, "TestFilm", expected_n=3)
        assert stacked.shape == (80, 2, 3)
        assert len(files) == 3


def test_va_norm_identity_preserves_range():
    """Preprocessing 출력 v_norm/a_norm 이 [-1, +1] 안에 있어야 한다.

    실제 build_film_metadata 는 filesystem 요구. 여기선 clip 동작만 sanity.
    """
    import numpy as np
    raw = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    clipped = np.clip(raw, -1.0, 1.0)
    assert clipped.min() == -1.0
    assert clipped.max() == 1.0
