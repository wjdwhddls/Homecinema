"""Raised-cosine crossfade correctness."""

import numpy as np

from model.autoEQ.playback.crossfade import (
    apply_boundary_crossfades,
    raised_cosine_crossfade,
)


def test_raised_cosine_endpoints_are_0_and_1():
    w = raised_cosine_crossfade(100)
    assert abs(w[0] - 0.0) < 1e-6
    assert abs(w[-1] - 1.0) < 1e-6


def test_raised_cosine_midpoint_is_half():
    w = raised_cosine_crossfade(101)  # odd so exact midpoint
    assert abs(w[50] - 0.5) < 1e-6


def test_raised_cosine_monotonic_non_decreasing():
    w = raised_cosine_crossfade(256)
    diffs = np.diff(w)
    assert np.all(diffs >= -1e-7), "raised-cosine envelope should be monotone"


def test_raised_cosine_length_one():
    w = raised_cosine_crossfade(1)
    assert len(w) == 1 and w[0] == 1.0


def test_raised_cosine_length_zero():
    w = raised_cosine_crossfade(0)
    assert len(w) == 0


def test_single_scene_returns_unchanged():
    sr = 48000
    audio = np.ones(sr * 2, dtype=np.float32)
    out = apply_boundary_crossfades([audio], sample_rate=sr, crossfade_ms=300)
    assert np.array_equal(out, audio)


def test_two_scenes_crossfade_produces_continuous_signal():
    # Two scenes: first all 1.0, second all -1.0
    # In the crossfade region we should see smooth transition from 1 → -1.
    sr = 1000  # simplicity
    cf_ms = 100  # 100 samples @ 1kHz
    a = np.ones(500, dtype=np.float32)
    b = -np.ones(500, dtype=np.float32)
    out = apply_boundary_crossfades([a, b], sample_rate=sr, crossfade_ms=cf_ms)
    # Output length: 500 + 500 - 100 = 900
    assert out.shape[0] == 900
    # Head unchanged
    assert out[0] == 1.0
    # Tail unchanged
    assert out[-1] == -1.0
    # Somewhere in the middle the signal crosses 0
    mid_region = out[395:505]  # boundary neighborhood
    assert mid_region.min() < 0 < mid_region.max()


def test_two_scenes_blend_matches_formula_midpoint():
    # If prev=1.0 and next=-1.0, midpoint of crossfade should be 1*0.5 + (-1)*0.5 = 0
    sr = 1000
    cf_ms = 200  # 200 samples
    a = np.ones(400, dtype=np.float32)
    b = -np.ones(400, dtype=np.float32)
    out = apply_boundary_crossfades([a, b], sample_rate=sr, crossfade_ms=cf_ms)
    # out length = 400+400-200 = 600
    # Boundary start at sample 200 (end of first scene minus cf), midpoint ~300
    # Check at approx mid of fade
    midpoint = 300
    assert abs(out[midpoint]) < 0.1, f"midpoint should be ~0, got {out[midpoint]}"


def test_three_scenes_total_length():
    sr = 1000
    cf = 100
    scenes = [np.ones(500, dtype=np.float32) * i for i in range(3)]
    out = apply_boundary_crossfades(scenes, sample_rate=sr, crossfade_ms=cf)
    # length = 500*3 - 2*100 = 1300
    assert out.shape[0] == 1300


def test_multichannel_stereo_preserves_shape():
    sr = 1000
    cf = 100
    a = np.ones((500, 2), dtype=np.float32)
    b = -np.ones((500, 2), dtype=np.float32)
    out = apply_boundary_crossfades([a, b], sample_rate=sr, crossfade_ms=cf)
    assert out.ndim == 2 and out.shape[1] == 2
    assert out.shape[0] == 900
    # Both channels transition smoothly
    assert out[0, 0] == 1.0 and out[0, 1] == 1.0
    assert out[-1, 0] == -1.0 and out[-1, 1] == -1.0


def test_empty_input_returns_empty():
    out = apply_boundary_crossfades([], sample_rate=48000)
    assert out.shape[0] == 0


def test_short_scene_cfade_clamps():
    # Scene shorter than cf_samples should not crash; fade clamped to scene len
    sr = 1000
    cf = 500  # 500 samples, but scenes are only 100 samples
    a = np.ones(100, dtype=np.float32)
    b = np.zeros(100, dtype=np.float32)
    out = apply_boundary_crossfades([a, b], sample_rate=sr, crossfade_ms=cf)
    # Output should still produce a valid array (not crash, no NaN)
    assert not np.any(np.isnan(out))
