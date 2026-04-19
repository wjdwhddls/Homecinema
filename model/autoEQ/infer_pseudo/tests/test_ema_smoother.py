"""EMA scene-boundary reset + aggregation correctness."""

from model.autoEQ.infer_pseudo.ema_smoother import (
    aggregate_by_scene,
    apply_ema_within_scenes,
)
from model.autoEQ.infer_pseudo.types import WindowVA


def _w(scene_idx: int, i: int, v: float, a: float) -> WindowVA:
    return WindowVA(
        scene_idx=scene_idx, window_idx_in_scene=i,
        start_sec=float(i), end_sec=float(i) + 4.0,
        valence=v, arousal=a, gate_w_v=0.5, gate_w_a=0.5,
    )


def test_ema_empty_input():
    assert apply_ema_within_scenes([]) == []


def test_cold_start_first_3_windows_unchanged():
    # All same scene, first 3 windows should keep raw values
    wins = [_w(0, i, v=0.1 * (i + 1), a=0.0) for i in range(3)]
    out = apply_ema_within_scenes(wins, alpha=0.5, cold_start=3)
    for i in range(3):
        assert out[i].valence == wins[i].valence, f"cold start window {i} modified"


def test_ema_applied_after_cold_start():
    wins = [
        _w(0, 0, 0.1, 0.0),  # cold
        _w(0, 1, 0.2, 0.0),  # cold
        _w(0, 2, 0.3, 0.0),  # cold
        _w(0, 3, 1.0, 0.0),  # ema: 0.5*1.0 + 0.5*0.3 = 0.65
        _w(0, 4, 1.0, 0.0),  # ema: 0.5*1.0 + 0.5*0.65 = 0.825
    ]
    out = apply_ema_within_scenes(wins, alpha=0.5, cold_start=3)
    assert abs(out[3].valence - 0.65) < 1e-6
    assert abs(out[4].valence - 0.825) < 1e-6


def test_scene_boundary_resets_ema():
    # Scene 0: converged to some EMA value. Scene 1: must reset to first raw.
    wins_s0 = [_w(0, i, 1.0, 1.0) for i in range(5)]
    wins_s1 = [_w(1, 0, -0.5, -0.5)]  # scene boundary
    out = apply_ema_within_scenes(wins_s0 + wins_s1, alpha=0.5, cold_start=3)
    # First window of scene 1 should equal its raw value (reset)
    assert out[-1].valence == -0.5
    assert out[-1].arousal == -0.5


def test_aggregate_mean_matches_numpy():
    wins = [
        _w(0, 0, 0.2, -0.1),
        _w(0, 1, 0.4,  0.1),
        _w(0, 2, 0.6,  0.3),
    ]
    agg = aggregate_by_scene(wins)
    assert len(agg) == 1
    assert abs(agg[0].valence - 0.4) < 1e-6
    assert abs(agg[0].arousal - 0.1) < 1e-6


def test_aggregate_preserves_scene_order_and_distinct_groups():
    wins = [
        _w(2, 0, 1.0, 0.0),
        _w(2, 1, 1.0, 0.0),
        _w(5, 0, -1.0, 0.0),
    ]
    agg = aggregate_by_scene(wins)
    assert [s.scene_idx for s in agg] == [2, 5]
    assert agg[0].valence == 1.0
    assert agg[1].valence == -1.0


def test_aggregate_uses_scene_bounds_if_provided():
    wins = [_w(3, 0, 0.5, 0.5)]  # window says 0-4s
    agg = aggregate_by_scene(wins, scene_bounds={3: (120.0, 125.0)})
    assert agg[0].start_sec == 120.0 and agg[0].end_sec == 125.0


def test_gate_weights_averaged():
    wins = [
        WindowVA(0, 0, 0, 4, 0.0, 0.0, gate_w_v=0.8, gate_w_a=0.2),
        WindowVA(0, 1, 1, 5, 0.0, 0.0, gate_w_v=0.6, gate_w_a=0.4),
    ]
    agg = aggregate_by_scene(wins)
    assert abs(agg[0].mean_gate_w_v - 0.7) < 1e-6
    assert abs(agg[0].mean_gate_w_a - 0.3) < 1e-6
