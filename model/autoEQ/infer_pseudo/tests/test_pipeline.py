"""Pipeline post-join logic (build_scene_eq_list) — no heavy I/O."""

from model.autoEQ.infer_pseudo.pipeline import build_scene_eq_list
from model.autoEQ.infer_pseudo.types import SceneVA


def _va(idx: int, v: float, a: float) -> SceneVA:
    return SceneVA(
        scene_idx=idx, start_sec=idx * 10.0, end_sec=(idx + 1) * 10.0,
        valence=v, arousal=a, mean_gate_w_v=0.5, mean_gate_w_a=0.5,
    )


def test_build_scene_eq_assigns_mood_from_va():
    # 2026-04-24 FINAL-A centroid calibration: (0.5, -0.5) now maps to
    # Tenderness (new Tenderness centroid (+0.18, -0.21)) instead of
    # Peacefulness (new Peacefulness centroid (+0.04, -0.42)).
    # To revert MOOD_CENTERS to ORIG: expected[2] → "Peacefulness".
    scene_va = [
        _va(0, 0.7, 0.6),   # → JoyfulActivation (JA centroid kept at ORIG)
        _va(1, -0.6, 0.7),  # → Tension (still nearest Tension even post-calibration)
        _va(2, 0.5, -0.5),  # → Tenderness (was Peacefulness pre-calibration)
    ]
    densities = {0: 0.0, 1: 0.5, 2: 0.2}
    out = build_scene_eq_list(scene_va, densities, alpha_d=0.5)
    assert [s.mood for s in out] == ["JoyfulActivation", "Tension", "Tenderness"]


def test_build_scene_eq_respects_density():
    scene_va = [_va(0, 0.7, 0.6)]  # JoyfulActivation
    # density 0 → effective == original
    r0 = build_scene_eq_list(scene_va, {0: 0.0}, alpha_d=0.5)
    for b_orig, b_eff in zip(r0[0].original_bands, r0[0].effective_bands):
        assert b_orig.gain_db == b_eff.gain_db
    # density 1 → voice bands attenuated
    r1 = build_scene_eq_list(scene_va, {0: 1.0}, alpha_d=0.5)
    # B7 (idx 6) JoyfulActivation = +2.0 → effective = +1.0
    assert abs(r1[0].effective_bands[6].gain_db - 1.0) < 1e-6


def test_missing_density_defaults_to_zero():
    scene_va = [_va(5, 0.0, 0.0)]
    out = build_scene_eq_list(scene_va, densities={}, alpha_d=0.5)
    assert out[0].dialogue_density == 0.0


def test_scene_eq_preserves_scene_idx_and_times():
    scene_va = [_va(7, 0.3, 0.1), _va(8, -0.2, 0.4)]
    densities = {7: 0.3, 8: 0.1}
    out = build_scene_eq_list(scene_va, densities, alpha_d=0.5)
    assert [s.scene_idx for s in out] == [7, 8]
    assert out[0].start_sec == 70.0 and out[0].end_sec == 80.0
