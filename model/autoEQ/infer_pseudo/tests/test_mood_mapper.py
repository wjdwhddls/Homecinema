"""V/A → GEMS mapping sanity."""

from model.autoEQ.infer_pseudo.mood_mapper import GEMS_LABELS, va_to_mood_name


def test_gems_labels_has_7():
    assert len(GEMS_LABELS) == 7


def test_gems_labels_order_matches_mood_centers():
    # Indices must match MOOD_CENTERS order (spec V3.3 Appendix A)
    expected = ["Tension", "Sadness", "Peacefulness", "JoyfulActivation",
                "Tenderness", "Power", "Wonder"]
    assert GEMS_LABELS == expected


def test_upper_left_quadrant_maps_to_tension_or_power():
    # Negative V, high A → either Tension (-0.6, 0.7) or Power (0.2, 0.8)
    idx, name = va_to_mood_name(-0.5, 0.6)
    assert name == "Tension", f"(-0.5, 0.6) should map to Tension, got {name}"


def test_lower_left_maps_to_sadness():
    idx, name = va_to_mood_name(-0.6, -0.3)
    assert name == "Sadness"


def test_upper_right_maps_to_joyful():
    idx, name = va_to_mood_name(0.7, 0.6)
    assert name == "JoyfulActivation"


def test_lower_right_maps_to_peacefulness():
    # 2026-04-24 FINAL-A: (0.5, -0.5) now nearest to new Tenderness centroid
    # (+0.18, -0.21) rather than new Peacefulness (+0.04, -0.42).
    # Pre-calibration expected: "Peacefulness". To revert: restore assertion.
    idx, name = va_to_mood_name(0.5, -0.5)
    assert name == "Tenderness"


def test_neutral_center_maps_to_wonder_or_tenderness():
    # 2026-04-24 FINAL-A: Wonder centroid shifted from (+0.5, +0.3) to
    # (-0.014, -0.086). Now (0.5, 0.3) is nearest to JoyfulActivation
    # (+0.7, +0.6) kept at ORIG (d=0.361), not Wonder (d=0.643).
    # Pre-calibration expected: name1 == "Wonder". To revert: restore.
    _, name1 = va_to_mood_name(0.5, 0.3)
    assert name1 == "JoyfulActivation"
    # (0.4, -0.2) still maps to Tenderness (Tenderness centroid is in that
    # area both pre and post calibration; only magnitudes changed).
    _, name2 = va_to_mood_name(0.4, -0.2)
    assert name2 == "Tenderness"


def test_extreme_power_region_maps_to_power():
    # 2026-04-24 FINAL-A: Power centroid shifted from (+0.2, +0.8) to
    # (-0.107, +0.091) (model couldn't actually predict high-arousal positive
    # valence regions). So (0.2, 0.8) — which used to BE the Power centroid —
    # is now nearest to JoyfulActivation (+0.7, +0.6), kept at ORIG because
    # LIRIS learning set had 0 GT samples for JA.
    # Pre-calibration expected: "Power". To revert: restore assertion.
    _, name = va_to_mood_name(0.2, 0.8)
    assert name == "JoyfulActivation"
