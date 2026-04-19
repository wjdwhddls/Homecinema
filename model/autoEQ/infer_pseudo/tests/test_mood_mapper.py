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
    idx, name = va_to_mood_name(0.5, -0.5)
    assert name == "Peacefulness"


def test_neutral_center_maps_to_wonder_or_tenderness():
    # Near (0.5, 0.3) → Wonder, (0.4, -0.2) → Tenderness
    _, name1 = va_to_mood_name(0.5, 0.3)
    assert name1 == "Wonder"
    _, name2 = va_to_mood_name(0.4, -0.2)
    assert name2 == "Tenderness"


def test_extreme_power_region_maps_to_power():
    # (0.2, 0.8) is Power centroid itself
    _, name = va_to_mood_name(0.2, 0.8)
    assert name == "Power"
