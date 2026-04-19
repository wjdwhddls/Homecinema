"""EQ preset table integrity + dialogue protection formula correctness."""

import pytest

from model.autoEQ.infer_pseudo.eq_preset import (
    BAND_SPECS,
    DEFAULT_ALPHA_D,
    EQ_PRESET_TABLE_DB,
    VOICE_PROTECTED_BAND_INDICES,
    apply_dialogue_protection,
    get_original_bands,
)
from model.autoEQ.infer_pseudo.mood_mapper import GEMS_LABELS


def test_all_gems_moods_have_presets():
    for name in GEMS_LABELS:
        assert name in EQ_PRESET_TABLE_DB, f"{name} missing from EQ preset table"


def test_all_presets_have_10_bands():
    for name, gains in EQ_PRESET_TABLE_DB.items():
        assert len(gains) == 10, f"{name} has {len(gains)} bands, expected 10"


def test_band_specs_are_10():
    assert len(BAND_SPECS) == 10
    # Frequency order must be ascending (31.5 → 16k)
    freqs = [f for f, q in BAND_SPECS]
    assert freqs == sorted(freqs)


def test_all_preset_gains_within_plus_minus_3db():
    # Spec V3.1/V3.2 claim: all values in [-3, +3] dB
    for name, gains in EQ_PRESET_TABLE_DB.items():
        for g in gains:
            assert -3.0 <= g <= 3.0, f"{name} band gain {g} outside [-3, 3]"


def test_get_original_bands_returns_correct_structure():
    bands = get_original_bands("Tension")
    assert len(bands) == 10
    assert bands[0].freq_hz == 31.5
    assert bands[6].freq_hz == 2000.0  # B7
    assert bands[6].gain_db == 2.5


def test_get_original_bands_unknown_mood_raises():
    with pytest.raises(ValueError):
        get_original_bands("NonexistentMood")


def test_dialogue_protection_density_zero_unchanged():
    bands = get_original_bands("Tension")
    out = apply_dialogue_protection(bands, dialogue_density=0.0)
    for b_in, b_out in zip(bands, out):
        assert b_in.gain_db == b_out.gain_db


def test_dialogue_protection_density_one_applies_alpha():
    bands = get_original_bands("Tension")  # B7=+2.5
    out = apply_dialogue_protection(bands, dialogue_density=1.0, alpha_d=0.5)
    # B7 index 6 → gain × 0.5
    assert abs(out[6].gain_db - 2.5 * 0.5) < 1e-6


def test_dialogue_protection_only_affects_voice_bands():
    bands = get_original_bands("Sadness")  # B1=0, B7=-2, B10=-1.5
    out = apply_dialogue_protection(bands, dialogue_density=1.0, alpha_d=0.0)
    for i, (b_in, b_out) in enumerate(zip(bands, out)):
        if i in VOICE_PROTECTED_BAND_INDICES:
            assert b_out.gain_db == b_in.gain_db * 0.0
        else:
            assert b_out.gain_db == b_in.gain_db


def test_dialogue_protection_preserves_sign():
    # Sadness has negative gains at B7/B8 — attenuation must keep them negative
    bands = get_original_bands("Sadness")
    out = apply_dialogue_protection(bands, dialogue_density=0.8, alpha_d=0.5)
    assert out[6].gain_db < 0  # B7 still negative
    assert out[7].gain_db < 0  # B8 still negative


def test_dialogue_protection_zero_gain_stays_zero():
    # Peacefulness B6 = 0.0, must stay 0.0 regardless of density
    bands = get_original_bands("Peacefulness")
    assert bands[5].gain_db == 0.0  # B6
    out = apply_dialogue_protection(bands, dialogue_density=1.0, alpha_d=0.5)
    assert out[5].gain_db == 0.0


def test_dialogue_protection_rejects_out_of_range_density():
    bands = get_original_bands("Tension")
    with pytest.raises(ValueError):
        apply_dialogue_protection(bands, dialogue_density=1.5)
    with pytest.raises(ValueError):
        apply_dialogue_protection(bands, dialogue_density=-0.1)


def test_voice_protected_indices_are_b6_b7_b8():
    # Indices 5, 6, 7 correspond to 1k, 2k, 4k Hz
    assert VOICE_PROTECTED_BAND_INDICES == {5, 6, 7}
    assert BAND_SPECS[5][0] == 1000.0
    assert BAND_SPECS[6][0] == 2000.0
    assert BAND_SPECS[7][0] == 4000.0
