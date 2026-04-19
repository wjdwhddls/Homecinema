"""10-band EQ presets (spec V3.3 §5-7 = V3.2 §6-4, values unchanged) +
dialogue-protection formula.

Gain values are the V3.1/V3.2 reference table. All bands use biquad peaking
filters; Q factors from spec V3.2 §6-2.
"""

from __future__ import annotations

from .types import EQBand

# Frequency + Q per §6-2
BAND_SPECS: list[tuple[float, float]] = [
    (31.5, 0.7),
    (63.0, 0.7),
    (125.0, 1.0),
    (250.0, 1.2),
    (500.0, 1.4),
    (1000.0, 1.4),
    (2000.0, 1.4),
    (4000.0, 1.2),
    (8000.0, 0.7),
    (16000.0, 0.7),
]

# §6-4 table — gain_db per (mood, band) in order above.
EQ_PRESET_TABLE_DB: dict[str, list[float]] = {
    "Tension":          [+2.0, +2.0, +1.0,  0.0,  0.0, +1.0, +2.5, +2.0,  0.0, -1.0],
    "Sadness":          [ 0.0, +1.0, +1.0, +1.0,  0.0,  0.0, -2.0, -2.0, -1.5, -1.5],
    "Peacefulness":     [ 0.0,  0.0, +0.5, +0.5,  0.0,  0.0, -0.5, -1.0, -0.5,  0.0],
    "JoyfulActivation": [-1.0, -1.0,  0.0,  0.0, +1.0, +1.5, +2.0, +2.0, +1.5, +1.0],
    "Tenderness":       [ 0.0, +1.0, +2.0, +1.5, +0.5,  0.0, -1.0, -1.5, -1.0, -0.5],
    "Power":            [+2.5, +2.0, +2.0, +1.0, +0.5, +1.0, +1.5, +1.0,  0.0,  0.0],
    "Wonder":           [ 0.0,  0.0,  0.0, -0.5,  0.0, +0.5, +1.0, +1.5, +1.5, +2.0],
}

# Voice-protected bands: B6 (1kHz), B7 (2kHz), B8 (4kHz) — indices 5, 6, 7
VOICE_PROTECTED_BAND_INDICES: set[int] = {5, 6, 7}

DEFAULT_ALPHA_D = 0.5  # dialogue protection strength (0.3..0.7 tunable)


def get_original_bands(mood_name: str) -> list[EQBand]:
    """Look up the raw EQ preset for a GEMS mood (no dialogue protection)."""
    if mood_name not in EQ_PRESET_TABLE_DB:
        raise ValueError(f"Unknown mood '{mood_name}'")
    gains = EQ_PRESET_TABLE_DB[mood_name]
    if len(gains) != len(BAND_SPECS):
        raise ValueError(f"Preset row length mismatch for {mood_name}")
    return [
        EQBand(freq_hz=freq, gain_db=gain, q=q)
        for (freq, q), gain in zip(BAND_SPECS, gains)
    ]


def apply_dialogue_protection(
    original_bands: list[EQBand],
    dialogue_density: float,
    alpha_d: float = DEFAULT_ALPHA_D,
) -> list[EQBand]:
    """Attenuate voice-critical bands (B6/B7/B8) proportional to dialogue density.

    Formula (spec §5-7):
        g_eff = g_orig × (1 - (1 - α_d) · density)

    - density = 0 → unchanged (1.0× multiplier)
    - density = 1 → g_orig × α_d  (maximum protection)
    - Sign preserved (cuts stay cuts, boosts stay boosts, just reduced in magnitude)
    - 0 dB bands naturally stay 0 dB (multiplication by 1.0)
    """
    if not 0.0 <= dialogue_density <= 1.0:
        raise ValueError(f"dialogue_density must be in [0, 1], got {dialogue_density}")
    mult = 1.0 - (1.0 - alpha_d) * dialogue_density
    out: list[EQBand] = []
    for i, b in enumerate(original_bands):
        if i in VOICE_PROTECTED_BAND_INDICES:
            out.append(EQBand(freq_hz=b.freq_hz, gain_db=b.gain_db * mult, q=b.q))
        else:
            out.append(b)
    return out
