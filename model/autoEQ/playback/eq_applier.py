"""Per-scene 10-band biquad EQ application via pedalboard.

Spec V3.3 §5-9: input is the `eq_preset.effective_bands` list from timeline.json
(already dialogue-protected). Each band is a biquad peaking filter; the chain
processes one scene's audio slice.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_eq_chain_from_bands", "apply_scene_eq_to_audio"]


def build_eq_chain_from_bands(bands: list[dict]):
    """Construct a pedalboard.Pedalboard chain from timeline band dicts.

    Each band dict: {"freq_hz": float, "gain_db": float, "q": float}
    """
    from pedalboard import Pedalboard, PeakFilter

    filters = [
        PeakFilter(
            cutoff_frequency_hz=float(b["freq_hz"]),
            gain_db=float(b["gain_db"]),
            q=float(b["q"]),
        )
        for b in bands
    ]
    return Pedalboard(filters)


def apply_scene_eq_to_audio(
    audio: np.ndarray,
    sample_rate: int,
    bands: list[dict],
) -> np.ndarray:
    """Apply the 10-band EQ chain to a scene's audio region.

    Args:
        audio: shape (T,) or (T, C), float32 preferred. Pedalboard handles both.
        sample_rate: audio sample rate (e.g. 48000).
        bands: list of band dicts as described above.

    Returns:
        Processed array, same shape as input.
    """
    if audio.size == 0:
        return audio.astype(np.float32, copy=True)
    x = np.asarray(audio, dtype=np.float32)
    chain = build_eq_chain_from_bands(bands)
    # pedalboard expects (channels, samples) OR (samples, channels) OR (samples,).
    # Docs: When given (num_samples, num_channels) it processes columns.
    # To be safe use explicit 2D (num_samples, num_channels) handling.
    if x.ndim == 1:
        y = chain(x, sample_rate=sample_rate)
        return np.asarray(y, dtype=np.float32)
    # (T, C)
    y = chain(x, sample_rate=sample_rate)
    return np.asarray(y, dtype=np.float32)
