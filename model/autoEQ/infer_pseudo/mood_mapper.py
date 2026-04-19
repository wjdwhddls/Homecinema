"""V/A scalar pair → 7 GEMS mood (Euclidean distance to MOOD_CENTERS).

Spec V3.3 §5-5: EQ preset selection uses V/A regression output only — Mood
Head's K=4 quadrant output is not consulted here. Reuses MOOD_CENTERS from
`model.autoEQ.train.dataset` (defined per spec V3.2 §2-5, preserved in V3.3).
"""

from __future__ import annotations

from ..train.dataset import MOOD_CENTERS, va_to_mood

GEMS_LABELS: list[str] = [
    "Tension",           # 0 — (-0.6, +0.7)
    "Sadness",           # 1 — (-0.6, -0.4)
    "Peacefulness",      # 2 — (+0.5, -0.5)
    "JoyfulActivation",  # 3 — (+0.7, +0.6)
    "Tenderness",        # 4 — (+0.4, -0.2)
    "Power",             # 5 — (+0.2, +0.8)
    "Wonder",            # 6 — (+0.5, +0.3)
]

assert len(GEMS_LABELS) == MOOD_CENTERS.shape[0], (
    "GEMS_LABELS must align 1:1 with MOOD_CENTERS rows"
)


def va_to_mood_name(valence: float, arousal: float) -> tuple[int, str]:
    """Return (mood_idx, mood_name) for the given V/A coordinate."""
    idx = va_to_mood(valence, arousal)
    return idx, GEMS_LABELS[idx]
