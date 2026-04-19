"""Shared dataclasses for the inference pipeline.

Each stage produces or consumes these — keeps module interfaces explicit and
enables isolated unit testing without mocking torch/ffmpeg dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scene:
    """One scene in the video (post-merge)."""
    idx: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class Window:
    """One 4s analysis window within a scene."""
    scene_idx: int
    window_idx_in_scene: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class WindowVA:
    """Per-window model output."""
    scene_idx: int
    window_idx_in_scene: int
    start_sec: float
    end_sec: float
    valence: float
    arousal: float
    gate_w_v: float
    gate_w_a: float


@dataclass(frozen=True)
class SceneVA:
    """Per-scene aggregated (EMA-smoothed, then mean-pooled) V/A."""
    scene_idx: int
    start_sec: float
    end_sec: float
    valence: float
    arousal: float
    mean_gate_w_v: float
    mean_gate_w_a: float


@dataclass(frozen=True)
class SpeechSegment:
    """One continuous speech region from Silero VAD (absolute seconds)."""
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class EQBand:
    """One EQ band spec."""
    freq_hz: float
    gain_db: float
    q: float


@dataclass(frozen=True)
class SceneEQ:
    """Final per-scene EQ decision (original + dialogue-protected)."""
    scene_idx: int
    start_sec: float
    end_sec: float
    valence: float
    arousal: float
    mood: str                       # GEMS name: Tension / Sadness / ...
    mood_idx: int                   # 0..6
    dialogue_density: float
    original_bands: list[EQBand]
    effective_bands: list[EQBand]
