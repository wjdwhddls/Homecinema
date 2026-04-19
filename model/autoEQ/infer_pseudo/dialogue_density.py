"""Join scene boundaries with Silero VAD speech segments → per-scene density.

Spec V3.3 §5-6 (= V3.2 §5-8-4). Pure function, no I/O.
"""

from __future__ import annotations

from .types import Scene, SpeechSegment


def compute_scene_dialogue_density(
    scene: Scene,
    speech_segments: list[SpeechSegment],
) -> float:
    """Return fraction of scene duration covered by speech segments.

    Overlap of each speech segment with the scene [start, end] is summed.
    Clamped to [0, 1].
    """
    if scene.duration_sec <= 0:
        return 0.0
    total_speech_sec = 0.0
    for seg in speech_segments:
        overlap_start = max(scene.start_sec, seg.start_sec)
        overlap_end = min(scene.end_sec, seg.end_sec)
        if overlap_end > overlap_start:
            total_speech_sec += overlap_end - overlap_start
    density = total_speech_sec / scene.duration_sec
    return max(0.0, min(1.0, density))


def compute_all_scene_densities(
    scenes: list[Scene],
    speech_segments: list[SpeechSegment],
) -> dict[int, float]:
    """Compute density for every scene; returns {scene_idx: density}."""
    return {s.idx: compute_scene_dialogue_density(s, speech_segments) for s in scenes}
