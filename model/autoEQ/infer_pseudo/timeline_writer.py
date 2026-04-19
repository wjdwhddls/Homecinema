"""JSON timeline serialization per spec V3.3 §5-8 (= V3.2 §5-11).

Schema version 1.0:
    schema_version: "1.0"
    metadata: {video, duration_sec, model_version, analyzed_at, ...}
    config:   {window_sec, stride_sec, ema_alpha, alpha_d, ...}
    scenes:   [ { start_sec, end_sec, va, mood, dialogue, eq_preset, windows? } ]
    global:   { mean_va, mood_distribution, avg_dialogue_density }

`windows` is optional and omitted by default (file size control); enable with
`include_windows=True`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .types import SceneEQ, SceneVA, SpeechSegment, WindowVA

SCHEMA_VERSION = "1.0"


def _band_to_dict(b) -> dict:
    return {"freq_hz": b.freq_hz, "gain_db": b.gain_db, "q": b.q}


def _speech_segments_in_scene(
    segments: list[SpeechSegment],
    start_sec: float,
    end_sec: float,
) -> list[list[float]]:
    """Return list of [rel_start, rel_end] pairs inside the given scene window."""
    out: list[list[float]] = []
    for seg in segments:
        o_start = max(start_sec, seg.start_sec)
        o_end = min(end_sec, seg.end_sec)
        if o_end > o_start:
            out.append([o_start - start_sec, o_end - start_sec])
    return out


def build_timeline_dict(
    *,
    video_path: str,
    duration_sec: float,
    scene_va: list[SceneVA],
    scene_eq: list[SceneEQ],
    speech_segments: list[SpeechSegment],
    scene_windows: dict[int, list[WindowVA]] | None,
    model_version: str,
    config: dict,
    include_windows: bool = False,
) -> dict:
    """Assemble the full timeline dict.

    scene_va and scene_eq must share scene_idx ordering.
    """
    # Index for alignment
    va_by_idx = {s.scene_idx: s for s in scene_va}
    eq_by_idx = {s.scene_idx: s for s in scene_eq}
    scene_indices = sorted(set(va_by_idx.keys()) | set(eq_by_idx.keys()))

    scenes_payload = []
    for sidx in scene_indices:
        va = va_by_idx.get(sidx)
        eq = eq_by_idx.get(sidx)
        if va is None or eq is None:
            continue
        dialogue_segments = _speech_segments_in_scene(
            speech_segments, eq.start_sec, eq.end_sec
        )
        scene_entry: dict = {
            "scene_idx": sidx,
            "start_sec": eq.start_sec,
            "end_sec": eq.end_sec,
            "duration_sec": eq.end_sec - eq.start_sec,
            "va": {
                "valence": va.valence,
                "arousal": va.arousal,
            },
            "gate": {
                "mean_w_v": va.mean_gate_w_v,
                "mean_w_a": va.mean_gate_w_a,
            },
            "mood": {
                "name": eq.mood,
                "idx": eq.mood_idx,
            },
            "dialogue": {
                "density": eq.dialogue_density,
                "segments_rel": dialogue_segments,
            },
            "eq_preset": {
                "original_bands": [_band_to_dict(b) for b in eq.original_bands],
                "effective_bands": [_band_to_dict(b) for b in eq.effective_bands],
            },
        }
        if include_windows and scene_windows is not None:
            ws = scene_windows.get(sidx, [])
            scene_entry["windows"] = [
                {
                    "window_idx": w.window_idx_in_scene,
                    "start_sec": w.start_sec,
                    "end_sec": w.end_sec,
                    "valence": w.valence,
                    "arousal": w.arousal,
                    "gate_w_v": w.gate_w_v,
                    "gate_w_a": w.gate_w_a,
                }
                for w in ws
            ]
        scenes_payload.append(scene_entry)

    # Global stats
    n = max(1, len(scene_va))
    mean_v = sum(s.valence for s in scene_va) / n
    mean_a = sum(s.arousal for s in scene_va) / n
    mood_dist: dict[str, int] = {}
    for eq in scene_eq:
        mood_dist[eq.mood] = mood_dist.get(eq.mood, 0) + 1
    avg_density = sum(eq.dialogue_density for eq in scene_eq) / max(1, len(scene_eq))

    timeline = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "video": str(video_path),
            "duration_sec": duration_sec,
            "model_version": model_version,
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_scenes": len(scenes_payload),
        },
        "config": config,
        "scenes": scenes_payload,
        "global": {
            "mean_va": {"valence": mean_v, "arousal": mean_a},
            "mood_distribution": mood_dist,
            "avg_dialogue_density": avg_density,
        },
    }
    return timeline


def write_timeline(
    output_path: Path,
    timeline: dict,
) -> None:
    """Write timeline dict to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
