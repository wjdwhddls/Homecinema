"""Sanity checks on dataclass shapes."""

from model.autoEQ.infer_pseudo.types import (
    EQBand, Scene, SceneEQ, SceneVA, SpeechSegment, Window, WindowVA,
)


def test_scene_duration_positive():
    s = Scene(idx=0, start_sec=0.0, end_sec=4.5)
    assert s.duration_sec == 4.5


def test_scene_is_frozen():
    s = Scene(idx=0, start_sec=0.0, end_sec=4.0)
    try:
        s.start_sec = 1.0  # type: ignore[misc]
    except Exception:
        return
    assert False, "Scene should be frozen"


def test_window_window_pair_fields():
    w = Window(scene_idx=1, window_idx_in_scene=0, start_sec=0.0, end_sec=4.0)
    assert w.scene_idx == 1 and w.end_sec - w.start_sec == 4.0


def test_eq_band_and_scene_eq_assembly():
    bands = [EQBand(freq_hz=1000, gain_db=1.0, q=1.4)]
    seq = SceneEQ(
        scene_idx=0, start_sec=0.0, end_sec=10.0,
        valence=0.3, arousal=0.1,
        mood="Wonder", mood_idx=6, dialogue_density=0.5,
        original_bands=bands, effective_bands=bands,
    )
    assert seq.mood == "Wonder"
    assert len(seq.original_bands) == 1
