"""Dialogue density overlap math."""

from model.autoEQ.infer_pseudo.dialogue_density import (
    compute_all_scene_densities,
    compute_scene_dialogue_density,
)
from model.autoEQ.infer_pseudo.types import Scene, SpeechSegment


def test_no_speech_yields_zero():
    scene = Scene(idx=0, start_sec=0.0, end_sec=10.0)
    assert compute_scene_dialogue_density(scene, []) == 0.0


def test_full_coverage_yields_one():
    scene = Scene(idx=0, start_sec=0.0, end_sec=10.0)
    segs = [SpeechSegment(0.0, 10.0)]
    assert compute_scene_dialogue_density(scene, segs) == 1.0


def test_partial_coverage_proportional():
    scene = Scene(idx=0, start_sec=0.0, end_sec=10.0)
    segs = [SpeechSegment(2.0, 5.0), SpeechSegment(7.0, 8.0)]
    # 3s + 1s = 4s / 10s = 0.4
    assert abs(compute_scene_dialogue_density(scene, segs) - 0.4) < 1e-6


def test_speech_outside_scene_ignored():
    scene = Scene(idx=0, start_sec=10.0, end_sec=20.0)
    segs = [SpeechSegment(0.0, 5.0), SpeechSegment(25.0, 30.0)]
    assert compute_scene_dialogue_density(scene, segs) == 0.0


def test_speech_partially_overlapping_scene_start():
    scene = Scene(idx=0, start_sec=5.0, end_sec=15.0)
    # Speech 3-8: overlaps 5-8 = 3s / 10s = 0.3
    segs = [SpeechSegment(3.0, 8.0)]
    assert abs(compute_scene_dialogue_density(scene, segs) - 0.3) < 1e-6


def test_speech_partially_overlapping_scene_end():
    scene = Scene(idx=0, start_sec=0.0, end_sec=10.0)
    # Speech 8-15: overlaps 8-10 = 2s / 10s = 0.2
    segs = [SpeechSegment(8.0, 15.0)]
    assert abs(compute_scene_dialogue_density(scene, segs) - 0.2) < 1e-6


def test_density_clamped_to_one():
    # Adversarial: overlapping speech segments summing > scene duration
    # (shouldn't happen from VAD but defensively clamp)
    scene = Scene(idx=0, start_sec=0.0, end_sec=10.0)
    segs = [SpeechSegment(0.0, 10.0), SpeechSegment(0.0, 10.0)]  # double-counted
    d = compute_scene_dialogue_density(scene, segs)
    assert d <= 1.0


def test_zero_duration_scene_returns_zero():
    scene = Scene(idx=0, start_sec=5.0, end_sec=5.0)
    segs = [SpeechSegment(0.0, 10.0)]
    assert compute_scene_dialogue_density(scene, segs) == 0.0


def test_all_scene_densities_returns_mapping():
    scenes = [
        Scene(idx=0, start_sec=0.0, end_sec=10.0),
        Scene(idx=1, start_sec=10.0, end_sec=20.0),
    ]
    segs = [SpeechSegment(5.0, 15.0)]  # 5s in scene 0, 5s in scene 1
    d = compute_all_scene_densities(scenes, segs)
    assert d == {0: 0.5, 1: 0.5}
