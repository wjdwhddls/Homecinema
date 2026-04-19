"""Scene → window sliding correctness."""

from model.autoEQ.infer_pseudo.types import Scene
from model.autoEQ.infer_pseudo.window_slider import (
    slide_windows_all_scenes,
    slide_windows_in_scene,
)


def test_15s_scene_yields_12_windows():
    # 15s scene with 4s window, 1s stride: starts at 0,1,2,...,11 = 12 windows
    scene = Scene(idx=0, start_sec=0.0, end_sec=15.0)
    wins = slide_windows_in_scene(scene)
    assert len(wins) == 12
    assert wins[0].start_sec == 0.0 and wins[0].end_sec == 4.0
    assert wins[-1].start_sec == 11.0 and wins[-1].end_sec == 15.0


def test_4s_scene_yields_single_window_covering_full_scene():
    scene = Scene(idx=0, start_sec=10.0, end_sec=14.0)
    wins = slide_windows_in_scene(scene)
    assert len(wins) == 1
    assert wins[0].start_sec == 10.0 and wins[0].end_sec == 14.0


def test_short_scene_below_window_extends_forward():
    # Scene shorter than window: single window of fixed 4s duration,
    # starting at scene.start (extends past scene.end — model requires 4s).
    # Boundary check (video end) happens later in model_inference.
    scene = Scene(idx=0, start_sec=100.0, end_sec=102.0)  # 2s
    wins = slide_windows_in_scene(scene)
    assert len(wins) == 1
    assert wins[0].start_sec == 100.0
    assert wins[0].end_sec == 104.0  # 4s duration preserved


def test_odd_length_scene_tail_coverage():
    # 4.5s scene: starts at 0 and 0.5? Actually stride=1 so starts at 0 only,
    # then tail-clamp adds start=0.5 → 2 windows
    scene = Scene(idx=0, start_sec=0.0, end_sec=4.5)
    wins = slide_windows_in_scene(scene)
    # Last window must cover up to 4.5s
    assert wins[-1].end_sec == 4.5
    assert wins[-1].start_sec == 0.5


def test_scene_idx_and_window_idx_preserved():
    scenes = [
        Scene(idx=0, start_sec=0.0, end_sec=10.0),
        Scene(idx=1, start_sec=10.0, end_sec=16.0),
    ]
    wins = slide_windows_all_scenes(scenes)
    # Scene 0: starts 0..6 = 7 windows
    # Scene 1: starts 10, 11, 12 = 3 windows
    s0 = [w for w in wins if w.scene_idx == 0]
    s1 = [w for w in wins if w.scene_idx == 1]
    assert len(s0) == 7 and len(s1) == 3
    assert [w.window_idx_in_scene for w in s0] == list(range(7))
    assert [w.window_idx_in_scene for w in s1] == list(range(3))


def test_empty_scene_list_returns_empty():
    assert slide_windows_all_scenes([]) == []


def test_zero_duration_scene_ignored():
    scene = Scene(idx=0, start_sec=5.0, end_sec=5.0)
    assert slide_windows_in_scene(scene) == []
