"""Timeline JSON schema and content correctness."""

import json

from model.autoEQ.infer_pseudo.eq_preset import (
    apply_dialogue_protection, get_original_bands,
)
from model.autoEQ.infer_pseudo.timeline_writer import (
    SCHEMA_VERSION, build_timeline_dict, write_timeline,
)
from model.autoEQ.infer_pseudo.types import (
    SceneEQ, SceneVA, SpeechSegment, WindowVA,
)


def _fake_inputs():
    # Two scenes: 0 (0-10s, Wonder mood, dialogue), 1 (10-20s, Tension, no dialogue)
    scene_va = [
        SceneVA(scene_idx=0, start_sec=0.0, end_sec=10.0,
                valence=0.3, arousal=0.1,
                mean_gate_w_v=0.6, mean_gate_w_a=0.4),
        SceneVA(scene_idx=1, start_sec=10.0, end_sec=20.0,
                valence=-0.5, arousal=0.7,
                mean_gate_w_v=0.4, mean_gate_w_a=0.6),
    ]
    wonder_orig = get_original_bands("Wonder")
    tension_orig = get_original_bands("Tension")
    scene_eq = [
        SceneEQ(
            scene_idx=0, start_sec=0.0, end_sec=10.0,
            valence=0.3, arousal=0.1,
            mood="Wonder", mood_idx=6, dialogue_density=0.8,
            original_bands=wonder_orig,
            effective_bands=apply_dialogue_protection(wonder_orig, 0.8),
        ),
        SceneEQ(
            scene_idx=1, start_sec=10.0, end_sec=20.0,
            valence=-0.5, arousal=0.7,
            mood="Tension", mood_idx=0, dialogue_density=0.0,
            original_bands=tension_orig,
            effective_bands=apply_dialogue_protection(tension_orig, 0.0),
        ),
    ]
    speech = [SpeechSegment(1.0, 9.0)]  # 8s in scene 0
    scene_windows = {
        0: [WindowVA(0, 0, 0.0, 4.0, 0.3, 0.1, 0.6, 0.4)],
        1: [WindowVA(1, 0, 10.0, 14.0, -0.5, 0.7, 0.4, 0.6)],
    }
    return scene_va, scene_eq, speech, scene_windows


def test_timeline_has_required_top_level_keys():
    scene_va, scene_eq, speech, _ = _fake_inputs()
    tl = build_timeline_dict(
        video_path="/fake/movie.mp4", duration_sec=20.0,
        scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None,
        model_version="train_pseudo_v3.3", config={"window_sec": 4},
    )
    assert tl["schema_version"] == SCHEMA_VERSION
    for k in ["metadata", "config", "scenes", "global"]:
        assert k in tl


def test_scenes_in_ascending_order():
    scene_va, scene_eq, speech, _ = _fake_inputs()
    tl = build_timeline_dict(
        video_path="movie.mp4", duration_sec=20.0,
        scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None,
        model_version="v", config={},
    )
    starts = [s["start_sec"] for s in tl["scenes"]]
    assert starts == sorted(starts)


def test_each_scene_has_eq_original_and_effective():
    scene_va, scene_eq, speech, _ = _fake_inputs()
    tl = build_timeline_dict(
        video_path="movie.mp4", duration_sec=20.0,
        scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None,
        model_version="v", config={},
    )
    for s in tl["scenes"]:
        assert "original_bands" in s["eq_preset"]
        assert "effective_bands" in s["eq_preset"]
        assert len(s["eq_preset"]["original_bands"]) == 10
        assert len(s["eq_preset"]["effective_bands"]) == 10


def test_dialogue_density_differs_between_scenes():
    scene_va, scene_eq, speech, _ = _fake_inputs()
    tl = build_timeline_dict(
        video_path="m.mp4", duration_sec=20, scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None, model_version="v", config={},
    )
    s0 = next(s for s in tl["scenes"] if s["scene_idx"] == 0)
    s1 = next(s for s in tl["scenes"] if s["scene_idx"] == 1)
    assert s0["dialogue"]["density"] == 0.8
    assert s1["dialogue"]["density"] == 0.0


def test_effective_bands_match_protection_formula():
    scene_va, scene_eq, speech, _ = _fake_inputs()
    tl = build_timeline_dict(
        video_path="m.mp4", duration_sec=20, scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None, model_version="v", config={},
    )
    s0 = next(s for s in tl["scenes"] if s["scene_idx"] == 0)
    # Wonder B7 original = +1.0; density=0.8, α_d=0.5 → eff = 1.0 × (1 - 0.5·0.8) = 0.6
    assert abs(s0["eq_preset"]["effective_bands"][6]["gain_db"] - 0.6) < 1e-6


def test_include_windows_flag_controls_output_size():
    scene_va, scene_eq, speech, windows = _fake_inputs()
    tl_no = build_timeline_dict(
        video_path="m.mp4", duration_sec=20, scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=windows, model_version="v", config={},
        include_windows=False,
    )
    tl_yes = build_timeline_dict(
        video_path="m.mp4", duration_sec=20, scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=windows, model_version="v", config={},
        include_windows=True,
    )
    assert "windows" not in tl_no["scenes"][0]
    assert "windows" in tl_yes["scenes"][0]


def test_global_mood_distribution_counts():
    scene_va, scene_eq, speech, _ = _fake_inputs()
    tl = build_timeline_dict(
        video_path="m.mp4", duration_sec=20, scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None, model_version="v", config={},
    )
    md = tl["global"]["mood_distribution"]
    assert md["Wonder"] == 1
    assert md["Tension"] == 1


def test_dialogue_segments_are_relative_to_scene_start():
    scene_va, scene_eq, speech, _ = _fake_inputs()
    # Scene 0: 0-10s, speech 1-9s → relative: [1, 9]
    tl = build_timeline_dict(
        video_path="m.mp4", duration_sec=20, scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None, model_version="v", config={},
    )
    s0 = next(s for s in tl["scenes"] if s["scene_idx"] == 0)
    assert s0["dialogue"]["segments_rel"] == [[1.0, 9.0]]


def test_write_timeline_produces_valid_json(tmp_path):
    scene_va, scene_eq, speech, _ = _fake_inputs()
    tl = build_timeline_dict(
        video_path="m.mp4", duration_sec=20, scene_va=scene_va, scene_eq=scene_eq,
        speech_segments=speech, scene_windows=None, model_version="v", config={},
    )
    out = tmp_path / "t.json"
    write_timeline(out, tl)
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert len(loaded["scenes"]) == 2
