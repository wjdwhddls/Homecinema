"""Short-scene merge logic (pure; detect_scenes_raw needs actual video)."""

from model.autoEQ.infer_pseudo.scene_detect import merge_short_scenes


def test_no_merge_when_all_long_enough():
    raw = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
    merged = merge_short_scenes(raw, min_duration_sec=2.0)
    assert len(merged) == 3
    assert [s.start_sec for s in merged] == [0.0, 5.0, 10.0]
    assert [s.idx for s in merged] == [0, 1, 2]


def test_short_first_scene_merged_into_next():
    raw = [(0.0, 1.0), (1.0, 10.0), (10.0, 20.0)]  # first is 1s
    merged = merge_short_scenes(raw, min_duration_sec=2.0)
    assert len(merged) == 2
    assert merged[0].start_sec == 0.0 and merged[0].end_sec == 10.0


def test_short_last_scene_merged_into_prev():
    raw = [(0.0, 10.0), (10.0, 20.0), (20.0, 20.5)]  # last is 0.5s
    merged = merge_short_scenes(raw, min_duration_sec=2.0)
    assert len(merged) == 2
    assert merged[-1].start_sec == 10.0 and merged[-1].end_sec == 20.5


def test_short_middle_merged_into_shorter_neighbor():
    # Short in middle: prev=5s, next=3s → merged into next (shorter)
    raw = [(0.0, 5.0), (5.0, 6.0), (6.0, 9.0)]  # middle is 1s, next=3s shorter than prev=5s
    merged = merge_short_scenes(raw, min_duration_sec=2.0)
    assert len(merged) == 2
    # Merged with next: first stays [0,5], second [5,9]
    assert merged[0].start_sec == 0.0 and merged[0].end_sec == 5.0
    assert merged[1].start_sec == 5.0 and merged[1].end_sec == 9.0


def test_all_short_collapse_to_one():
    # All scenes are 0.5s → will merge repeatedly until one remains
    raw = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    merged = merge_short_scenes(raw, min_duration_sec=2.0)
    assert len(merged) == 1
    assert merged[0].start_sec == 0.0 and merged[0].end_sec == 2.0


def test_idx_is_sequential_post_merge():
    raw = [(0.0, 1.0), (1.0, 5.0), (5.0, 5.5), (5.5, 10.0)]
    merged = merge_short_scenes(raw, min_duration_sec=2.0)
    assert [s.idx for s in merged] == list(range(len(merged)))


def test_empty_input():
    assert merge_short_scenes([]) == []


def test_single_scene_passes_through_unchanged():
    raw = [(0.0, 0.5)]
    merged = merge_short_scenes(raw, min_duration_sec=2.0)
    # Can't merge single scene; returns it as-is
    assert len(merged) == 1
    assert merged[0].start_sec == 0.0 and merged[0].end_sec == 0.5
