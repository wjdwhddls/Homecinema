"""Scene detection + short-scene merging.

Spec V3.3 §5-1 (= V3.2 §5-2, §5-3):
    - PySceneDetect ContentDetector, threshold=27.0, min_scene_len=1s.
    - Short-scene merge: scenes < min_duration (default 2s) are folded into the
      shorter adjacent scene. Repeats until all scenes ≥ min_duration OR only
      one scene remains, with MAX_MERGE_ITERATIONS safeguard.
"""

from __future__ import annotations

from pathlib import Path

from .types import Scene

DEFAULT_THRESHOLD = 27.0
DEFAULT_MIN_SCENE_LEN_SEC = 1.0
DEFAULT_MIN_DURATION_SEC = 2.0
MAX_MERGE_ITERATIONS = 1000


def detect_scenes_raw(
    video_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len_sec: float = DEFAULT_MIN_SCENE_LEN_SEC,
) -> list[tuple[float, float]]:
    """Run PySceneDetect and return [(start_sec, end_sec), ...].

    Raises ImportError if PySceneDetect not installed.
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    # min_scene_len is in frames in newer API; compute from fps
    fps = video.frame_rate if video.frame_rate else 24.0
    min_scene_len_frames = max(1, int(round(min_scene_len_sec * fps)))
    sm = SceneManager()
    sm.add_detector(ContentDetector(
        threshold=threshold, min_scene_len=min_scene_len_frames
    ))
    sm.detect_scenes(video=video)
    scene_list = sm.get_scene_list()
    if not scene_list:
        # If no cuts detected, treat whole video as one scene
        duration_sec = float(video.duration.get_seconds()) if video.duration else 0.0
        return [(0.0, duration_sec)]
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def merge_short_scenes(
    raw_scenes: list[tuple[float, float]],
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> list[Scene]:
    """Iteratively merge scenes shorter than `min_duration_sec` into their
    shorter neighbor. Returns a list of Scene dataclasses with stable idx.
    """
    if not raw_scenes:
        return []
    # Work with tuples for in-place-like mutation
    scenes = [list(s) for s in raw_scenes]  # [[start, end], ...]
    iteration = 0
    while True:
        if all((e - s) >= min_duration_sec for s, e in scenes):
            break
        if len(scenes) <= 1:
            break
        iteration += 1
        if iteration > MAX_MERGE_ITERATIONS:
            break
        # Find shortest
        durations = [e - s for s, e in scenes]
        shortest_idx = durations.index(min(durations))
        # Pick merge target: shorter neighbor
        if shortest_idx == 0:
            merge_target = 1
        elif shortest_idx == len(scenes) - 1:
            merge_target = len(scenes) - 2
        else:
            prev_dur = durations[shortest_idx - 1]
            next_dur = durations[shortest_idx + 1]
            merge_target = (shortest_idx - 1
                            if prev_dur <= next_dur else shortest_idx + 1)
        # Merge shortest into neighbor (combine start/end)
        lo = min(shortest_idx, merge_target)
        hi = max(shortest_idx, merge_target)
        new_start = scenes[lo][0]
        new_end = scenes[hi][1]
        scenes[lo] = [new_start, new_end]
        del scenes[hi]

    return [Scene(idx=i, start_sec=s, end_sec=e) for i, (s, e) in enumerate(scenes)]


def detect_and_merge(
    video_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    min_scene_len_sec: float = DEFAULT_MIN_SCENE_LEN_SEC,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
) -> list[Scene]:
    """Full pipeline: detect + merge. Returns [Scene(idx, start, end), ...]."""
    raw = detect_scenes_raw(video_path, threshold, min_scene_len_sec)
    return merge_short_scenes(raw, min_duration_sec)
