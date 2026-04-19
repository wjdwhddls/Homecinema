"""Scene → sliding windows.

Per spec V3.3 §5-2: 4-second windows with 1-second stride inside each scene.
If the last window would extend past scene end, clamp it to end-4s (ensures
final window is fully covered).
"""

from __future__ import annotations

from .types import Scene, Window

WINDOW_SEC = 4.0
STRIDE_SEC = 1.0


def slide_windows_in_scene(
    scene: Scene,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
) -> list[Window]:
    """Emit windows for a single scene.

    Behavior:
      - Scenes shorter than `window_sec` → single window clamped to scene end
        (never yields 0 windows even for a 2.0s scene; in that case the window
        starts at max(scene.start, end - window_sec)).
      - Last window is guaranteed to cover up to `scene.end_sec` by clamping.
    """
    if scene.duration_sec <= 0:
        return []

    windows: list[Window] = []
    # Generate stride-based start times
    starts: list[float] = []
    t = scene.start_sec
    max_start = scene.end_sec - window_sec
    if max_start < scene.start_sec:
        # scene shorter than window → one clamped window
        starts.append(max(scene.start_sec, scene.end_sec - window_sec))
    else:
        while t <= max_start + 1e-6:
            starts.append(t)
            t += stride_sec
        # Ensure we cover the tail: last window ends ≥ scene.end_sec
        if starts[-1] < max_start - 1e-6:
            starts.append(max_start)

    for i, s in enumerate(starts):
        end = s + window_sec
        # Do not clip below scene_end if scene is shorter; but keep window_sec duration
        windows.append(Window(
            scene_idx=scene.idx,
            window_idx_in_scene=i,
            start_sec=s,
            end_sec=end,
        ))
    return windows


def slide_windows_all_scenes(
    scenes: list[Scene],
    window_sec: float = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
) -> list[Window]:
    """Emit windows for every scene, preserving order."""
    out: list[Window] = []
    for s in scenes:
        out.extend(slide_windows_in_scene(s, window_sec, stride_sec))
    return out
