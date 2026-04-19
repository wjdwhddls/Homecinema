"""Scene-boundary-reset EMA smoothing + scene-level V/A aggregation.

Spec V3.3 §5-4 (= V3.2 §5-6/§5-7):
    - Apply EMA to within-scene V/A sequence: ema_t = α·v_t + (1-α)·ema_{t-1}
    - At scene boundaries, reset EMA to raw prediction (do NOT smooth across
      director-intended cuts).
    - Cold start: first 2-3 windows of each scene keep raw values (instead of
      ramping from 0) — prevents EMA underestimating the scene's initial mood.
    - Scene-level output = mean of EMA-smoothed V/A over the scene's windows.
"""

from __future__ import annotations

from .types import SceneVA, WindowVA

DEFAULT_ALPHA = 0.3
COLD_START_WINDOWS = 3


def apply_ema_within_scenes(
    windows: list[WindowVA],
    alpha: float = DEFAULT_ALPHA,
    cold_start: int = COLD_START_WINDOWS,
) -> list[WindowVA]:
    """Return a new list of WindowVA with EMA applied within each scene.

    The returned dataclasses preserve scene_idx / window_idx_in_scene / times
    and gate_w values; only valence/arousal are modified.
    """
    if not windows:
        return []
    out: list[WindowVA] = []
    current_scene = None
    ema_v = 0.0
    ema_a = 0.0
    idx_in_scene = 0
    for w in windows:
        if w.scene_idx != current_scene:
            # Reset on scene boundary
            current_scene = w.scene_idx
            idx_in_scene = 0
            ema_v = w.valence
            ema_a = w.arousal
        else:
            if idx_in_scene < cold_start:
                # cold start: use raw value
                ema_v = w.valence
                ema_a = w.arousal
            else:
                ema_v = alpha * w.valence + (1.0 - alpha) * ema_v
                ema_a = alpha * w.arousal + (1.0 - alpha) * ema_a
        out.append(WindowVA(
            scene_idx=w.scene_idx,
            window_idx_in_scene=w.window_idx_in_scene,
            start_sec=w.start_sec,
            end_sec=w.end_sec,
            valence=ema_v,
            arousal=ema_a,
            gate_w_v=w.gate_w_v,
            gate_w_a=w.gate_w_a,
        ))
        idx_in_scene += 1
    return out


def aggregate_by_scene(
    smoothed_windows: list[WindowVA],
    scene_bounds: dict[int, tuple[float, float]] | None = None,
) -> list[SceneVA]:
    """Mean-pool EMA-smoothed per-window V/A to scene-level V/A.

    Args:
        smoothed_windows: EMA-smoothed window outputs.
        scene_bounds: optional {scene_idx: (start_sec, end_sec)} to carry
            accurate scene time ranges. If omitted, derived from min/max of
            window times (less accurate for short clamped windows).
    """
    if not smoothed_windows:
        return []
    # Group by scene_idx preserving order
    groups: dict[int, list[WindowVA]] = {}
    order: list[int] = []
    for w in smoothed_windows:
        if w.scene_idx not in groups:
            groups[w.scene_idx] = []
            order.append(w.scene_idx)
        groups[w.scene_idx].append(w)

    out: list[SceneVA] = []
    for sidx in order:
        group = groups[sidx]
        mean_v = sum(w.valence for w in group) / len(group)
        mean_a = sum(w.arousal for w in group) / len(group)
        mean_gv = sum(w.gate_w_v for w in group) / len(group)
        mean_ga = sum(w.gate_w_a for w in group) / len(group)
        if scene_bounds and sidx in scene_bounds:
            start_s, end_s = scene_bounds[sidx]
        else:
            start_s = min(w.start_sec for w in group)
            end_s = max(w.end_sec for w in group)
        out.append(SceneVA(
            scene_idx=sidx,
            start_sec=start_s,
            end_sec=end_s,
            valence=mean_v,
            arousal=mean_a,
            mean_gate_w_v=mean_gv,
            mean_gate_w_a=mean_ga,
        ))
    return out
