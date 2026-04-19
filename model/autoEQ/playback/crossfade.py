"""Scene-boundary raised-cosine crossfade.

Spec V3.3 §5-9 (= V3.2 §5-13): at each scene boundary, blend the EQ-processed
audio from the preceding scene with the EQ-processed audio of the following
scene over `crossfade_ms` (default 300ms) using a raised-cosine envelope.

This eliminates the "click" / spectral discontinuity that arises when two
different EQ filter chains are spliced. Output `y[n] = (1-w[n]) · prev[n] + w[n] · next[n]`
where `w` ramps 0 → 1 following a raised-cosine shape.

Implementation notes:
  - Applies in-place style on a single processed audio array.
  - Requires both sides of each boundary to be already EQ'd independently —
    callers should process each scene with its own pedalboard chain, then pass
    both versions here for blending.
  - Edge cases: first and last boundary half-zones may be trimmed; fade length
    is capped at min(crossfade_samples, boundary_distance / 2).
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_CROSSFADE_MS = 300


def raised_cosine_crossfade(n: int) -> np.ndarray:
    """Raised-cosine envelope of length n, rising 0 → 1.

    Formula: w[i] = 0.5 · (1 − cos(π · i / (n-1)))  for i in 0..n-1

    - w[0] = 0.0, w[n-1] = 1.0
    - Smooth, zero derivative at both ends → no spectral splatter.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.array([1.0], dtype=np.float32)
    i = np.arange(n, dtype=np.float32)
    w = 0.5 * (1.0 - np.cos(math.pi * i / (n - 1)))
    return w.astype(np.float32)


def apply_boundary_crossfades(
    scene_audio: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
) -> np.ndarray:
    """Concatenate scene audio arrays with raised-cosine crossfades at each boundary.

    Args:
        scene_audio: list of np.ndarray, each shape (T_scene,) or (T_scene, C).
                     Consecutive entries MUST have identical channel count and
                     sample rate. Length of each scene is arbitrary.
        sample_rate: audio sample rate (Hz).
        crossfade_ms: total crossfade duration at each boundary.

    Returns:
        Concatenated audio, shape matches input channel layout. Total length
        equals sum(scene lengths) - (N-1) * crossfade_samples / 2  (i.e. the
        crossfade overlap eats into both sides equally).
    """
    if not scene_audio:
        return np.zeros(0, dtype=np.float32)
    if len(scene_audio) == 1:
        return np.asarray(scene_audio[0]).astype(np.float32, copy=True)

    cf_samples = max(1, int(round(crossfade_ms * sample_rate / 1000.0)))

    # Determine channel count
    first = np.asarray(scene_audio[0])
    is_multichannel = first.ndim == 2
    # Clone arrays as float32
    scenes = [np.asarray(s, dtype=np.float32) for s in scene_audio]

    # Total output length accounts for the overlap at each boundary.
    # Each boundary removes cf_samples of length (overlapping region).
    n_boundaries = len(scenes) - 1
    total_len = sum(s.shape[0] for s in scenes) - n_boundaries * cf_samples
    if total_len <= 0:
        # Degenerate — pure concatenation fallback
        return np.concatenate(scenes, axis=0)

    if is_multichannel:
        out = np.zeros((total_len, scenes[0].shape[1]), dtype=np.float32)
    else:
        out = np.zeros(total_len, dtype=np.float32)

    write_pos = 0
    for i, scene in enumerate(scenes):
        scene_len = scene.shape[0]
        if i == 0:
            # First scene: write in full except the trailing half-overlap
            tail_fade = min(cf_samples, scene_len)
            # Copy everything
            out[write_pos : write_pos + scene_len] = scene
            write_pos += scene_len - tail_fade
            # The tail_fade region will be overlapped by the next scene's head
        else:
            prev = scenes[i - 1]
            this_fade = min(cf_samples, scene_len, prev.shape[0])
            fade = raised_cosine_crossfade(this_fade)
            if is_multichannel:
                fade = fade[:, None]
            # write_pos currently points to start of blend region (prev's tail-start)
            # Blend: out[write_pos:write_pos+this_fade] = prev_tail*(1-w) + this_head*w
            # But out currently holds prev_tail from the previous iteration's raw copy.
            # So blend in place:
            prev_tail = out[write_pos : write_pos + this_fade]
            this_head = scene[:this_fade]
            blended = prev_tail * (1.0 - fade) + this_head * fade
            out[write_pos : write_pos + this_fade] = blended
            write_pos += this_fade
            # Copy rest of this scene
            rest_start = this_fade
            if i < len(scenes) - 1:
                # Save room for tail overlap with next scene
                next_scene = scenes[i + 1]
                next_fade = min(cf_samples, scene.shape[0] - this_fade,
                                next_scene.shape[0])
                copy_len = scene_len - this_fade - next_fade
                out[write_pos : write_pos + copy_len] = scene[rest_start : rest_start + copy_len]
                write_pos += copy_len
                # Now write the tail that will blend with next
                out[write_pos : write_pos + next_fade] = (
                    scene[rest_start + copy_len : rest_start + copy_len + next_fade]
                )
                # Do not advance write_pos; next iteration will blend from here.
            else:
                # Last scene: copy rest in full
                out[write_pos : write_pos + (scene_len - this_fade)] = scene[rest_start:]
                write_pos += scene_len - this_fade

    return out[:write_pos] if write_pos < out.shape[0] else out
