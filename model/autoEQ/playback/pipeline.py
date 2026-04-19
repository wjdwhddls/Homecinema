"""End-to-end playback pipeline orchestrator.

Input: original video + timeline.json (from infer_pseudo)
Output: EQ-applied video

Flow:
  1. extract_audio(video → scratch.wav)  # preserve sr + channels
  2. for each scene in timeline.scenes:
        slice audio[start:end]
        eq_chain(effective_bands) → processed slice
  3. apply_boundary_crossfades(scene_audio_list) → concatenated processed audio
  4. write processed wav
  5. remux_video_with_audio(video, processed_wav → output)
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .crossfade import DEFAULT_CROSSFADE_MS, raised_cosine_crossfade
from .eq_applier import apply_scene_eq_to_audio
from .remux import extract_audio, get_audio_info, remux_video_with_audio


def _slice_scene_audio(
    full_audio: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Slice [start, end] seconds from full audio, clamped to array bounds."""
    s = int(round(start_sec * sample_rate))
    e = int(round(end_sec * sample_rate))
    s = max(0, s)
    total = full_audio.shape[0]
    e = min(total, max(e, s))
    return full_audio[s:e]


def apply_timeline_to_audio(
    audio: np.ndarray,
    sample_rate: int,
    timeline: dict,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
) -> np.ndarray:
    """Apply timeline's scene EQ chains to audio, preserving total length.

    Strategy (A/V sync-safe):
      1. Allocate output = copy(audio)   # same length as input
      2. For each scene: compute that scene's EQ over its [start, end] region
         and write into output[start:end] (in place)
      3. For each boundary: recompute both prev EQ and next EQ over a
         `crossfade_ms`-wide window centered at the boundary (chains extended
         past/before the scene border to get proper tails). Raised-cosine blend
         those two extensions into output.

    Effect: total length = input length (video sync preserved); only the small
    boundary band (~300 ms) is a smooth blend; everywhere else is that scene's
    pure EQ output.
    """
    scenes = timeline.get("scenes", [])
    if not scenes:
        return np.asarray(audio, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)
    out = audio.copy()
    n_total = audio.shape[0]

    # Pass 1: in-place per-scene EQ
    scene_meta: list[tuple[int, int, list]] = []  # (start_sample, end_sample, bands)
    for sc in scenes:
        start = float(sc["start_sec"])
        end = float(sc["end_sec"])
        s_sample = max(0, int(round(start * sample_rate)))
        e_sample = min(n_total, int(round(end * sample_rate)))
        if e_sample <= s_sample:
            continue
        bands = sc["eq_preset"]["effective_bands"]
        slice_a = audio[s_sample:e_sample]
        processed = apply_scene_eq_to_audio(slice_a, sample_rate, bands)
        out[s_sample:e_sample] = processed
        scene_meta.append((s_sample, e_sample, bands))

    # Pass 2: boundary crossfades (raised-cosine blend of prev-chain vs next-chain
    # computed over a window straddling the boundary)
    cf_samples = max(1, int(round(crossfade_ms * sample_rate / 1000.0)))
    half_cf = cf_samples // 2
    for i in range(len(scene_meta) - 1):
        prev_s, prev_e, prev_bands = scene_meta[i]
        next_s, next_e, next_bands = scene_meta[i + 1]
        boundary = prev_e  # = next_s (or close to it)
        blend_start = max(prev_s, boundary - half_cf)
        blend_end = min(next_e, boundary + (cf_samples - half_cf))
        if blend_end <= blend_start:
            continue
        # Run both EQ chains over the SAME raw slice to get prev_tail and next_head
        raw_slice = audio[blend_start:blend_end]
        prev_ext = apply_scene_eq_to_audio(raw_slice, sample_rate, prev_bands)
        next_ext = apply_scene_eq_to_audio(raw_slice, sample_rate, next_bands)
        n = blend_end - blend_start
        w = raised_cosine_crossfade(n)
        if audio.ndim == 2:
            w = w[:, None]
        out[blend_start:blend_end] = prev_ext * (1.0 - w) + next_ext * w

    return out


def apply_eq_to_video(
    video_path: str | Path,
    timeline_json: str | Path,
    output_video: str | Path,
    *,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    audio_bitrate: str = "192k",
    work_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """End-to-end playback pipeline.

    Args:
        video_path: original video.
        timeline_json: analysis output from infer_pseudo pipeline.
        output_video: destination path (.mp4/.mkv).
        crossfade_ms: scene-boundary crossfade duration.
        audio_bitrate: AAC bitrate for remuxed audio.
        work_dir: persistent scratch directory (default: temp).

    Returns:
        Path to written output video.
    """
    t0 = time.time()
    video_path = Path(video_path)
    timeline_json = Path(timeline_json)
    output_video = Path(output_video)

    timeline = json.loads(timeline_json.read_text())

    scratch_ctx = None
    if work_dir is None:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="moodeq_playback_")
        work = Path(scratch_ctx.name)
    else:
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
    extracted_wav = work / "orig_audio.wav"
    processed_wav = work / "processed_audio.wav"

    try:
        if verbose:
            print(f"[info] extracting audio from {video_path.name}")
        info = get_audio_info(video_path)
        extract_audio(video_path, extracted_wav)  # preserve sr + channels
        audio, sr = sf.read(str(extracted_wav), dtype="float32", always_2d=False)
        if verbose:
            print(f"[info] audio: sr={sr}  channels="
                  f"{audio.shape[1] if audio.ndim == 2 else 1}  "
                  f"duration={audio.shape[0]/sr:.1f}s")

        if verbose:
            print(f"[info] applying {len(timeline['scenes'])} scene EQ chains")
        processed = apply_timeline_to_audio(audio, sr, timeline, crossfade_ms=crossfade_ms)
        if verbose:
            print(f"[info] processed audio: {processed.shape[0]/sr:.1f}s")

        sf.write(str(processed_wav), processed, sr)

        if verbose:
            print(f"[info] remuxing video + processed audio → {output_video}")
        remux_video_with_audio(
            video_path, processed_wav, output_video, audio_bitrate=audio_bitrate
        )
        if verbose:
            print(f"[done] total elapsed={time.time() - t0:.1f}s")
        return output_video
    finally:
        if scratch_ctx is not None:
            scratch_ctx.cleanup()
