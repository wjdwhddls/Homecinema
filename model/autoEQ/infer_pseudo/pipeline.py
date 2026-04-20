"""End-to-end inference pipeline orchestrator.

Parallel structure per spec V3.3 §5-0:

    Thread A (main): scene_detect → window_slider → model_inference → ema
    Thread B (side): vad (independent, needs audio only)
                                  │
                                  ▼
                         dialogue_density join
                                  │
                                  ▼
                     mood_mapper + eq_preset
                                  │
                                  ▼
                         timeline_writer → JSON

Join point: once both threads finish, compute per-scene dialogue density and
produce final EQ decisions.

Usage:
    from model.autoEQ.infer_pseudo.pipeline import analyze_video
    timeline = analyze_video(
        video_path="movie.mp4",
        ckpt_path="runs/phase3_v2_gemini_target/best_model.pt",
        output_json="movie.timeline.json",
    )
"""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dialogue_density import compute_all_scene_densities
from .ema_smoother import aggregate_by_scene, apply_ema_within_scenes
from .eq_preset import apply_dialogue_protection, get_original_bands
from .mood_mapper import va_to_mood_name
from .model_inference import predict_windows
from .scene_detect import detect_and_merge
from .timeline_writer import build_timeline_dict, write_timeline
from .types import Scene, SceneEQ, SceneVA, SpeechSegment
from .vad import extract_audio_16k_mono, run_vad
from .window_slider import slide_windows_all_scenes


def _video_duration(video_path: Path) -> float:
    """Cheap duration lookup via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps if fps > 0 else 0.0


def _thread_a_visual_branch(video_path: Path) -> list[Scene]:
    """Scene detection only (video-dependent)."""
    return detect_and_merge(str(video_path))


def _thread_b_vad_branch(
    video_path: Path, work_audio_wav: Path
) -> list[SpeechSegment]:
    """Extract 16k mono + Silero VAD. Saves WAV to shared location for model_inference."""
    extract_audio_16k_mono(video_path, work_audio_wav)
    return run_vad(work_audio_wav)


def build_scene_eq_list(
    scene_va_list: list[SceneVA],
    densities: dict[int, float],
    alpha_d: float,
) -> list[SceneEQ]:
    """For each SceneVA, map to mood, build preset, apply dialogue protection."""
    out: list[SceneEQ] = []
    for va in scene_va_list:
        mood_idx, mood_name = va_to_mood_name(va.valence, va.arousal)
        density = float(densities.get(va.scene_idx, 0.0))
        original = get_original_bands(mood_name)
        effective = apply_dialogue_protection(original, density, alpha_d=alpha_d)
        out.append(SceneEQ(
            scene_idx=va.scene_idx,
            start_sec=va.start_sec,
            end_sec=va.end_sec,
            valence=va.valence,
            arousal=va.arousal,
            mood=mood_name,
            mood_idx=mood_idx,
            dialogue_density=density,
            original_bands=original,
            effective_bands=effective,
        ))
    return out


def analyze_video(
    video_path: str | Path,
    ckpt_path: str | Path,
    output_json: str | Path,
    *,
    num_mood_classes: int = 4,
    alpha_d: float = 0.5,
    ema_alpha: float = 0.3,
    batch_size: int = 16,
    variant: str = "base",
    include_windows: bool = False,
    work_dir: str | Path | None = None,
    model_version: str = "train_pseudo_v3.3",
    verbose: bool = True,
) -> dict:
    """Run the full analysis pipeline → write timeline.json, return dict.

    Args:
        video_path:    Input video (mp4/mkv/etc., any format ffmpeg reads).
        ckpt_path:     Trained model .pt from run_train.py.
        output_json:   Destination for timeline.json.
        num_mood_classes: Must match training (K=4 default).
        alpha_d:       Dialogue protection strength (§5-7). 0.5 default.
        ema_alpha:     EMA smoothing α (§5-4). 0.3 default.
        batch_size:    Windows per model forward pass.
        variant:       Training variant key — ``base`` / ``gmu`` / ``ast_gmu``.
                       Must match the ckpt's training config. V3.3 official
                       final is ``ast_gmu``.
        include_windows: If True, JSON includes per-window details (larger file).
        work_dir:      Scratch directory for extracted 16 kHz WAV (defaults to
                       a temp dir cleaned up on exit).
    """
    t0 = time.time()
    video_path = Path(video_path)
    ckpt_path = Path(ckpt_path)
    output_json = Path(output_json)

    scratch_ctx = None
    if work_dir is None:
        scratch_ctx = tempfile.TemporaryDirectory(prefix="moodeq_")
        work_dir_path = Path(scratch_ctx.name)
    else:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)
    audio_16k_wav = work_dir_path / "audio_16k_mono.wav"

    try:
        duration_sec = _video_duration(video_path)
        if verbose:
            print(f"[info] video: {video_path.name}  duration={duration_sec:.1f}s")

        # === Parallel stage: scene detection (A) + audio extract+VAD (B) ===
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_scenes = pool.submit(_thread_a_visual_branch, video_path)
            fut_vad = pool.submit(_thread_b_vad_branch, video_path, audio_16k_wav)

            scenes: list[Scene] = fut_scenes.result()
            if verbose:
                print(f"[info] scenes: {len(scenes)} (after merge)")

            # Model inference uses scenes → can't proceed until thread A done.
            # VAD (thread B) continues in parallel here.
            windows = slide_windows_all_scenes(scenes)
            if verbose:
                print(f"[info] windows: {len(windows)}")

            if verbose:
                audio_enc = "AST" if variant == "ast_gmu" else "PANNs"
                print(f"[info] running model inference (X-CLIP + {audio_enc} + {variant} head)...")
            t_model0 = time.time()
            window_va = predict_windows(
                windows=windows,
                video_path=video_path,
                audio_16k_path=audio_16k_wav,
                ckpt_path=ckpt_path,
                batch_size=batch_size,
                num_mood_classes=num_mood_classes,
                variant=variant,
            )
            if verbose:
                print(f"[info] model done ({time.time() - t_model0:.1f}s)")

            # Join thread B
            speech_segments = fut_vad.result()
            if verbose:
                print(f"[info] VAD: {len(speech_segments)} speech segments")

        # === Post-join: EMA + aggregation + mood + EQ + dialogue protection ===
        smoothed = apply_ema_within_scenes(window_va, alpha=ema_alpha)
        scene_bounds = {s.idx: (s.start_sec, s.end_sec) for s in scenes}
        scene_va = aggregate_by_scene(smoothed, scene_bounds=scene_bounds)

        densities = compute_all_scene_densities(scenes, speech_segments)
        scene_eq = build_scene_eq_list(scene_va, densities, alpha_d=alpha_d)

        # Group windows by scene for optional JSON inclusion
        scene_windows: dict[int, list] = {}
        for w in smoothed:
            scene_windows.setdefault(w.scene_idx, []).append(w)

        config_used = {
            "window_sec": 4, "stride_sec": 1,
            "ema_alpha": ema_alpha, "alpha_d": alpha_d,
            "num_mood_classes": num_mood_classes,
            "batch_size": batch_size,
        }
        timeline = build_timeline_dict(
            video_path=str(video_path),
            duration_sec=duration_sec,
            scene_va=scene_va,
            scene_eq=scene_eq,
            speech_segments=speech_segments,
            scene_windows=scene_windows,
            model_version=model_version,
            config=config_used,
            include_windows=include_windows,
        )
        write_timeline(output_json, timeline)

        elapsed = time.time() - t0
        if verbose:
            print(f"[done] wrote {output_json}  total elapsed={elapsed:.1f}s")
        return timeline
    finally:
        if scratch_ctx is not None:
            scratch_ctx.cleanup()
