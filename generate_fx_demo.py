"""Mood-driven FX chain → kakao_eq_fx.mp4 생성.

입력:  runs/demo_kakao/kakao_eq_applied.mp4 (이미 1× EQ 적용됨)
       runs/demo_kakao/timeline.json (scene mood)
출력:  runs/demo_kakao/kakao_eq_fx.mp4   (1× EQ + mood FX)

MOOD_FX 레시피는 peer-reviewed 문헌을 근거로 한다.

레시피 근거 (임의값 금지, 방향성 인용 가능):
    - Sub-bass / low-shelf boost: Juslin & Västfjäll 2008 (BRECVEM brain stem reflex),
      Zentner et al. 2008 (GEMS Power/Tension cluster)
    - Binary reverb mode (dry/small/large): Rumsey 2002 spatial quality,
      Sato et al. 2007 spaciousness, Berg & Rumsey 2001 attributes
    - High-shelf brightness boost: McAdams 1995 timbre, Eerola 2011 brightness→arousal

문헌 근거 없는 craft 매개변수 (compression ratio, stereo width)는 제외.
Reverb 는 pedalboard 기본 preset 3택 (room_size + wet_level 짝) 사용 —
   구체적 decay 수치를 직접 정하지 않음.

Usage:
    venv/bin/python generate_fx_demo.py                # 기본: kakao 데모
    venv/bin/python generate_fx_demo.py --force        # 이미 있어도 재생성
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from pedalboard import Pedalboard, LowShelfFilter, HighShelfFilter, Reverb, Limiter

# === Mood → FX 레시피 (peer-reviewed 방향성만) ===

# 문헌 뒷받침되는 3-모드 reverb preset.
# 구체 decay 는 pedalboard 기본값, room_size/wet_level 만 조절.
REVERB_PRESETS: dict[str, dict[str, float]] = {
    "dry":        {"room_size": 0.05, "wet_level": 0.00, "dry_level": 1.00},
    "small_room": {"room_size": 0.25, "wet_level": 0.10, "dry_level": 0.90},
    "large_hall": {"room_size": 0.85, "wet_level": 0.20, "dry_level": 0.85},
}

# 각 mood 별 레시피. 각 필드 근거:
#   sub_bass_shelf_db: Juslin/Zentner — Tension/Power 저역 visceral
#   low_shelf_db:      Eerola 2011 — 저역 weight → 슬픔
#   high_shelf_db:     McAdams/Eerola — 고역 brightness → wonder/arousal
#   reverb_mode:       Rumsey/Sato — 공간감 ↔ mood 방향
MOOD_FX_RECIPE: dict[str, dict] = {
    "Tension": {
        "sub_bass_shelf_db": +2.0,
        "reverb_mode": "dry",
    },
    "Wonder": {
        "high_shelf_db": +1.0,
        "reverb_mode": "large_hall",
    },
    "Tenderness": {
        "reverb_mode": "small_room",
    },
    "Sadness": {
        "low_shelf_db": +1.0,
        "reverb_mode": "dry",
    },
    "Power": {
        "sub_bass_shelf_db": +3.0,
    },
    "Peacefulness": {
        "reverb_mode": "small_room",
    },
    # JoyfulActivation 은 우리 K=7 Mood 에서 거의 0% (LIRIS training 분포상) — default no-op
}

# Shelf cutoff 는 표준 마스터링 관행 (not 임의)
SUB_BASS_CUTOFF_HZ = 60.0
LOW_SHELF_CUTOFF_HZ = 200.0
HIGH_SHELF_CUTOFF_HZ = 8000.0


def build_mood_fx_board(mood: str) -> Pedalboard:
    """mood → pedalboard chain. 빈 체인이면 no-op."""
    recipe = MOOD_FX_RECIPE.get(mood, {})
    board = Pedalboard([])

    if "sub_bass_shelf_db" in recipe:
        board.append(LowShelfFilter(
            cutoff_frequency_hz=SUB_BASS_CUTOFF_HZ,
            gain_db=recipe["sub_bass_shelf_db"],
        ))
    if "low_shelf_db" in recipe:
        board.append(LowShelfFilter(
            cutoff_frequency_hz=LOW_SHELF_CUTOFF_HZ,
            gain_db=recipe["low_shelf_db"],
        ))
    if "high_shelf_db" in recipe:
        board.append(HighShelfFilter(
            cutoff_frequency_hz=HIGH_SHELF_CUTOFF_HZ,
            gain_db=recipe["high_shelf_db"],
        ))

    rev_mode = recipe.get("reverb_mode", "dry")
    cfg = REVERB_PRESETS[rev_mode]
    if cfg["wet_level"] > 0:
        board.append(Reverb(
            room_size=cfg["room_size"],
            wet_level=cfg["wet_level"],
            dry_level=cfg["dry_level"],
            damping=0.5,
            width=1.0,
            freeze_mode=0.0,
        ))
    return board


def _strip_reverb(board: Pedalboard) -> Pedalboard:
    """dialogue-safe board: Reverb stage 만 제거, shelf 는 유지."""
    return Pedalboard([e for e in board if not isinstance(e, Reverb)])


def _process_segment(
    seg: np.ndarray, board: Pedalboard, sr: int, is_stereo: bool,
) -> np.ndarray:
    """세그먼트에 board 적용 후 길이를 seg 와 맞춘다 (pad or truncate)."""
    if len(board) == 0:
        return seg.astype(np.float32, copy=True)
    if is_stereo:
        proc = board(seg, sample_rate=float(sr))
    else:
        proc = board(seg.reshape(-1, 1), sample_rate=float(sr)).reshape(-1)
    if proc.shape[0] > seg.shape[0]:
        proc = proc[: seg.shape[0]]
    elif proc.shape[0] < seg.shape[0]:
        pad = seg.shape[0] - proc.shape[0]
        if is_stereo:
            proc = np.concatenate(
                [proc, np.zeros((pad, proc.shape[1]), dtype=np.float32)], axis=0
            )
        else:
            proc = np.concatenate([proc, np.zeros(pad, dtype=np.float32)], axis=0)
    return proc


def extract_audio(video: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vn", "-acodec", "pcm_s16le", str(dst)],
        check=True,
    )


def remux(video: Path, audio_wav: Path, out: Path, bitrate: str = "192k") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video), "-i", str(audio_wav),
         "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
         "-map", "0:v:0", "-map", "1:a:0",
         str(out)],
        check=True,
    )


def apply_mood_fx_per_scene(
    audio: np.ndarray,
    sr: int,
    timeline: dict,
    edge_smooth_ms: int = 50,
    dialogue_xfade_ms: int = 30,
) -> np.ndarray:
    """Scene 구간마다 FX chain 적용 (**in-place length-preserving**).

    audio shape: (N,) mono or (N, 2) stereo.
    원본과 동일 길이 보장 — scene timeline 과 시간 정렬 유지.
    경계 불연속은 짧은 (~50 ms) raised-cosine crossfade 로 부드럽게.

    Dialogue-aware reverb bypass (Phase 5-A rev.):
        scene["dialogue"]["segments_rel"] 구간에서만 Reverb stage 제거,
        shelf (sub-bass/low/high) 는 유지 → 대사 명료도 확보 + 배경 mood 보존.
    """
    is_stereo = audio.ndim == 2
    total = audio.shape[0]
    result = audio.astype(np.float32).copy()
    smooth_n = int(sr * edge_smooth_ms / 1000)
    dlg_xfade_n = int(sr * dialogue_xfade_ms / 1000)

    mood_counts: dict[str, int] = {}
    dialogue_bypass_segments = 0

    for sc in timeline["scenes"]:
        s = max(0, int(round(sc["start_sec"] * sr)))
        e = min(total, int(round(sc["end_sec"] * sr)))
        if e <= s:
            continue
        mood = sc["mood"]["name"]
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
        board_full = build_mood_fx_board(mood)
        if len(board_full) == 0:
            continue  # FX 없는 mood → 원본 유지

        seg = audio[s:e].astype(np.float32)
        proc = _process_segment(seg, board_full, sr, is_stereo)

        # dialogue-aware reverb bypass — reverb 포함 mood 에서만 유의미
        has_reverb = any(isinstance(x, Reverb) for x in board_full)
        dlg = sc.get("dialogue") or {}
        segments_rel = dlg.get("segments_rel") or [] if has_reverb else []

        if segments_rel:
            board_safe = _strip_reverb(board_full)
            proc_safe = _process_segment(seg, board_safe, sr, is_stereo)
            seg_len = e - s
            for rs, re in segments_rel:
                d_start = max(0, int(round(float(rs) * sr)))
                d_end = min(seg_len, int(round(float(re) * sr)))
                if d_end <= d_start:
                    continue
                xn = min(dlg_xfade_n, (d_end - d_start) // 2)
                if xn > 0:
                    t = np.arange(xn) / xn
                    fi = 0.5 * (1 - np.cos(np.pi * t))  # 0→1
                    fo = fi[::-1]
                    if is_stereo:
                        fi = fi[:, None]; fo = fo[:, None]
                    # 대사 시작: full → safe 로 전환
                    proc[d_start:d_start + xn] = (
                        proc[d_start:d_start + xn] * (1 - fi)
                        + proc_safe[d_start:d_start + xn] * fi
                    )
                    # 대사 중간: safe
                    if d_end - xn > d_start + xn:
                        proc[d_start + xn:d_end - xn] = proc_safe[d_start + xn:d_end - xn]
                    # 대사 끝: safe → full 로 전환
                    proc[d_end - xn:d_end] = (
                        proc_safe[d_end - xn:d_end] * fo
                        + proc[d_end - xn:d_end] * (1 - fo)
                    )
                else:
                    proc[d_start:d_end] = proc_safe[d_start:d_end]
                dialogue_bypass_segments += 1

        # 경계 crossfade: 시작부와 끝부 짧게 블렌드 (scene 경계)
        out_seg = proc.copy()
        n = min(smooth_n, (e - s) // 2)
        if n > 0:
            t = np.arange(n) / n
            fi = 0.5 * (1 - np.cos(np.pi * t))  # fade-in 0→1
            fo = fi[::-1]                        # fade-out 1→0
            if is_stereo:
                fi = fi[:, None]; fo = fo[:, None]
            # 시작 경계: 원본 seg 에서 FX proc 으로 블렌드
            out_seg[:n] = seg[:n] * (1 - fi) + proc[:n] * fi
            # 끝 경계: FX proc 에서 원본 seg 로 블렌드
            out_seg[-n:] = proc[-n:] * fo + seg[-n:] * (1 - fo)

        result[s:e] = out_seg

    print(f"[fx] mood 분포: {mood_counts}")
    if dialogue_bypass_segments > 0:
        print(f"[fx] dialogue reverb-bypass 적용: {dialogue_bypass_segments} segments "
              f"(shelf 유지, reverb 만 dry)")
    return result


def generate_fx_video(
    eq_video: Path,
    timeline_json: Path,
    output_video: Path,
    crossfade_ms: int = 300,
) -> None:
    """1× EQ 영상 + mood FX → 1× EQ+FX 영상."""
    if not eq_video.exists():
        raise FileNotFoundError(f"EQ video 없음: {eq_video}")
    if not timeline_json.exists():
        raise FileNotFoundError(f"timeline.json 없음: {timeline_json}")

    timeline = json.loads(timeline_json.read_text())
    print(f"[fx] input:    {eq_video.name}")
    print(f"[fx] timeline: {timeline_json.name} ({len(timeline['scenes'])} scenes)")

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        eq_wav = td_p / "eq.wav"
        fx_wav = td_p / "fx.wav"

        print("[fx] extracting EQ audio...")
        extract_audio(eq_video, eq_wav)
        audio, sr = sf.read(str(eq_wav), dtype="float32", always_2d=False)
        print(f"[fx] audio shape={audio.shape}, sr={sr}")

        print("[fx] applying mood FX chain per scene...")
        processed = apply_mood_fx_per_scene(audio, sr, timeline, crossfade_ms)
        # 길이 맞추기 (크로스페이드로 약간 짧아질 수 있음 → pad)
        if processed.shape[0] < audio.shape[0]:
            pad_len = audio.shape[0] - processed.shape[0]
            if processed.ndim == 2:
                processed = np.concatenate(
                    [processed, np.zeros((pad_len, processed.shape[1]), dtype=np.float32)],
                    axis=0,
                )
            else:
                processed = np.concatenate(
                    [processed, np.zeros(pad_len, dtype=np.float32)],
                    axis=0,
                )

        # 피크 안전 — 전체 감쇠 대신 Limiter (peak 만 compress, 평균 level 유지)
        peak_before = float(np.max(np.abs(processed)))
        if peak_before > 0.95:
            limiter = Pedalboard([Limiter(threshold_db=-0.5, release_ms=100.0)])
            if processed.ndim == 2:
                processed = limiter(processed, sample_rate=float(sr))
            else:
                processed = limiter(
                    processed.reshape(-1, 1), sample_rate=float(sr)
                ).reshape(-1)
            peak_after = float(np.max(np.abs(processed)))
            print(f"[fx] peak {peak_before:.3f} → limiter → {peak_after:.3f} "
                  f"(평균 level 유지)")

        sf.write(str(fx_wav), processed, sr, subtype="PCM_16")
        print(f"[fx] wrote processed WAV ({processed.shape[0] / sr:.1f}s)")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fx] remuxing video + fx audio → {output_video.name}")
        remux(eq_video, fx_wav, output_video)

    print(f"[done] wrote {output_video}")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate 1× EQ + mood FX video")
    p.add_argument(
        "--eq-video",
        type=Path,
        default=Path("runs/demo_kakao/kakao_eq_applied.mp4"),
        help="입력: 1× EQ 적용된 영상",
    )
    p.add_argument(
        "--timeline",
        type=Path,
        default=Path("runs/demo_kakao/timeline.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("runs/demo_kakao/kakao_eq_fx.mp4"),
    )
    p.add_argument("--crossfade-ms", type=int, default=300)
    p.add_argument("--force", action="store_true",
                   help="이미 output 이 있어도 재생성")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        print(f"[skip] 이미 존재: {args.output} (재생성은 --force)")
        return 0

    generate_fx_video(
        eq_video=args.eq_video,
        timeline_json=args.timeline,
        output_video=args.output,
        crossfade_ms=args.crossfade_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
