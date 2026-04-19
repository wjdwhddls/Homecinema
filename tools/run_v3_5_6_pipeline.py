"""run_v3_5_6_pipeline.py — V3.5.6 파이프라인 (V3.5.5 + M/S + Sidechain Ducking).

트랙 5+ Phase X4-3. V3.5.5 구조 보존 + perceptual enhancement 2종 추가:
  - Mid/Side processing: no_vocals 스템의 mid 감쇠 / side 강화
  - Sidechain ducking: vocals envelope 기반 no_vocals 2-5 kHz 동적 감쇠

처리 순서:
  1) 스템 로드 (original / vocals / no_vocals)
  2) no_vocals 시변 EQ (V3.3 presets + dialogue protection + crossfade)
  3) no_vocals M/S (mid -1.0, side +1.5)
  4) no_vocals Sidechain ducking (vocals envelope 기반, 2-5 kHz 대역)
  5) vocals 고정 명료도 EQ (V3.5.5와 동일)
  6) 합산
  7) Compressor + makeup gain
  8) Peak clipping prevent

V3.5.5 apply_v3_5_5 함수 불변. apply_v3_5_6 신규 함수로 분리.

실행:
  PYTHONIOENCODING=utf-8 PYTHONPATH=. venv/Scripts/python.exe \\
      tools/run_v3_5_6_pipeline.py --job topgun
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from pedalboard import Compressor, Gain, Pedalboard
from scipy.io import wavfile


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.autoEQ.inference.eq_engine import EQ_PRESETS_V3_3  # noqa: E402
from model.autoEQ.inference.playback import (  # noqa: E402
    apply_timevarying_eq_array, build_eq_chain,
)
from model.autoEQ.inference.spatial_processing import (  # noqa: E402
    apply_mid_side_processing, apply_sidechain_ducking,
)
from tools.run_v3_5_5_pipeline import (  # noqa: E402
    VOCALS_CLARITY_EQ,
    COMPRESSOR_THRESHOLD_DB, COMPRESSOR_RATIO,
    COMPRESSOR_ATTACK_MS, COMPRESSOR_RELEASE_MS,
    MAKEUP_GAIN_DB,
    load_wav_float32, save_wav_int16, measure, count_scenes_in_clip,
)


# ────────────────────────────────────────────────────────
# V3.5.6 기본 파라미터 (튜닝 시 수정)
# ────────────────────────────────────────────────────────
DEFAULT_MID_GAIN_DB            = -1.0
DEFAULT_SIDE_GAIN_DB           = +1.5
DEFAULT_DUCKING_THRESHOLD_DB   = -30.0
DEFAULT_DUCKING_RATIO          = 4.0
DEFAULT_DUCKING_MAX_REDUCTION  = 6.0
DEFAULT_DUCKING_ATTACK_MS      = 10.0
DEFAULT_DUCKING_RELEASE_MS     = 150.0
DEFAULT_DUCKING_BAND_LOW_HZ    = 2000.0
DEFAULT_DUCKING_BAND_HIGH_HZ   = 5000.0


def apply_v3_5_6(
    original_wav_path: Path,
    vocals_wav_path: Path,
    no_vocals_wav_path: Path,
    timeline_path: Path,
    clip_start_sec: float,
    clip_end_sec: float,
    output_wav_path: Path,
    eq_intensity: float = 1.0,
    dialogue_protection_no_vocals: bool = True,
    use_crossfade: bool = True,
    # V3.5.6 신규
    use_mid_side: bool = True,
    mid_gain_db: float = DEFAULT_MID_GAIN_DB,
    side_gain_db: float = DEFAULT_SIDE_GAIN_DB,
    use_sidechain_ducking: bool = True,
    ducking_threshold_db: float = DEFAULT_DUCKING_THRESHOLD_DB,
    ducking_ratio: float = DEFAULT_DUCKING_RATIO,
    ducking_max_reduction_db: float = DEFAULT_DUCKING_MAX_REDUCTION,
    ducking_attack_ms: float = DEFAULT_DUCKING_ATTACK_MS,
    ducking_release_ms: float = DEFAULT_DUCKING_RELEASE_MS,
    ducking_band_low_hz: float = DEFAULT_DUCKING_BAND_LOW_HZ,
    ducking_band_high_hz: float = DEFAULT_DUCKING_BAND_HIGH_HZ,
) -> dict:
    """V3.5.6 전체 파이프라인. V3.5.5 처리 보존 + M/S + sidechain ducking."""
    # 1) timeline + 스템 로드
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    scenes = timeline["scenes"]
    scenes_in_clip = count_scenes_in_clip(scenes, clip_start_sec, clip_end_sec)

    sr_o, orig = load_wav_float32(original_wav_path)
    sr_v, voc  = load_wav_float32(vocals_wav_path)
    sr_n, nov  = load_wav_float32(no_vocals_wav_path)
    if sr_o != sr_v or sr_o != sr_n:
        raise ValueError(f"SR mismatch: orig={sr_o} voc={sr_v} nov={sr_n}")
    sr = sr_o

    L = min(orig.shape[1], voc.shape[1], nov.shape[1])
    drift = max(orig.shape[1], voc.shape[1], nov.shape[1]) - L
    orig = orig[:, :L]
    voc  = voc [:, :L]
    nov  = nov [:, :L]

    # 2) no_vocals 시변 EQ (V3.3과 동일 처리)
    nov_eq = apply_timevarying_eq_array(
        audio_array=nov,
        sample_rate=sr,
        scenes=scenes,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        alpha_d=0.5,
        intensity=eq_intensity,
        confidence_scaling=True,
        dialogue_protection=dialogue_protection_no_vocals,
        presets=EQ_PRESETS_V3_3,
        prevent_clipping=False,
        use_crossfade=use_crossfade,
    )

    # 3) no_vocals M/S processing
    if use_mid_side:
        nov_eq = apply_mid_side_processing(
            nov_eq, mid_gain_db=mid_gain_db, side_gain_db=side_gain_db,
        )

    # 4) no_vocals Sidechain ducking (vocals envelope 기반)
    if use_sidechain_ducking:
        nov_eq = apply_sidechain_ducking(
            no_vocals=nov_eq.astype(np.float64),
            vocals=voc.astype(np.float64),
            sample_rate=sr,
            threshold_db=ducking_threshold_db,
            ratio=ducking_ratio,
            max_reduction_db=ducking_max_reduction_db,
            attack_ms=ducking_attack_ms,
            release_ms=ducking_release_ms,
            band_low_hz=ducking_band_low_hz,
            band_high_hz=ducking_band_high_hz,
        ).astype(np.float32)

    # 5) vocals 고정 명료도 EQ (V3.5.5와 동일)
    vocals_gains = VOCALS_CLARITY_EQ * eq_intensity
    voc_chain = build_eq_chain(vocals_gains)
    voc_eq = voc_chain(voc.astype(np.float32, copy=False), sr)

    # 6) 합산
    summed = voc_eq + nov_eq
    pre_peak_db, pre_rms_db = measure(summed)

    # 7) Compressor + makeup
    board = Pedalboard([
        Compressor(
            threshold_db=COMPRESSOR_THRESHOLD_DB,
            ratio=COMPRESSOR_RATIO,
            attack_ms=COMPRESSOR_ATTACK_MS,
            release_ms=COMPRESSOR_RELEASE_MS,
        ),
        Gain(gain_db=MAKEUP_GAIN_DB),
    ])
    out = board(summed, sr)

    # 8) Peak clipping prevent
    final_peak_lin = float(np.abs(out).max())
    clipping_warning = False
    if final_peak_lin > 0.99:
        clipping_warning = True
        scale = 0.95 / final_peak_lin
        out = out * scale
        print(f"    클리핑 방지: peak {final_peak_lin:.3f} → 0.95 (scale {scale:.4f})")

    final_peak_db, final_rms_db = measure(out)

    save_wav_int16(out, sr, output_wav_path)

    return {
        "sr": sr,
        "samples": L,
        "duration_sec": L / sr,
        "drift_samples": drift,
        "scenes_in_clip": scenes_in_clip,
        "eq_intensity": eq_intensity,
        "use_mid_side": use_mid_side,
        "mid_gain_db": mid_gain_db,
        "side_gain_db": side_gain_db,
        "use_sidechain_ducking": use_sidechain_ducking,
        "ducking_params": {
            "threshold_db": ducking_threshold_db,
            "ratio": ducking_ratio,
            "max_reduction_db": ducking_max_reduction_db,
            "attack_ms": ducking_attack_ms,
            "release_ms": ducking_release_ms,
            "band_low_hz": ducking_band_low_hz,
            "band_high_hz": ducking_band_high_hz,
        },
        "levels": {
            "pre_compressor": {"peak_db": pre_peak_db, "rms_db": pre_rms_db},
            "final_output":   {"peak_db": final_peak_db, "rms_db": final_rms_db},
        },
        "clipping_warning": clipping_warning,
    }


# ────────────────────────────────────────────────────────
# CLI (풀 트레일러 처리)
# ────────────────────────────────────────────────────────
JOB_PATHS = {
    "topgun":   REPO / "backend" / "data" / "jobs" / "fe2ecad8-dc25-4131-adfe-ffeea6d977a1",
    "lalaland": REPO / "backend" / "data" / "jobs" / "lalaland-demo",
}
FT_BASE = REPO / "tmp" / "full_trailer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="topgun", choices=list(JOB_PATHS.keys()))
    ap.add_argument("--no-mid-side", action="store_true")
    ap.add_argument("--no-sidechain", action="store_true")
    args = ap.parse_args()

    ft_dir  = FT_BASE / args.job
    job_dir = JOB_PATHS[args.job]

    original_wav = ft_dir / "original.wav"
    vocals_wav   = ft_dir / "stems" / "vocals.wav"
    no_vocals_wav = ft_dir / "stems" / "no_vocals.wav"
    timeline_path = job_dir / "timeline.json"
    v3_5_6_wav   = ft_dir / "v3_5_6.wav"

    for p in (original_wav, vocals_wav, no_vocals_wav, timeline_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} — run_v3_5_5_full_trailer.py 먼저 실행")

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    clip_end = timeline["scenes"][-1]["end_sec"]

    print(f"=== V3.5.6 full-trailer {args.job} ===")
    print(f"  M/S: {not args.no_mid_side}  Sidechain: {not args.no_sidechain}")

    t0 = time.perf_counter()
    report = apply_v3_5_6(
        original_wav_path=original_wav,
        vocals_wav_path=vocals_wav,
        no_vocals_wav_path=no_vocals_wav,
        timeline_path=timeline_path,
        clip_start_sec=0.0,
        clip_end_sec=clip_end,
        output_wav_path=v3_5_6_wav,
        use_mid_side=not args.no_mid_side,
        use_sidechain_ducking=not args.no_sidechain,
    )
    elapsed = time.perf_counter() - t0

    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  scenes_in_clip: {report['scenes_in_clip']}  drift: {report['drift_samples']} smp")
    print(f"  pre-compressor: peak={report['levels']['pre_compressor']['peak_db']:+.2f}  "
          f"rms={report['levels']['pre_compressor']['rms_db']:+.2f}")
    print(f"  final v3_5_6:   peak={report['levels']['final_output']['peak_db']:+.2f}  "
          f"rms={report['levels']['final_output']['rms_db']:+.2f}")
    if report["clipping_warning"]:
        print(f"  ⚠ clipping normalized")
    print(f"  → {v3_5_6_wav.relative_to(REPO)}")


if __name__ == "__main__":
    main()
