"""run_v3_5_7_pipeline.py — V3.5.7 aggressive 파라미터 변종 (Phase P1).

V3.5.6과 동일한 파이프라인 (apply_v3_5_6 함수 재사용)을 더 강한 spatial /
ducking 파라미터로 호출. 청각 체감 차이가 더 뚜렷한 대안 버전.

V3.5.6 vs V3.5.7 파라미터 비교:
                          V3.5.6        V3.5.7
  mid_gain_db             -1.0 dB       -2.0 dB        (mid 더 양보)
  side_gain_db            +1.5 dB       +2.5 dB        (공간 더 확장)
  ducking max_reduction   6.0 dB        9.0 dB         (ducking 더 깊음)
  ducking threshold       -30 dB        -36 dB         (더 자주 발동)
  EQ intensity            1.0           1.0            (동일)

설계 의도:
  · V3.5.6 = pro audio 철학 (subtle, transparent)
  · V3.5.7 = consumer appeal (aggressive, palpable)
  · MUSHRA 4조건으로 청취자 선호 분포 측정

주의: V3.5.6 의 파라미터는 변경하지 않음 (apply_v3_5_6 함수에 인자만 다르게 전달).
htdemucs 분리 결과(stems) 재사용 → 처리 시간 단축.

후처리:
  1) v3_5_7.wav 생성
  2) original_matched.wav LUFS 에 매칭 → v3_5_7_matched.wav
  3) libopus 320 kbps → v3_5_7_matched.opus
  4) 평가 페이지 디렉토리에 복사

사용:
  PYTHONIOENCODING=utf-8 PYTHONPATH=. D:/Homecinema/venv/Scripts/python.exe \\
      tools/run_v3_5_7_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from scipy.io import wavfile


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.run_v3_5_6_pipeline import apply_v3_5_6  # noqa: E402


JOB_PATHS = {
    "topgun":   REPO / "backend" / "data" / "jobs" / "fe2ecad8-dc25-4131-adfe-ffeea6d977a1",
    "lalaland": REPO / "backend" / "data" / "jobs" / "lalaland-demo",
}
FT_BASE = REPO / "tmp" / "full_trailer"
EVAL_BASE = REPO / "evaluation" / "webmushra" / "configs" / "resources" / "audio" / "full_trailer"


# V3.5.7 강화 파라미터
V3_5_7_PARAMS = dict(
    use_mid_side=True,
    mid_gain_db=-2.0,                       # V3.5.6: -1.0
    side_gain_db=+2.5,                      # V3.5.6: +1.5
    use_sidechain_ducking=True,
    ducking_threshold_db=-36.0,             # V3.5.6: -30
    ducking_ratio=4.0,
    ducking_max_reduction_db=9.0,           # V3.5.6: 6.0
    ducking_attack_ms=10.0,
    ducking_release_ms=150.0,
    ducking_band_low_hz=2000.0,
    ducking_band_high_hz=5000.0,
)

PEAK_CEILING_DB = -1.0


# ────────────────────────────────────────────────────────
# I/O & helpers
# ────────────────────────────────────────────────────────
def _load_float64(path: Path) -> tuple[int, np.ndarray]:
    sr, a = wavfile.read(str(path))
    if a.dtype == np.int16:
        a = a.astype(np.float64) / 32768.0
    elif a.dtype == np.int32:
        a = a.astype(np.float64) / 2147483648.0
    else:
        a = a.astype(np.float64)
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    return sr, a


def _save_int16(arr_smp_ch: np.ndarray, sr: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(arr_smp_ch, -1.0, 1.0)
    wavfile.write(str(path), sr, (a * 32767.0).astype(np.int16))


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-10 else -float("inf")


def loudness_match(input_wav: Path, target_wav: Path, output_wav: Path) -> dict:
    """input 을 target LUFS 에 매칭, peak ceiling -1 dBFS."""
    sr_t, t = _load_float64(target_wav)
    sr_i, i = _load_float64(input_wav)
    if sr_t != sr_i:
        raise ValueError(f"SR mismatch: target={sr_t} input={sr_i}")
    meter = pyln.Meter(sr_t)
    target_lufs = float(meter.integrated_loudness(t))
    orig_lufs = float(meter.integrated_loudness(i))
    gain_db = target_lufs - orig_lufs
    matched = i * (10.0 ** (gain_db / 20.0))
    peak_db = _db(float(np.abs(matched).max()))
    peak_atten_db = 0.0
    if peak_db > PEAK_CEILING_DB:
        peak_atten_db = PEAK_CEILING_DB - peak_db
        matched = matched * (10.0 ** (peak_atten_db / 20.0))
    final_lufs = float(meter.integrated_loudness(matched))
    final_peak = _db(float(np.abs(matched).max()))
    _save_int16(matched, sr_t, output_wav)
    return {
        "target_lufs": target_lufs, "input_lufs": orig_lufs,
        "raw_gain_db": gain_db, "peak_atten_db": peak_atten_db,
        "final_lufs": final_lufs, "final_peak_db": final_peak,
        "delta_lu": final_lufs - target_lufs,
    }


def encode_opus(input_wav: Path, output_opus: Path, bitrate_k: int = 320) -> dict:
    output_opus.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_wav),
        "-c:a", "libopus", "-b:a", f"{bitrate_k}k",
        "-vbr", "on", "-application", "audio",
        str(output_opus),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
    return {"output": str(output_opus), "size_kb": output_opus.stat().st_size / 1024.0}


# ────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="topgun", choices=list(JOB_PATHS.keys()))
    args = ap.parse_args()

    job_dir = JOB_PATHS[args.job]
    ft_dir = FT_BASE / args.job

    original_wav  = ft_dir / "original.wav"
    vocals_wav    = ft_dir / "stems" / "vocals.wav"
    no_vocals_wav = ft_dir / "stems" / "no_vocals.wav"
    timeline_path = job_dir / "timeline.json"
    target_wav    = ft_dir / "original_matched.wav"   # 매칭 기준
    v3_5_7_wav         = ft_dir / "v3_5_7.wav"
    v3_5_7_matched_wav = ft_dir / "v3_5_7_matched.wav"
    v3_5_7_matched_opus = ft_dir / "v3_5_7_matched.opus"

    for p in (original_wav, vocals_wav, no_vocals_wav, timeline_path, target_wav):
        if not p.exists():
            raise FileNotFoundError(f"필수 입력 누락: {p}")

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    clip_end = timeline["scenes"][-1]["end_sec"]

    print("=" * 64)
    print(f"V3.5.7 aggressive variant  ({args.job})")
    print("=" * 64)
    print(f"  파라미터: mid {V3_5_7_PARAMS['mid_gain_db']:+.1f} / "
          f"side {V3_5_7_PARAMS['side_gain_db']:+.1f} / "
          f"duck max {V3_5_7_PARAMS['ducking_max_reduction_db']:.1f} dB / "
          f"th {V3_5_7_PARAMS['ducking_threshold_db']:.0f} dB")

    # 1) V3.5.7 처리 (apply_v3_5_6 함수 + 강한 인자)
    print(f"\n[1/4] apply_v3_5_6 with aggressive params  → {v3_5_7_wav.name}")
    t0 = time.perf_counter()
    report = apply_v3_5_6(
        original_wav_path=original_wav,
        vocals_wav_path=vocals_wav,
        no_vocals_wav_path=no_vocals_wav,
        timeline_path=timeline_path,
        clip_start_sec=0.0,
        clip_end_sec=clip_end,
        output_wav_path=v3_5_7_wav,
        eq_intensity=1.0,
        dialogue_protection_no_vocals=True,
        use_crossfade=True,
        **V3_5_7_PARAMS,
    )
    elapsed = time.perf_counter() - t0
    print(f"      elapsed: {elapsed:.1f}s")
    print(f"      pre-comp: peak={report['levels']['pre_compressor']['peak_db']:+.2f}  "
          f"rms={report['levels']['pre_compressor']['rms_db']:+.2f}")
    print(f"      final:    peak={report['levels']['final_output']['peak_db']:+.2f}  "
          f"rms={report['levels']['final_output']['rms_db']:+.2f}")
    if report["clipping_warning"]:
        print(f"      ⚠ clipping normalized")

    # 2) Loudness matching
    print(f"\n[2/4] LUFS 매칭 (target = original_matched.wav)")
    lm = loudness_match(v3_5_7_wav, target_wav, v3_5_7_matched_wav)
    print(f"      target LUFS: {lm['target_lufs']:+.2f}")
    print(f"      v3_5_7 raw LUFS: {lm['input_lufs']:+.2f}  → gain {lm['raw_gain_db']:+.2f} dB"
          + (f"  + peak atten {lm['peak_atten_db']:+.2f} dB" if lm['peak_atten_db'] != 0 else "  (peak OK)"))
    print(f"      ✓ final LUFS: {lm['final_lufs']:+.2f}  peak: {lm['final_peak_db']:+.2f} dBFS"
          f"  Δ {lm['delta_lu']:+.2f} LU")

    # 3) opus 인코딩
    print(f"\n[3/4] opus 320 kbps 인코딩")
    enc = encode_opus(v3_5_7_matched_wav, v3_5_7_matched_opus, bitrate_k=320)
    print(f"      ✓ {Path(enc['output']).relative_to(REPO)}  ({enc['size_kb']:.1f} KB)")

    # 4) 평가 디렉토리에 복사 (wav + opus 둘 다)
    eval_dir = EVAL_BASE / args.job
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_wav  = eval_dir / "v3_5_7_matched.wav"
    eval_opus = eval_dir / "v3_5_7_matched.opus"
    shutil.copy2(v3_5_7_matched_wav,  eval_wav)
    shutil.copy2(v3_5_7_matched_opus, eval_opus)
    print(f"\n[4/4] 평가 디렉토리 복사")
    print(f"      ✓ {eval_wav.relative_to(REPO)}")
    print(f"      ✓ {eval_opus.relative_to(REPO)}")

    print("\n" + "=" * 64)
    print("완료. 다음 단계:")
    print("  · 객관 지표 재측정 (P2): tools/objective_metrics_v2.py")
    print("  · MUSHRA 4조건 확장 (P3): full_trailer_comparison.html")
    print("=" * 64)


if __name__ == "__main__":
    main()
