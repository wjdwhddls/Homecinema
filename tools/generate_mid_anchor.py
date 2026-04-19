"""generate_mid_anchor.py — MUSHRA mid anchor 생성 (7 kHz LPF) + LUFS 매칭 + opus 변환.

Phase W4: 풀 트레일러 비교용 anchor를 Low Anchor (3.5 kHz LPF) → Mid Anchor (7 kHz LPF)
로 교체. ITU-R BS.1534-3 표준은 Low/Mid 둘 다 정의하므로 표준 준수는 유지.

배경:
  Low Anchor (3.5 kHz LPF) 는 "라디오 수준" 으로 너무 명백하게 구별돼 평가가
  너무 쉬워짐 (참가자 1차 청취 피드백). Mid Anchor (7 kHz LPF) 는 "약간 어두운
  저음질 MP3" 정도라 V3.5.6과의 비교가 더 의미 있어짐. Post-screening 임계값도
  anchor ≤ 30 → ≤ 40 으로 완화.

워크플로 (--all 또는 단계별):
  1) 7kHz Butterworth 8차 LPF로 anchor wav 생성
  2) original_matched.wav 의 LUFS로 loudness 매칭 (peak ceiling -1.0 dBFS)
  3) FFT 검증: 7 kHz 이상 대역 에너지 ≤ -40 dBFS, passband 보존
  4) ffmpeg libopus 320 kbps stereo 변환

기존 Low Anchor 파일은 호출자가 archive/ 로 이동 (이 스크립트는 덮어씀).

사용:
  D:/Homecinema/venv/Scripts/python.exe tools/generate_mid_anchor.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


CUTOFF_HZ = 7000.0   # W4: Mid anchor (Low anchor 는 3500 Hz)
ORDER = 8
PEAK_CEILING_DB = -1.0
SPECTRUM_THRESHOLD_DB = -40.0


REPO = Path(__file__).resolve().parent.parent
TRAILER_DIR = REPO / "evaluation" / "webmushra" / "configs" / "resources" / "audio" / "full_trailer" / "topgun"


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


def generate_mid_anchor(original_wav: Path, output_wav: Path) -> dict:
    """원본을 7 kHz Butterworth 8차 LPF로 통과시켜 mid anchor 저장 (loudness 매칭 전)."""
    sr, audio = _load_float64(original_wav)
    sos = butter(ORDER, CUTOFF_HZ, btype="low", fs=sr, output="sos")
    filtered = sosfilt(sos, audio, axis=0)
    _save_int16(filtered, sr, output_wav)
    return {
        "sr": sr, "samples": audio.shape[0],
        "peak_db": _db(float(np.abs(filtered).max())),
        "rms_db": _db(float(np.sqrt(np.mean(filtered ** 2)))),
        "cutoff_hz": CUTOFF_HZ, "order": ORDER,
    }


def loudness_match(input_wav: Path, target_wav: Path, output_wav: Path) -> dict:
    """input_wav 를 target_wav 의 integrated LUFS 로 매칭, output_wav 에 덮어씀."""
    sr_t, t = _load_float64(target_wav)
    sr_i, i = _load_float64(input_wav)
    if sr_t != sr_i:
        raise ValueError(f"sample rate mismatch: target={sr_t} input={sr_i}")

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
        "target_lufs": target_lufs,
        "input_lufs": orig_lufs,
        "raw_gain_db": gain_db,
        "peak_atten_db": peak_atten_db,
        "final_lufs": final_lufs,
        "final_peak_db": final_peak,
        "delta_lu": final_lufs - target_lufs,
    }


def verify_spectrum(anchor_wav: Path) -> dict:
    """FFT로 7 kHz 이상 대역 최대 에너지가 -40 dBFS 이하인지 검증."""
    sr, audio = _load_float64(anchor_wav)
    mono = audio.mean(axis=1)
    n = len(mono)
    spec = np.fft.rfft(mono * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(spec) / (n / 2)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-10))

    above = freqs >= CUTOFF_HZ
    above_max_db = float(mag_db[above].max())
    above_freq = float(freqs[above][mag_db[above].argmax()])
    pass_mask = (freqs >= 1000) & (freqs < CUTOFF_HZ - 500)
    pass_max_db = float(mag_db[pass_mask].max()) if pass_mask.any() else -999.0

    # 추가: cutoff 직전 (3-6 kHz) 평균 vs 직후 (8-10 kHz) 평균 비율
    band_below = (freqs >= 3000) & (freqs < 6000)
    band_above = (freqs >= 8000) & (freqs < 10000)
    energy_below = float(np.mean(mag[band_below] ** 2))
    energy_above = float(np.mean(mag[band_above] ** 2))
    ratio = energy_above / max(energy_below, 1e-20)

    return {
        "sr": sr, "samples": n,
        "cutoff_hz": CUTOFF_HZ,
        "above_cutoff_max_db": above_max_db,
        "above_cutoff_max_freq_hz": above_freq,
        "passband_max_db": pass_max_db,
        "stopband_attenuation_db": pass_max_db - above_max_db,
        "above_below_energy_ratio_pct": ratio * 100.0,
        "passes": above_max_db <= SPECTRUM_THRESHOLD_DB,
    }


def encode_opus(input_wav: Path, output_opus: Path, bitrate_k: int = 320) -> dict:
    """ffmpeg libopus 320 kbps stereo 변환."""
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
    return {
        "input": str(input_wav),
        "output": str(output_opus),
        "size_kb": output_opus.stat().st_size / 1024.0,
    }


def main() -> None:
    print("=" * 64)
    print("Phase W4 — Mid Anchor (7 kHz LPF) 생성 + LUFS 매칭 + opus 변환")
    print("=" * 64)

    original = TRAILER_DIR / "original_matched.wav"
    anchor_wav = TRAILER_DIR / "anchor_matched.wav"
    anchor_opus = TRAILER_DIR / "anchor_matched.opus"

    if not original.exists():
        raise FileNotFoundError(f"원본 누락: {original}")

    # Step 1: LPF 적용 (임시 파일, 매칭 후 덮어씀)
    print(f"\n[1/4] {CUTOFF_HZ:.0f} Hz LPF 적용")
    tmp_filtered = TRAILER_DIR / "_tmp_anchor_filtered.wav"
    gen = generate_mid_anchor(original, tmp_filtered)
    print(f"      sr={gen['sr']}  samples={gen['samples']}  "
          f"peak={gen['peak_db']:+.2f} dBFS  rms={gen['rms_db']:+.2f}")

    # Step 2: LUFS 매칭 (original_matched 기준)
    print(f"\n[2/4] LUFS 매칭 (target = original_matched.wav)")
    lm = loudness_match(tmp_filtered, original, anchor_wav)
    print(f"      target_LUFS={lm['target_lufs']:+.2f}  filtered_LUFS={lm['input_lufs']:+.2f}")
    print(f"      gain {lm['raw_gain_db']:+.2f} dB"
          + (f"  + peak atten {lm['peak_atten_db']:+.2f} dB" if lm['peak_atten_db'] != 0 else "  (peak OK)"))
    print(f"      ✓ final_LUFS={lm['final_lufs']:+.2f}  final_peak={lm['final_peak_db']:+.2f} dBFS"
          f"  Δ={lm['delta_lu']:+.2f} LU")

    # Step 3: Spectrum 검증
    print(f"\n[3/4] FFT spectrum 검증")
    ver = verify_spectrum(anchor_wav)
    status = "✓ PASS" if ver["passes"] else "✗ FAIL"
    print(f"      [{status}] cutoff {ver['cutoff_hz']:.0f} Hz")
    print(f"      passband max  (1k–6.5k):  {ver['passband_max_db']:+.2f} dBFS")
    print(f"      stopband max  (≥7k):      {ver['above_cutoff_max_db']:+.2f} dBFS"
          f" (@{ver['above_cutoff_max_freq_hz']:.0f} Hz)")
    print(f"      stopband attenuation:     {ver['stopband_attenuation_db']:.1f} dB"
          f" (threshold ≤ {SPECTRUM_THRESHOLD_DB:.0f} dB)")
    print(f"      energy ratio (8–10k / 3–6k): {ver['above_below_energy_ratio_pct']:.4f} %"
          f"  (LPF 효과 — 0 에 가까워야 함)")

    # Step 4: opus 인코딩
    print(f"\n[4/4] opus 320 kbps 인코딩")
    enc = encode_opus(anchor_wav, anchor_opus, bitrate_k=320)
    print(f"      ✓ {Path(enc['output']).relative_to(REPO)}  ({enc['size_kb']:.1f} KB)")

    # cleanup
    try:
        tmp_filtered.unlink()
    except OSError:
        pass

    print("\n" + "=" * 64)
    print("완료. 다음 단계:")
    print("  · 브라우저 강력 새로고침 → 새 anchor 청취")
    print("  · post-screening 임계: anchor ≤ 40 (Mid 대응)")
    print("=" * 64)


if __name__ == "__main__":
    main()
