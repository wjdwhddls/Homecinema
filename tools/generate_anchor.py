"""generate_anchor.py — MUSHRA 표준 low anchor 생성 (3.5kHz LPF).

ITU-R BS.1534-3 표준 low anchor: 원본을 3.5 kHz low-pass filter로 통과.
명백한 고역 손실로 "낮은 품질"임이 청취자에게 즉시 인지되어 MUSHRA 척도의
하한 기준점 역할.

설계:
  - Filter: Butterworth 8th order, low-pass at 3500 Hz
  - sosfilt + sosfiltfilt(zero-phase) 옵션 — 본 구현은 sosfilt(forward만, group delay
    있음 대신 phase 왜곡 없음). MUSHRA anchor는 청취 품질 차이가 핵심이므로
    timing 정밀도보다 spectrum shape이 중요.
  - 입출력 동일 샘플 수 보장 (filter delay 보정 없이 그대로 — anchor는 reference
    대비 청감 차이가 목적이라 sample-level alignment 불필요)

검증:
  - FFT로 3.5 kHz 이상 대역 에너지가 ≤ -40 dBFS 인지 확인
  - peak / rms 측정

V3.5.5 PoC 4조건 MUSHRA 평가에 사용:
  - 탑건 scene_topgun_category_eq/anchor_mushra.wav
  - 라라랜드 scene_lalaland_wonder/anchor_mushra.wav
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


CUTOFF_HZ = 3500.0
ORDER = 8


def _load_float64(path: Path) -> tuple[int, np.ndarray]:
    """wav → (sr, [samples, channels] float64)."""
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


def generate_low_anchor(
    original_wav: Path,
    output_wav: Path,
    cutoff_hz: float = CUTOFF_HZ,
    order: int = ORDER,
) -> dict:
    """원본을 cutoff_hz LPF로 통과시켜 MUSHRA low anchor 저장.

    Args:
        original_wav: 원본 wav (48 kHz stereo 권장, int16)
        output_wav:   저장 경로
        cutoff_hz:    LPF cutoff (기본 3500 Hz, MUSHRA 표준)
        order:        Butterworth 차수 (기본 8)

    Returns:
        {sr, samples, peak_db, rms_db, cutoff_hz, order}
    """
    sr, audio = _load_float64(original_wav)

    # SOS(Second-Order Sections) 형식으로 필터 설계 — 수치 안정성
    sos = butter(order, cutoff_hz, btype="low", fs=sr, output="sos")
    # 채널별 forward filter (group delay 있음)
    filtered = sosfilt(sos, audio, axis=0)

    peak_lin = float(np.abs(filtered).max())
    peak_db = _db(peak_lin)
    rms_db = _db(float(np.sqrt(np.mean(filtered ** 2))))

    _save_int16(filtered, sr, output_wav)

    return {
        "input": str(original_wav),
        "output": str(output_wav),
        "sr": sr,
        "samples": audio.shape[0],
        "peak_db": peak_db,
        "rms_db": rms_db,
        "cutoff_hz": cutoff_hz,
        "order": order,
    }


def verify_anchor_spectrum(
    anchor_wav: Path,
    cutoff_hz: float = CUTOFF_HZ,
    threshold_db: float = -40.0,
) -> dict:
    """FFT로 cutoff_hz 이상 대역의 최대 spectral energy가 threshold_db 이하인지 검증.

    MUSHRA low anchor의 효과(고역 손실)가 실제로 spectrum 상에 존재하는지 확인.
    """
    sr, audio = _load_float64(anchor_wav)
    # 모노 합산해서 단일 spectrum 분석
    mono = audio.mean(axis=1)

    # FFT (양 spectrum, magnitude in dBFS reference)
    n = len(mono)
    spec = np.fft.rfft(mono * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    # Normalize: 0 dBFS = 1.0 amplitude after windowing/scaling
    mag = np.abs(spec) / (n / 2)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-10))

    # 측정 대역 (cutoff 이상)
    above_cutoff_mask = freqs >= cutoff_hz
    if not above_cutoff_mask.any():
        return {"status": "no_bins_above_cutoff", "cutoff_hz": cutoff_hz}

    above_max_db = float(mag_db[above_cutoff_mask].max())
    above_freq_at_max = float(freqs[above_cutoff_mask][mag_db[above_cutoff_mask].argmax()])

    # cutoff 직전 대역의 최대치 (passband 보존 확인)
    passband_mask = (freqs >= 1000) & (freqs < cutoff_hz - 200)
    passband_max_db = float(mag_db[passband_mask].max()) if passband_mask.any() else -999.0

    return {
        "sr": sr,
        "samples": n,
        "cutoff_hz": cutoff_hz,
        "above_cutoff_max_db": above_max_db,
        "above_cutoff_max_freq_hz": above_freq_at_max,
        "passband_max_db (1k-3.3k)": passband_max_db,
        "stopband_attenuation_db": passband_max_db - above_max_db,
        "threshold_db": threshold_db,
        "passes": above_max_db <= threshold_db,
    }


# ────────────────────────────────────────────────────────
# CLI: V3.5.5 PoC 평가 4조건 MUSHRA 셋업
# ────────────────────────────────────────────────────────
REPO         = Path(__file__).resolve().parent.parent
MUSHRA_AUDIO = REPO / "evaluation" / "webmushra" / "configs" / "resources" / "audio"

POC_SCENES = [
    "scene_topgun_category_eq",
    "scene_lalaland_wonder",
]


def main() -> None:
    print("=" * 64)
    print("V3.5.5 PoC — MUSHRA Low Anchor 생성 (3.5 kHz Butterworth 8차 LPF)")
    print("=" * 64)

    for scene in POC_SCENES:
        print(f"\n--- {scene} ---")
        original = MUSHRA_AUDIO / scene / "original.wav"
        anchor   = MUSHRA_AUDIO / scene / "anchor_mushra.wav"

        if not original.exists():
            print(f"  ⚠ 원본 누락: {original}")
            continue

        gen = generate_low_anchor(original, anchor)
        print(f"  ✓ 생성: {anchor.relative_to(REPO)}")
        print(f"    sr={gen['sr']}  samples={gen['samples']}  "
              f"peak={gen['peak_db']:+.2f} dBFS  rms={gen['rms_db']:+.2f}")

        ver = verify_anchor_spectrum(anchor)
        status = "✓ PASS" if ver["passes"] else "✗ FAIL"
        print(f"    FFT 검증 [{status}]")
        print(f"      cutoff {ver['cutoff_hz']:.0f} Hz")
        print(f"      passband (1k-3.3k) max: {ver['passband_max_db (1k-3.3k)']:+.2f} dBFS")
        print(f"      stopband (≥{ver['cutoff_hz']:.0f}) max: {ver['above_cutoff_max_db']:+.2f} dBFS"
              f" (@{ver['above_cutoff_max_freq_hz']:.0f} Hz)")
        print(f"      stopband attenuation: {ver['stopband_attenuation_db']:.1f} dB"
              f" (threshold ≤ {ver['threshold_db']:.0f} dB at >cutoff)")


if __name__ == "__main__":
    main()
