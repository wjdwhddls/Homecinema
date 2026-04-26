"""
EQ 분석 모듈

sweep 원본 + 마이크 녹음 → Transfer Function → 23밴드 보정값 계산

원본 코드(Colab)에서 수정된 내용:
  - librosa.load → soundfile (의존성 경량화)
  - matplotlib 제거
  - Google Drive 경로 → bytes 입력으로 변환
  - get_calibration_values의 전역변수 버그 수정 (freqs 파라미터로 명시)
"""

from __future__ import annotations

import io
import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import List, Tuple, Dict, Any


# ── 1. 오디오 로드 ───────────────────────────────────────────────

def load_audio_bytes(data: bytes, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    bytes → (samples, sr) 반환
    soundfile이 지원하는 포맷 (wav, m4a 불가 → wav 변환 필요)
    """
    buf = io.BytesIO(data)
    y, sr = sf.read(buf, always_2d=False)

    # 스테레오 → 모노
    if y.ndim > 1:
        y = y.mean(axis=1)

    # 리샘플링 (target_sr과 다를 경우)
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        y = resample_poly(y, target_sr // g, sr // g)
        sr = target_sr

    return y.astype(np.float32), sr


# ── 2. 시간 정렬 ─────────────────────────────────────────────────

def align_audio(original: np.ndarray, recorded: np.ndarray) -> np.ndarray:
    """
    Cross-correlation으로 두 신호의 시간 오프셋을 찾아 정렬.
    """
    correlation = signal.correlate(recorded, original, mode='full')
    offset = int(np.argmax(correlation)) - (len(original) - 1)

    if offset >= 0:
        aligned = recorded[offset: offset + len(original)]
    else:
        aligned = np.pad(recorded, (abs(offset), 0))[:len(original)]

    if len(aligned) < len(original):
        aligned = np.pad(aligned, (0, len(original) - len(aligned)))

    return aligned.astype(np.float32)


# ── 3. Inverse Sweep 생성 ────────────────────────────────────────

def generate_inverse_sweep(
    sweep: np.ndarray, sr: int, f_start: float = 1.0, f_end: float = 20000.0
) -> np.ndarray:
    """ESS(Exponential Sine Sweep)용 inverse sweep 생성"""
    T = len(sweep) / sr
    t = np.linspace(0, T, len(sweep), endpoint=False)
    R = np.log(f_end / f_start)
    envelope = np.exp(t * R / T)
    inverse = sweep[::-1] / (envelope + 1e-10)
    scale = np.max(np.abs(sweep)) + 1e-10
    return (inverse / scale).astype(np.float32)


# ── 4. Transfer Function 계산 ────────────────────────────────────

def fractional_octave_smooth_log(
    freqs: np.ndarray, values: np.ndarray, fraction: float = 1 / 6
) -> np.ndarray:
    """log-frequency 기반 octave smoothing (cumsum O(N))"""
    smoothed = np.copy(values)
    valid_idx = np.where(freqs > 0)[0]
    if len(valid_idx) == 0:
        return smoothed

    log_freqs   = np.log2(freqs[valid_idx])
    valid_values = values[valid_idx]
    half_band   = fraction / 2

    left_indices  = np.searchsorted(log_freqs, log_freqs - half_band, side='left')
    right_indices = np.searchsorted(log_freqs, log_freqs + half_band, side='right')
    cumsum = np.concatenate(([0.0], np.cumsum(valid_values)))

    for i in range(len(valid_idx)):
        lo, hi = left_indices[i], right_indices[i]
        if hi > lo:
            smoothed[valid_idx[i]] = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    return smoothed


def compute_transfer_function(
    y_org: np.ndarray,
    y_rec: np.ndarray,
    sr: int = 44100,
    smooth_octave: float = 1 / 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ESS 기반 Transfer Function 계산.

    Returns:
        freqs : 주파수 배열 (Hz)
        H_db  : 레벨 차이 배열 (dB)
    """
    min_len = min(len(y_org), len(y_rec))
    y_org = y_org[:min_len]
    y_rec = y_rec[:min_len]

    inverse_sweep = generate_inverse_sweep(y_org, sr)
    ir = signal.fftconvolve(y_rec, inverse_sweep, mode='full')

    # IR 게이트 — 직접음 위치(center) 앞 5ms ~ 뒤 200ms 만 추출.
    # 이전 코드는 center/end를 계산만 하고 ir 전체를 FFT해서 잔향/노이즈/pre-ringing이
    # 모두 spectrum에 섞였고, mid_band median 정규화와 결합해 모든 밴드가 saturation됨.
    center       = int(np.argmax(np.abs(ir)))
    pre_samples  = int(sr * 5   / 1000)   # 5 ms
    post_samples = int(sr * 200 / 1000)   # 200 ms
    start = max(0, center - pre_samples)
    end   = min(len(ir), center + post_samples)
    ir_gated = ir[start:end]

    if len(ir_gated) < 16:
        raise ValueError(
            f"IR 게이트 결과가 너무 짧습니다 ({len(ir_gated)} 샘플). 녹음 정렬에 문제가 있을 수 있습니다."
        )

    # 꼬리 fade-out (잔향 끝의 cliff 효과 방지)
    fade_len = max(1, len(ir_gated) // 4)
    window = np.ones(len(ir_gated))
    window[-fade_len:] = signal.windows.hann(fade_len * 2)[-fade_len:]
    ir_gated = ir_gated * window

    fft_ir = np.fft.rfft(ir_gated)
    freqs  = np.fft.rfftfreq(len(ir_gated), d=1.0 / sr)
    mag    = np.abs(fft_ir)

    mid_band = (freqs > 500) & (freqs < 2000)
    ref = np.median(mag[mid_band]) if np.any(mid_band) else 1.0
    H_db = 20 * np.log10(mag / (ref + 1e-10))

    if smooth_octave is not None:
        H_db = fractional_octave_smooth_log(freqs, H_db, smooth_octave)

    # 다운샘플링 (속도 최적화)
    step  = 4
    freqs = freqs[::step]
    H_db  = H_db[::step]

    return freqs, H_db


# ── 5. 보정값 계산 ───────────────────────────────────────────────

# ISO 1/3 옥타브 23개 밴드 중심 주파수
ISO_BANDS = [
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
    1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
]


def get_calibration_values(
    freqs: np.ndarray,          # ← 버그 수정: 전역변수 대신 파라미터로 명시
    H_db: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Transfer Function → 23밴드 보정 gain 계산.

    Returns:
        List of { freq, theory_gain_db, actual_gain_db }
    """
    results = []
    for f_target in ISO_BANDS:
        response    = float(np.interp(f_target, freqs, H_db))
        theory_gain = -response
        soft_gain   = theory_gain * 0.6
        # ±6 dB는 BT 스피커+폰 마이크 frequency response 변동(±10 dB+)에 비해 좁아 saturation 발생.
        # 음향 EQ에서 ±9 dB는 일반적인 보정 범위.
        actual      = float(np.clip(soft_gain, -9.0, 9.0))
        if abs(actual) < 0.3:
            actual = 0.0
        results.append({
            "freq":           f_target,
            "theory_gain_db": round(theory_gain, 2),
            "actual_gain_db": round(actual, 2),
        })
    return results


# ── 6. Bass/Mid/Treble 요약 ──────────────────────────────────────

def _strength_label(gain: float) -> str:
    if gain >= 2.0:
        return "Strong"
    elif gain <= -2.0:
        return "Weak"
    return "Normal"


def generate_simple_settings(
    calibration_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """23밴드 결과 → Bass / Mid / Treble 평균"""
    def avg(lo, hi):
        gains = [r["actual_gain_db"] for r in calibration_results if lo <= r["freq"] < hi]
        return float(np.mean(gains)) if gains else 0.0

    bass    = avg(100,  250)
    mid     = avg(250,  4000)
    treble  = avg(4000, 20000)

    return {
        "bass":   {"gain_db": round(bass,   2), "label": _strength_label(bass)},
        "mid":    {"gain_db": round(mid,    2), "label": _strength_label(mid)},
        "treble": {"gain_db": round(treble, 2), "label": _strength_label(treble)},
    }


# ── 7. Parametric EQ ─────────────────────────────────────────────

def generate_parametric_eq(
    calibration_results: List[Dict[str, Any]],
    max_filters: int = 5,
) -> List[Dict[str, Any]]:
    """23밴드 결과 → Parametric EQ 필터 (최대 5개)"""
    low  = [r for r in calibration_results if r["freq"] <  300]
    mid  = [r for r in calibration_results if 300 <= r["freq"] < 3000]
    high = [r for r in calibration_results if r["freq"] >= 3000]

    picks = []
    for band, count in [(low, 2), (mid, 2), (high, 1)]:
        sorted_b = sorted(band, key=lambda x: abs(x["theory_gain_db"]), reverse=True)
        filtered = [b for b in sorted_b if abs(b["theory_gain_db"]) >= 1.0]
        picks.extend(filtered[:count])

    picks.sort(key=lambda x: x["freq"])

    result = []
    for p in picks[:max_filters]:
        freq  = p["freq"]
        gain  = float(np.clip(p["theory_gain_db"], -12.0, 12.0))
        Q     = 0.8 if freq < 300 else (1.4 if freq < 2000 else 2.0)
        result.append({"freq": freq, "gain_db": round(gain, 2), "Q": Q})

    return result


# ── 8. EQ 적용 후 WAV 저장 ───────────────────────────────────────

def apply_eq_and_save(
    audio_bytes: bytes,
    freqs: np.ndarray,
    H_db: np.ndarray,
    sr: int = 44100,
) -> bytes:
    """
    입력 오디오에 EQ 보정 적용 후 WAV bytes 반환.

    Args:
        audio_bytes : 원본 오디오 bytes (WAV)
        freqs       : compute_transfer_function의 freqs
        H_db        : compute_transfer_function의 H_db
        sr          : 샘플레이트

    Returns:
        EQ가 적용된 WAV bytes
    """
    y, _ = load_audio_bytes(audio_bytes, target_sr=sr)

    # 23밴드/Parametric과 동일한 범위로 통일.
    # RMS 매칭 + peak normalize가 뒤따르므로 ±9 dB로 늘려도 출력 클리핑 위험 없음.
    clamped_gain  = np.clip(H_db * -0.6, -9.0, 9.0)
    gain_linear   = 10 ** (clamped_gain / 20.0)

    n_fft    = len(y)
    Y        = np.fft.rfft(y, n_fft)
    f_axis   = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    interp_gain  = np.interp(f_axis, freqs, gain_linear)
    Y_corrected  = Y * interp_gain
    y_eq         = np.fft.irfft(Y_corrected, n_fft).astype(np.float32)

    # RMS 매칭 + 클리핑 방지
    rms_org = float(np.sqrt(np.mean(y ** 2))) + 1e-10
    rms_eq  = float(np.sqrt(np.mean(y_eq ** 2))) + 1e-10
    y_eq   *= rms_org / rms_eq
    peak    = float(np.max(np.abs(y_eq)))
    if peak > 1.0:
        y_eq /= peak

    buf = io.BytesIO()
    sf.write(buf, y_eq, sr, format='WAV', subtype='FLOAT')
    return buf.getvalue()


# ── 메인 파이프라인 ──────────────────────────────────────────────

def run_eq_pipeline(
    sweep_bytes: bytes,
    recorded_bytes: bytes,
    sr: int = 44100,
) -> Dict[str, Any]:
    """
    sweep + 녹음 → EQ 분석 결과 반환

    Args:
        sweep_bytes    : sweep 원본 WAV bytes
        recorded_bytes : 마이크 녹음 WAV bytes (최적 위치에서 측정)
        sr             : 처리 샘플레이트

    Returns:
        {
            "bands":      [...],   # 23밴드 보정값
            "simple":     {...},   # Bass/Mid/Treble 요약
            "parametric": [...],   # Parametric EQ 필터
        }
    """
    y_org, _ = load_audio_bytes(sweep_bytes,    target_sr=sr)
    y_rec, _ = load_audio_bytes(recorded_bytes, target_sr=sr)

    min_samples = int(sr * 0.5)
    if y_org.size < min_samples:
        raise ValueError(
            f"sweep 파일이 비어있거나 너무 짧습니다 ({y_org.size} 샘플). "
            "번들의 sweep.wav가 손상되지 않았는지 확인하세요."
        )
    if y_rec.size < min_samples:
        raise ValueError(
            f"녹음 파일이 비어있거나 너무 짧습니다 ({y_rec.size} 샘플). "
            "마이크 권한과 블루투스 스피커 연결을 확인한 뒤 다시 측정해주세요."
        )

    y_rec_aligned = align_audio(y_org, y_rec)

    # 진폭 정규화
    scale = float(np.max(np.abs(y_org))) + 1e-10
    y_org_n = y_org / scale
    y_rec_n = y_rec_aligned / scale

    freqs, H_db = compute_transfer_function(y_org_n, y_rec_n, sr=sr)
    H_db = H_db - float(np.mean(H_db))    # 평균 0 정규화

    bands      = get_calibration_values(freqs, H_db)
    simple     = generate_simple_settings(bands)
    parametric = generate_parametric_eq(bands)

    return {
        "bands":      bands,
        "simple":     simple,
        "parametric": parametric,
    }
