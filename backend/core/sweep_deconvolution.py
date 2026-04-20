"""
sweep deconvolution → ref_rir.wav 추출

recorded.wav (마이크 녹음) + sweep.wav (원본 신호)
→ deconvolution → ref_rir.wav (실제 Room Impulse Response)
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def deconvolve_sweep(
    recorded_path: str,
    sweep_path: str,
    output_path: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    sweep deconvolution으로 RIR 추출

    Args:
        recorded_path : 마이크 녹음 wav 경로 (recorded.wav)
        sweep_path    : sweep 원본 wav 경로 (sweep.wav)
        output_path   : 결과 저장 경로 (ref_rir.wav)
        normalize     : 출력 정규화 여부

    Returns:
        ref_rir : (N,) numpy array
    """
    # ── 1. 파일 로드 ──────────────────────────────────────────────
    recorded, sr_rec = sf.read(recorded_path)
    sweep, sr_sw     = sf.read(sweep_path)

    # 모노 변환
    if recorded.ndim > 1:
        recorded = recorded[:, 0]
    if sweep.ndim > 1:
        sweep = sweep[:, 0]

    # 샘플레이트 일치 확인
    if sr_rec != sr_sw:
        raise ValueError(
            f"샘플레이트 불일치: recorded={sr_rec}Hz, sweep={sr_sw}Hz\n"
            "두 파일의 샘플레이트가 같아야 합니다."
        )

    sr = sr_rec
    print(f"로드 완료: recorded={len(recorded)/sr:.2f}s, sweep={len(sweep)/sr:.2f}s, sr={sr}Hz")

    # ── 2. FFT 기반 deconvolution ─────────────────────────────────
    # RIR = IFFT(FFT(recorded) / FFT(sweep))
    # 단, sweep의 FFT가 0에 가까운 부분은 regularization으로 안정화

    n_fft = len(recorded) + len(sweep) - 1
    # 2의 거듭제곱으로 올림 (FFT 속도 최적화)
    n_fft = 1 << (n_fft - 1).bit_length()

    rec_fft   = np.fft.rfft(recorded, n=n_fft)
    sweep_fft = np.fft.rfft(sweep,    n=n_fft)

    # Wiener deconvolution (regularization으로 노이즈 억제)
    sweep_power = np.abs(sweep_fft) ** 2
    noise_floor = 1e-6 * sweep_power.max()  # regularization 계수
    h_fft = rec_fft * np.conj(sweep_fft) / (sweep_power + noise_floor)

    rir_full = np.fft.irfft(h_fft)

    # ── 3. RIR 자르기 ─────────────────────────────────────────────
    # 최대값 위치 기준으로 앞뒤 잘라서 의미있는 RIR만 추출
    max_idx = np.argmax(np.abs(rir_full))

    # 최대값 앞 2ms, 뒤 최대 2초
    pre_samples  = int(0.002 * sr)
    post_samples = int(2.0   * sr)

    start = max(0, max_idx - pre_samples)
    end   = min(len(rir_full), max_idx + post_samples)
    ref_rir = rir_full[start:end].astype(np.float32)

    print(f"RIR 추출 완료: {len(ref_rir)/sr:.3f}s ({len(ref_rir)} samples)")

    # ── 4. 정규화 ─────────────────────────────────────────────────
    if normalize:
        peak = np.max(np.abs(ref_rir))
        if peak > 0:
            ref_rir = ref_rir / peak * 0.9

    # ── 5. 저장 ───────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, ref_rir, sr, subtype="FLOAT")
    print(f"ref_rir.wav 저장 완료: {output_path}")

    return ref_rir
