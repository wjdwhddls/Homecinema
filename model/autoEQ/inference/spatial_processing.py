"""spatial_processing.py — Mid/Side Processing + Sidechain Ducking (V3.5.6).

트랙 5+ Phase X4. V3.5.5 pipeline에 추가되는 perceptual enhancement 모듈.

적용 대상: no_vocals 스템 (music + SFX 혼합, mid 성분에 vocal spill 존재)
  - Mid 감쇠 → 대사 공간 양보 (대사는 center=mid에 위치)
  - Side 강화 → 분위기 확장
  - Sidechain ducking → 대사 있는 구간만 2-5 kHz 동적 감쇠 (정적 EQ 대비 효율적)

학술 근거: Liu & Reiss (2025, JAES Vol.73 No.10)
실무 통설: music stem mid -1~1.5 dB 감쇠 시 dialogue 명료도 극적 향상.

V3.5.5는 static vocals EQ (+2.5 dB B7, +2 dB B8)로 항상 boost.
V3.5.6은 envelope 기반 동적 ducking으로 대사 없을 때는 음악 full energy 보장.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


# ────────────────────────────────────────────────────────
# Mid/Side Processing
# ────────────────────────────────────────────────────────
def apply_mid_side_processing(
    audio_stereo: np.ndarray,
    mid_gain_db: float = -1.0,
    side_gain_db: float = +1.5,
) -> np.ndarray:
    """Stereo 신호를 M/S 도메인에서 게인 조정 후 L/R로 복원.

    공식:
        M = (L + R) / 2          (center 공통 성분)
        S = (L - R) / 2          (side 차이 성분)
        L' = M * g_m + S * g_s   (복원)
        R' = M * g_m - S * g_s

    Args:
        audio_stereo: shape (2, N). 채널 2개여야 함. (N, 2)이면 먼저 전치.
        mid_gain_db:  mid 성분 게인 (dB). 기본 -1.0 (대사 공간 양보)
        side_gain_db: side 성분 게인 (dB). 기본 +1.5 (분위기 확장)

    Returns:
        shape (2, N), 동일 포맷.

    Raises:
        ValueError: 입력이 stereo 아닐 때.
    """
    arr = np.asarray(audio_stereo)
    if arr.ndim != 2 or (arr.shape[0] != 2 and arr.shape[-1] != 2):
        raise ValueError(f"expected stereo, got shape {arr.shape}")
    if arr.shape[0] != 2 and arr.shape[-1] == 2:
        arr = arr.T

    L, R = arr[0], arr[1]
    M = (L + R) * 0.5
    S = (L - R) * 0.5

    g_m = 10.0 ** (mid_gain_db / 20.0)
    g_s = 10.0 ** (side_gain_db / 20.0)

    M_adj = M * g_m
    S_adj = S * g_s

    L_out = M_adj + S_adj
    R_out = M_adj - S_adj
    return np.stack([L_out, R_out], axis=0)


# ────────────────────────────────────────────────────────
# Envelope Follower
# ────────────────────────────────────────────────────────
def compute_rms_envelope(
    audio: np.ndarray,
    sample_rate: int,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
) -> np.ndarray:
    """RMS envelope follower (one-pole attack/release).

    시간 상수로부터 alpha 계산: alpha = 1 - exp(-1 / (sr * tau_sec))
    Attack은 envelope 증가 시 (빠름), release는 감소 시 (느림).

    Args:
        audio:       mono 입력 (1D) 또는 stereo (2, N). stereo면 채널 평균 후 처리.
        sample_rate: 샘플 레이트
        attack_ms:   envelope 상승 시간 (기본 10 ms)
        release_ms:  envelope 하강 시간 (기본 150 ms)

    Returns:
        envelope 1D array (동일 샘플 수), linear amplitude 스케일.
    """
    a = np.asarray(audio)
    if a.ndim == 2:
        a = a.mean(axis=0)

    abs_a = np.abs(a).astype(np.float64)
    env = np.zeros_like(abs_a)

    # 시간 상수 → 계수
    alpha_attack  = 1.0 - np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
    alpha_release = 1.0 - np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))

    prev = 0.0
    for i in range(len(abs_a)):
        x = abs_a[i]
        alpha = alpha_attack if x > prev else alpha_release
        prev = alpha * x + (1.0 - alpha) * prev
        env[i] = prev

    return env


# ────────────────────────────────────────────────────────
# Sidechain Ducking (multiband 2-5 kHz)
# ────────────────────────────────────────────────────────
def apply_sidechain_ducking(
    no_vocals: np.ndarray,
    vocals: np.ndarray,
    sample_rate: int,
    threshold_db: float = -30.0,
    ratio: float = 4.0,
    max_reduction_db: float = 6.0,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
    band_low_hz: float = 2000.0,
    band_high_hz: float = 5000.0,
    filter_order: int = 4,
) -> np.ndarray:
    """Vocals envelope 기반으로 no_vocals의 2-5 kHz 대역을 동적 감쇠.

    처리 흐름:
        1) vocals envelope 계산 (RMS follower)
        2) envelope_dB → gain reduction (threshold 초과분 × (1 - 1/ratio))
        3) no_vocals 대역 분리: 2-5 kHz bandpass + 나머지 complement
        4) 대역에 샘플별 gain reduction 적용
        5) 복원: ducked_band + rest

    Args:
        no_vocals:       shape (2, N) stereo
        vocals:          shape (2, N) stereo — sidechain 입력
        sample_rate:     샘플 레이트
        threshold_db:    envelope dB가 이 값 초과 시 ducking 발동 (기본 -30 dB)
        ratio:           compression ratio (기본 4:1)
        max_reduction_db: gain reduction 상한 (기본 6 dB — 과도 축소 방지)
        attack_ms/release_ms: envelope follower 시간 상수
        band_low_hz/band_high_hz: ducking 대상 대역 (기본 2-5 kHz, voice critical)
        filter_order:    Butterworth 차수 (기본 4)

    Returns:
        shape (2, N), ducking 적용된 no_vocals.

    Notes:
        - no_vocals/vocals 길이 일치 필요. 짧은 쪽에 맞춤.
        - 대역 밖 (저/고음)은 영향 없음 → 음악의 bass/bright 보존.
    """
    nov = np.asarray(no_vocals)
    voc = np.asarray(vocals)
    if nov.ndim != 2 or nov.shape[0] != 2:
        raise ValueError(f"no_vocals must be (2, N), got {nov.shape}")
    if voc.ndim != 2 or voc.shape[0] != 2:
        raise ValueError(f"vocals must be (2, N), got {voc.shape}")

    L = min(nov.shape[1], voc.shape[1])
    nov = nov[:, :L]
    voc = voc[:, :L]

    # 1) vocals envelope (linear amplitude)
    env = compute_rms_envelope(voc, sample_rate, attack_ms, release_ms)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-10))

    # 2) gain reduction (dB). 0 when below threshold.
    over = env_db - threshold_db
    over = np.maximum(over, 0.0)
    reduction_db = over * (1.0 - 1.0 / ratio)
    reduction_db = np.minimum(reduction_db, max_reduction_db)
    gain_lin = 10.0 ** (-reduction_db / 20.0)  # negative gain = attenuation

    # 3) bandpass no_vocals 2-5 kHz
    sos_bp = butter(
        filter_order,
        [band_low_hz, band_high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    band = sosfilt(sos_bp, nov, axis=1)
    rest = nov - band  # complement (band-reject equivalent)

    # 4) apply sample-wise gain to band (both channels)
    band_ducked = band * gain_lin[np.newaxis, :]

    # 5) recombine
    return rest + band_ducked


# ────────────────────────────────────────────────────────
# 검증
# ────────────────────────────────────────────────────────
def _verify_mid_side() -> None:
    """M/S 처리 unit tests."""
    # 1) 단일 게인 0,0 → 원본 보존 (round-trip 정확도)
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, 48000)).astype(np.float32) * 0.3
    y = apply_mid_side_processing(x, mid_gain_db=0.0, side_gain_db=0.0)
    assert np.allclose(y, x, atol=1e-6), f"round-trip broken: max diff {np.abs(y-x).max()}"

    # 2) mid -∞ (무한 감쇠 근사) → mid 제거됨 (L-R만 남음)
    y = apply_mid_side_processing(x, mid_gain_db=-60.0, side_gain_db=0.0)
    # L_out = M*g_m + S ≈ 0 + S, R_out = M*g_m - S ≈ 0 - S
    # 즉 L_out + R_out ≈ 0 (mid 제거)
    mid_residual = np.abs(y[0] + y[1]).max()
    assert mid_residual < 0.01, f"mid not sufficiently reduced: {mid_residual}"

    # 3) side = 0 → mono 화 (L = R)
    y = apply_mid_side_processing(x, mid_gain_db=0.0, side_gain_db=-120.0)
    diff = np.abs(y[0] - y[1]).max()
    assert diff < 1e-4, f"not mono: L-R max = {diff}"

    # 4) 기본 파라미터 (-1, +1.5) → 에너지 변화 작음 (±2 dB 이내)
    y = apply_mid_side_processing(x, mid_gain_db=-1.0, side_gain_db=+1.5)
    rms_in  = float(np.sqrt(np.mean(x ** 2)))
    rms_out = float(np.sqrt(np.mean(y ** 2)))
    assert 0.8 < (rms_out / rms_in) < 1.3, f"unexpected energy change: {rms_out/rms_in:.3f}"

    print("  [OK] M/S: round-trip / mid-kill / mono-collapse / default 4 tests passed")


def _verify_envelope() -> None:
    """Envelope follower unit tests."""
    sr = 48000
    # 무음 → 무음
    z = np.zeros((2, sr))
    env = compute_rms_envelope(z, sr)
    assert env.max() < 1e-8, f"envelope of silence != 0: {env.max()}"

    # Step 입력 (1초 동안 0 → 0.5로 점프) → attack ramp
    x = np.zeros((2, sr * 2))
    x[:, sr:] = 0.5
    env = compute_rms_envelope(x, sr, attack_ms=10.0, release_ms=150.0)
    # 10ms 후 envelope는 0.5의 1-1/e ≈ 63.2%에 근접
    idx_10ms = sr + int(sr * 0.01)
    approx = env[idx_10ms]
    assert 0.25 < approx < 0.45, f"attack at 10ms unexpected: {approx:.3f} (expected ~0.315)"
    # 충분한 시간 후 envelope는 0.5에 수렴
    assert env[-1] > 0.45, f"envelope did not reach target: {env[-1]:.3f}"

    # Release: step from 0.5 to 0 → release decay
    x = np.zeros((2, sr * 2))
    x[:, :sr] = 0.5
    env = compute_rms_envelope(x, sr, attack_ms=10.0, release_ms=150.0)
    # release 150ms 후: ~37% of 0.5 = 0.185 (1/e)
    idx_150ms = sr + int(sr * 0.15)
    approx = env[idx_150ms]
    assert 0.12 < approx < 0.25, f"release at 150ms unexpected: {approx:.3f} (expected ~0.184)"

    print("  [OK] envelope: silence / attack@10ms / release@150ms 3 tests passed")


def _verify_sidechain() -> None:
    """Sidechain ducking unit tests."""
    sr = 48000
    rng = np.random.default_rng(1)

    # 1) 무음 vocals → ducking 없음 (no_vocals 변화 최소)
    nov = rng.standard_normal((2, sr * 2)).astype(np.float32) * 0.2
    silent_voc = np.zeros((2, sr * 2), dtype=np.float32)
    out = apply_sidechain_ducking(nov, silent_voc, sr)
    # 모든 gain=0 이지만 bandpass+complement round-trip은 ±1e-12 정도 오차
    diff = np.abs(out - nov).max()
    assert diff < 1e-3, f"silent vocals should not change no_vocals: diff {diff}"

    # 2) 강한 vocals (envelope >> threshold) → 2-5 kHz 대역 감쇠 적용됨
    t = np.arange(sr * 2) / sr
    # 3 kHz 톤 (vocals) + 3 kHz 톤 (no_vocals → 대역 내)
    loud_voc = np.stack([0.5 * np.sin(2 * np.pi * 3000 * t)] * 2)
    nov_3k = np.stack([0.3 * np.sin(2 * np.pi * 3000 * t)] * 2).astype(np.float32)
    out = apply_sidechain_ducking(nov_3k, loud_voc, sr)
    # 정상 동작 시 out의 RMS가 nov_3k보다 작아야 (대역 감쇠됨)
    rms_in  = float(np.sqrt(np.mean(nov_3k ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    assert rms_out < rms_in, f"ducking did not reduce in-band RMS: in={rms_in:.4f} out={rms_out:.4f}"

    # 3) 저역 (100 Hz) → 대역 밖 → 변화 최소
    nov_lo = np.stack([0.3 * np.sin(2 * np.pi * 100 * t)] * 2).astype(np.float32)
    out = apply_sidechain_ducking(nov_lo, loud_voc, sr)
    rms_in  = float(np.sqrt(np.mean(nov_lo ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    # 100 Hz는 2-5kHz 대역 밖이라 거의 원본 유지 (filter leak 약간 허용)
    assert abs(rms_out - rms_in) / rms_in < 0.05, \
        f"out-of-band should be unchanged: in={rms_in:.4f} out={rms_out:.4f}"

    # 4) max_reduction_db 상한 준수 (극단적 입력)
    huge_voc = np.stack([np.ones(sr * 2) * 0.9] * 2)
    out = apply_sidechain_ducking(nov_3k, huge_voc, sr, max_reduction_db=3.0)
    # 3 dB reduction → linear 0.707, nov_3k in-band RMS ≈ 0.21 → out RMS ≈ 0.15
    rms_in  = float(np.sqrt(np.mean(nov_3k ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    ratio = rms_out / rms_in
    # 3dB 감쇠 → 0.707; 허용 0.6~0.85 (bandpass edge effects)
    assert 0.6 < ratio < 0.85, f"max_reduction cap violated: ratio={ratio:.3f}"

    print("  [OK] sidechain: silent-voc / in-band duck / out-of-band preserve / cap 4 tests passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Spatial Processing (V3.5.6) 검증")
    print("=" * 60)
    print("\n[1/3] M/S processing")
    _verify_mid_side()
    print("\n[2/3] RMS envelope follower")
    _verify_envelope()
    print("\n[3/3] Sidechain ducking")
    _verify_sidechain()
    print("\n🎉 모든 검증 통과")
