"""playback.py — pedalboard 기반 EQ 적용.

작업 10, 10-2, 14 (Day 4~5, 8~9):
- build_eq_chain: 10밴드 게인 → pedalboard 필터 체인
- apply_eq_to_file: 단일 EQ를 파일 전체에 적용
- apply_timevarying_eq: 씬별 EQ + 크로스페이드를 시간축으로 적용
- should_apply_eq: 무음 / 저신뢰도 엣지 케이스 처리
"""

from __future__ import annotations

import numpy as np
from pedalboard import Pedalboard, PeakFilter
from pedalboard.io import AudioFile

from .eq_engine import BAND_FREQS, BAND_Q
from .smoothing import get_eq_at_time


# ────────────────────────────────────────────────────────
# EQ 체인 생성
# ────────────────────────────────────────────────────────
def build_eq_chain(gains: np.ndarray) -> Pedalboard:
    """10밴드 EQ gains로 pedalboard 필터 체인 생성."""
    filters = []
    for freq, gain, q in zip(BAND_FREQS, gains, BAND_Q):
        if abs(gain) > 0.01:
            filters.append(
                PeakFilter(cutoff_frequency_hz=freq, gain_db=float(gain), q=q)
            )
    return Pedalboard(filters)


# ────────────────────────────────────────────────────────
# 단일 EQ 적용
# ────────────────────────────────────────────────────────
def apply_eq_to_file(input_wav, output_wav, gains, prevent_clipping: bool = True):
    """원본 wav에 단일 EQ를 적용해서 새 wav로 저장."""
    with AudioFile(str(input_wav)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    eq_chain = build_eq_chain(gains)
    processed = eq_chain(audio, sr)

    if prevent_clipping:
        peak = np.abs(processed).max()
        if peak > 0.99:
            scale = 0.95 / peak
            processed = processed * scale
            print(f"    ⚠️ 클리핑 방지 정규화: peak {peak:.3f} → 0.95")

    with AudioFile(str(output_wav), "w", sr, audio.shape[0]) as f:
        f.write(processed)

    return sr, audio, processed


# ────────────────────────────────────────────────────────
# 시간축 EQ 적용
# ────────────────────────────────────────────────────────
def apply_timevarying_eq(
    input_wav, output_wav, scenes_eq: list[dict],
    block_sec: float = 0.5, prevent_clipping: bool = True,
):
    """씬별 effective_gains + 크로스페이드를 시간축으로 보간하며 EQ 적용.

    block_sec 단위로 EQ 체인을 갱신. 0.5초마다 EQ가 바뀌므로
    크로스페이드 시간(0.3~2초) 안에서 부드럽게 변함.

    Args:
        scenes_eq: [{start_sec, end_sec, transition_out, effective_gains}, ...]
                   effective_gains는 길이 10의 np.array
    """
    with AudioFile(str(input_wav)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    block_samples = int(block_sec * sr)
    out = np.zeros_like(audio)

    for start in range(0, audio.shape[-1], block_samples):
        end = min(start + block_samples, audio.shape[-1])
        t_center = (start + end) / 2 / sr

        gains = get_eq_at_time(t_center, scenes_eq)
        chain = build_eq_chain(gains)

        block = audio[..., start:end]
        out[..., start:end] = chain(block, sr)

    if prevent_clipping:
        peak = np.abs(out).max()
        if peak > 0.99:
            out = out * (0.95 / peak)
            print(f"    ⚠️ 클리핑 방지: peak {peak:.3f} → 0.95")

    with AudioFile(str(output_wav), "w", sr, audio.shape[0]) as f:
        f.write(out)


# ────────────────────────────────────────────────────────
# 엣지 케이스 (무음 / 저신뢰도)
# ────────────────────────────────────────────────────────
def rms_dbfs(audio_segment: np.ndarray) -> float:
    """RMS 에너지를 dBFS로. -inf는 -100으로 clip."""
    rms = np.sqrt(np.mean(audio_segment ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def should_apply_eq(
    scene: dict,
    audio_data: np.ndarray,
    sr: int,
    confidence_threshold: float = 0.3,
    silence_threshold_dbfs: float = -50,
    window_sec: float = 4.0,
) -> tuple[bool, str]:
    """EQ 적용 여부 판정.

    Returns: (should_apply, reason) — reason은 'ok'/'silence'/'low_confidence'/'too_short'
    """
    start_sample = int(scene["start_sec"] * sr)
    end_sample = int(scene["end_sec"] * sr)
    segment = (
        audio_data[start_sample:end_sample]
        if audio_data.ndim == 1
        else audio_data[:, start_sample:end_sample].mean(axis=0)
    )

    window_samples = int(window_sec * sr)
    window_rms_values = []
    for i in range(0, len(segment), window_samples):
        window = segment[i : i + window_samples]
        if len(window) > window_samples * 0.5:
            window_rms_values.append(rms_dbfs(window))

    if not window_rms_values:
        return False, "too_short"
    if max(window_rms_values) < silence_threshold_dbfs:
        return False, "silence"

    if "mood_probs_mean" in scene.get("aggregated", {}):
        max_prob = max(scene["aggregated"]["mood_probs_mean"].values())
        if max_prob < confidence_threshold:
            return False, "low_confidence"

    return True, "ok"


# ────────────────────────────────────────────────────────
# 검증
# ────────────────────────────────────────────────────────
def verify_timevarying_eq() -> None:
    """시간축 EQ 적용이 에러 없이 동작하는지 + 길이 보존."""
    import os
    from pathlib import Path

    from .eq_engine import EQ_PRESETS
    from .utils import get_audio_info

    sr = 48000
    duration = 10
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    Path("/tmp/_test_input.wav").parent.mkdir(parents=True, exist_ok=True)
    with AudioFile("/tmp/_test_input.wav", "w", sr, 1) as f:
        f.write(test_audio.reshape(1, -1))

    scenes_eq = [
        {"start_sec": 0, "end_sec": 5, "transition_out": "cut",
         "effective_gains": EQ_PRESETS["Peacefulness"]},
        {"start_sec": 5, "end_sec": 10, "transition_out": "cut",
         "effective_gains": EQ_PRESETS["Tension"]},
    ]

    apply_timevarying_eq("/tmp/_test_input.wav", "/tmp/_test_output.wav", scenes_eq)
    assert Path("/tmp/_test_output.wav").exists()

    info = get_audio_info("/tmp/_test_output.wav")
    assert abs(info["duration"] - 10.0) < 0.1

    os.remove("/tmp/_test_input.wav")
    os.remove("/tmp/_test_output.wav")
    print(f"  ✓ 시간축 EQ 적용 함수 정상 동작 (길이 보존, 클리핑 처리)")


def verify_edge_cases() -> None:
    from .eq_engine import EQ_PRESETS

    sr = 48000
    duration = 10

    # 무음
    silent = np.zeros(sr * duration)
    scene = {"start_sec": 0, "end_sec": duration}
    ok, reason = should_apply_eq(scene, silent, sr)
    assert not ok and reason == "silence"

    # 정상 톤
    t = np.linspace(0, duration, sr * duration)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    ok, reason = should_apply_eq(scene, tone, sr)
    assert ok

    # 낮은 신뢰도
    scene_low = {
        "start_sec": 0, "end_sec": duration,
        "aggregated": {"mood_probs_mean": {k: 1 / 7 for k in EQ_PRESETS.keys()}},
    }
    ok, reason = should_apply_eq(scene_low, tone, sr)
    assert not ok and reason == "low_confidence"

    # 높은 신뢰도
    scene_high = {
        "start_sec": 0, "end_sec": duration,
        "aggregated": {"mood_probs_mean": {"Tension": 0.8, "Power": 0.2}},
    }
    ok, reason = should_apply_eq(scene_high, tone, sr)
    assert ok

    print(f"  ✓ 무음 / 정상 / 낮은 신뢰도 / 높은 신뢰도 모두 정상")


if __name__ == "__main__":
    print("=" * 60)
    print("Playback 검증")
    print("=" * 60)
    print("\n[시간축 EQ 적용]")
    verify_timevarying_eq()
    print("\n[엣지 케이스]")
    verify_edge_cases()
    print("\n🎉 Playback 모든 검증 통과")
