"""loudness_match.py — pyloudnorm 기반 ITU-R BS.1770-4 Integrated Loudness 매칭.

V3.5.5 PoC Step 4.7: A/B 청취 공정성을 위해 v3_5_5.wav를 v3_3.wav의 LUFS에 매칭.
스템별 EQ + 합산으로 v3_5_5가 v3_3보다 라우드해진 결과를 청취 비교 시점에 균등화.

방식:
  1) target wav의 integrated LUFS 측정
  2) input wav의 integrated LUFS 측정
  3) 차이만큼 선형 게인 적용 (pyln.normalize.loudness 사용)
  4) 결과의 peak가 ceiling(기본 -1.0 dBFS) 초과 시 추가 attenuation

기존 파일은 보존, 새 파일에 저장 (예: v3_5_5_matched.wav).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
from scipy.io import wavfile


def _load_float32(path: Path) -> tuple[int, np.ndarray]:
    """wav → (sr, [samples, channels] float64). pyloudnorm은 float64 권장."""
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


def measure_integrated_lufs(path: Path) -> dict:
    """파일 LUFS/peak 측정."""
    sr, a = _load_float32(path)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(a))
    peak_db = _db(float(np.abs(a).max()))
    return {"sr": sr, "samples": a.shape[0], "lufs": lufs, "peak_db": peak_db}


def match_loudness(
    input_wav: Path,
    target_lufs: float,
    output_wav: Path,
    peak_ceiling_db: float = -1.0,
) -> dict:
    """input을 target_lufs로 매칭해서 output_wav로 저장.

    Args:
        input_wav:        조정할 파일
        target_lufs:      목표 integrated LUFS (예: v3_3의 LUFS)
        output_wav:       저장 경로 (기존 파일 덮어쓰기 주의)
        peak_ceiling_db:  매칭 후 peak 상한 (기본 -1.0 dBFS).
                          매칭 결과가 이 한도 초과 시 추가 attenuation 적용.

    Returns:
        {original_lufs, target_lufs, gain_applied_db, final_lufs, final_peak_db,
         peak_attenuation_db, samples, sr}
    """
    sr, audio = _load_float32(input_wav)
    meter = pyln.Meter(sr)
    original_lufs = float(meter.integrated_loudness(audio))

    gain_db = target_lufs - original_lufs
    matched = audio * (10.0 ** (gain_db / 20.0))

    peak_lin = float(np.abs(matched).max())
    peak_db  = _db(peak_lin)

    peak_atten_db = 0.0
    if peak_db > peak_ceiling_db:
        peak_atten_db = peak_ceiling_db - peak_db
        matched = matched * (10.0 ** (peak_atten_db / 20.0))

    final_lufs = float(meter.integrated_loudness(matched))
    final_peak_db = _db(float(np.abs(matched).max()))

    _save_int16(matched, sr, output_wav)

    return {
        "input":              str(input_wav),
        "output":             str(output_wav),
        "sr":                 sr,
        "samples":            audio.shape[0],
        "original_lufs":      original_lufs,
        "target_lufs":        target_lufs,
        "gain_applied_db":    gain_db + peak_atten_db,
        "raw_match_gain_db":  gain_db,
        "peak_attenuation_db": peak_atten_db,
        "final_lufs":         final_lufs,
        "final_peak_db":      final_peak_db,
    }


# ────────────────────────────────────────────────────────
# CLI 진입점 — V3.5.5 PoC Step 4.7
# ────────────────────────────────────────────────────────
REPO         = Path(__file__).resolve().parent.parent
MUSHRA_AUDIO = REPO / "evaluation" / "webmushra" / "configs" / "resources" / "audio"

POC_PAIRS = [
    {
        "scene": "scene_topgun_category_eq",
        "ref":   MUSHRA_AUDIO / "scene_topgun_category_eq" / "v3_3.wav",
        "src":   MUSHRA_AUDIO / "scene_topgun_category_eq" / "v3_5_5.wav",
        "dst":   MUSHRA_AUDIO / "scene_topgun_category_eq" / "v3_5_5_matched.wav",
    },
    {
        "scene": "scene_lalaland_wonder",
        "ref":   MUSHRA_AUDIO / "scene_lalaland_wonder" / "v3_3.wav",
        "src":   MUSHRA_AUDIO / "scene_lalaland_wonder" / "v3_5_5.wav",
        "dst":   MUSHRA_AUDIO / "scene_lalaland_wonder" / "v3_5_5_matched.wav",
    },
]


def main() -> None:
    print("=" * 64)
    print("V3.5.5 PoC Step 4.7 — Loudness matching (ITU-R BS.1770-4)")
    print("=" * 64)

    for pair in POC_PAIRS:
        print(f"\n--- {pair['scene']} ---")
        ref_meas = measure_integrated_lufs(pair["ref"])
        print(f"  ref(v3_3)  LUFS={ref_meas['lufs']:+.2f}  peak={ref_meas['peak_db']:+.2f} dBFS")

        report = match_loudness(
            input_wav=pair["src"],
            target_lufs=ref_meas["lufs"],
            output_wav=pair["dst"],
            peak_ceiling_db=-1.0,
        )
        print(f"  src(v3_5_5)  original_LUFS={report['original_lufs']:+.2f}")
        print(f"  → match gain {report['raw_match_gain_db']:+.2f} dB", end="")
        if report["peak_attenuation_db"] != 0:
            print(f"  + peak attenuation {report['peak_attenuation_db']:+.2f} dB"
                  f" (총 {report['gain_applied_db']:+.2f} dB)")
        else:
            print(f"  (peak ceiling OK)")
        print(f"  ✓ {Path(report['output']).relative_to(REPO)}")
        print(f"    final_LUFS={report['final_lufs']:+.2f}  "
              f"final_peak={report['final_peak_db']:+.2f} dBFS  "
              f"Δ_to_target={report['final_lufs'] - report['target_lufs']:+.2f} LU")


if __name__ == "__main__":
    main()
