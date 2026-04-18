"""generate_calibration_tones.py — JND 캘리브레이션 톤 생성.

simple_player_v3.html 캘리브레이션 섹션에서 재생할 1kHz 사인 톤 3종.
평가자가 이 3개 구분이 되는지로 청각 민감도 + 이어폰/환경 표준화 확인.

사양:
- 1kHz 사인, 2.0s, 48kHz stereo, 50ms fade in/out (pop 방지)
- 기준 레벨 -20 dBFS
  - tone_reference.wav   : -20 dBFS
  - tone_plus_1db.wav    : -19 dBFS (일반 JND 임계)
  - tone_plus_3db.wav    : -17 dBFS (본 시스템 EQ 수준)

출력: evaluation/webmushra/configs/resources/audio/calibration/

실행: PYTHONPATH=. venv/Scripts/python.exe tools/generate_calibration_tones.py
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from model.autoEQ.inference.paths import MUSHRA_CLIPS_DIR


SAMPLE_RATE = 48000
DURATION_SEC = 2.0
TONE_HZ = 1000.0
FADE_MS = 50
REF_DBFS = -20.0

TONES = [
    ("tone_reference.wav",  0.0),   # -20 dBFS (basis)
    ("tone_plus_1db.wav",   1.0),   # -19 dBFS
    ("tone_plus_3db.wav",   3.0),   # -17 dBFS
]


def make_tone(delta_db: float) -> np.ndarray:
    n = int(SAMPLE_RATE * DURATION_SEC)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    level_dbfs = REF_DBFS + delta_db
    amplitude = 10.0 ** (level_dbfs / 20.0)

    mono = amplitude * np.sin(2.0 * np.pi * TONE_HZ * t)

    fade_n = int(SAMPLE_RATE * FADE_MS / 1000)
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float64)
    mono[:fade_n] *= ramp
    mono[-fade_n:] *= ramp[::-1]

    stereo = np.stack([mono, mono], axis=1)
    pcm = np.clip(stereo, -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16)


def main() -> None:
    out_dir = MUSHRA_CLIPS_DIR / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, delta_db in TONES:
        pcm = make_tone(delta_db)
        path = out_dir / filename
        wavfile.write(str(path), SAMPLE_RATE, pcm)
        level_dbfs = REF_DBFS + delta_db
        print(f"  ✓ {filename}  ({level_dbfs:+.0f} dBFS, Δ{delta_db:+.0f} dB)")

    print(f"\n=== 완료 ===\n출력: {out_dir}")


if __name__ == "__main__":
    main()
