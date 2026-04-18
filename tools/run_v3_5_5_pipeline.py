"""run_v3_5_5_pipeline.py — V3.5.5 PoC: Demucs 분리 + 시변 EQ + Compressor.

트랙 3 PoC. v3_3 대비 변수 1개(스템 분리)만 격리해서 효과 측정.
Step 4.6 재구현(B-1 채택): v3_3과 동일한 시변 EQ 처리 방식으로 격리 변수 단순화.

파이프라인:
  1) 입력으로 Demucs 사전 분리 결과(vocals.wav + no_vocals.wav)를 받음
     (Step 2/3에서 tmp/demucs_test/htdemucs/<scene>/ 에 생성된 11s 클립)
  2) timeline.json에서 클립 구간 [clip_start_sec, clip_end_sec] 내 씬 추출
  3) no_vocals 스템에 v3_3과 동일한 시변 EQ 적용:
       - compute_effective_eq (mood blending + dialogue protection α_d=0.5 +
         confidence scaling + V3.3 ±6dB presets)
       - apply_timevarying_eq_array로 클립 좌표계에 맞춰 시변 적용
  4) vocals 스템에 명료도 EQ 적용 (씬 무관 고정, 비시변)
  5) 합산 (vocals_eq + no_vocals_eq)
  6) Compressor + Makeup gain (V3.3와 동일 사양)
  7) Peak > 0.99 시 0.95로 정규화 (V3.3와 동일 정책)

처리 일관성 (v3_3 대비 격리 변수):
  - EQ granularity: 시변 (동일)
  - Mood blending: 적용 (동일)
  - Dialogue protection: no_vocals에 적용 (B-1 동일)
  - Compressor: 동일 사양
  - Peak 정규화: 동일 정책
  → 유일한 차이 = 스템 분리 + vocals 명료도 EQ 추가

명료도 EQ (vocals 스템 전용, 씬 무관 고정):
  표준 voice intelligibility 부스트 — 보수적 강도(±2.5dB 이내).
    - sub/rumble (B0~B1):     -2.0 / -1.5
    - mud (B3, 250Hz):        -1.0
    - presence (B5~B6):       +0.5 / +1.5
    - clarity (B7, 4kHz):     +1.5
    - air (B8, 8kHz):         +1.0

Compressor (합산 후, V3.3와 동일):
  threshold -12 dB, ratio 3:1, attack 10 ms, release 100 ms, makeup +3 dB

실행:
  PYTHONIOENCODING=utf-8 PYTHONPATH=. venv/Scripts/python.exe \\
      tools/run_v3_5_5_pipeline.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pedalboard import Compressor, Gain, Pedalboard
from scipy.io import wavfile

from model.autoEQ.inference.eq_engine import EQ_PRESETS_V3_3
from model.autoEQ.inference.playback import (
    apply_timevarying_eq_array, build_eq_chain,
)


# ────────────────────────────────────────────────────────
# 사양 (V3.3와 동일)
# ────────────────────────────────────────────────────────
COMPRESSOR_THRESHOLD_DB = -12.0
COMPRESSOR_RATIO        = 3.0
COMPRESSOR_ATTACK_MS    = 10.0
COMPRESSOR_RELEASE_MS   = 100.0
MAKEUP_GAIN_DB          = 3.0

# Vocals 스템 명료도 EQ (씬 무관 고정, 시변 X, ±2.5dB 이내 보수적)
VOCALS_CLARITY_EQ = np.array([
    -2.0,   # B0  31.5 Hz   sub/rumble
    -1.5,   # B1  63   Hz   low rumble
     0.0,   # B2  125  Hz
    -1.0,   # B3  250  Hz   mud
     0.0,   # B4  500  Hz
    +0.5,   # B5  1000 Hz   upper-mid
    +1.5,   # B6  2000 Hz   presence
    +1.5,   # B7  4000 Hz   clarity / consonant
    +1.0,   # B8  8000 Hz   air
     0.0,   # B9  16000 Hz
])


# ────────────────────────────────────────────────────────
# PoC 작업 대상 — Step 2/3 산출물 + run_v3_3_pipeline.py CURATED_SCENES와 동일 구간
# ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PoCJob:
    scene_dir_name: str         # evaluation/.../audio/<scene_dir_name>/
    job_id: str                 # backend/data/jobs/<job_id>/
    clip_start_sec: float       # 풀 트레일러 내 클립 시작 시각
    clip_end_sec: float         # 풀 트레일러 내 클립 종료 시각
    stems_dir: Path             # tmp/demucs_test/htdemucs/<scene>/

REPO         = Path(__file__).resolve().parent.parent
MUSHRA_AUDIO = REPO / "evaluation" / "webmushra" / "configs" / "resources" / "audio"
DEMUCS_OUT   = REPO / "tmp" / "demucs_test" / "htdemucs"
JOBS_ROOT    = REPO / "backend" / "data" / "jobs"

POC_JOBS = [
    PoCJob(
        scene_dir_name="scene_topgun_category_eq",
        job_id="fe2ecad8-dc25-4131-adfe-ffeea6d977a1",
        clip_start_sec=69.0,   # run_v3_3_pipeline.py CURATED_SCENES[topgun] category_eq
        clip_end_sec=80.0,
        stems_dir=DEMUCS_OUT / "topgun_cat_eq",
    ),
    PoCJob(
        scene_dir_name="scene_lalaland_wonder",
        job_id="lalaland-demo",
        clip_start_sec=100.0,  # run_v3_3_pipeline.py CURATED_SCENES[lalaland] wonder
        clip_end_sec=111.0,
        stems_dir=DEMUCS_OUT / "lalaland_wonder",
    ),
]


# ────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────
def db_from_linear(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-10 else -float("inf")


def load_wav_float32(path: Path) -> tuple[int, np.ndarray]:
    """wav → (sr, [channels, samples] float32)."""
    sr, a = wavfile.read(str(path))
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    elif a.dtype == np.int32:
        a = a.astype(np.float32) / 2147483648.0
    elif a.dtype == np.float32:
        pass
    else:
        a = a.astype(np.float32)
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    return sr, a.T.copy()  # [ch, samples]


def save_wav_int16(arr_ch_samp: np.ndarray, sr: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = arr_ch_samp.T  # [samples, ch]
    a = np.clip(a, -1.0, 1.0)
    wavfile.write(str(out_path), sr, (a * 32767.0).astype(np.int16))


def measure(arr_ch_samp: np.ndarray) -> tuple[float, float]:
    peak = float(np.abs(arr_ch_samp).max())
    rms  = float(np.sqrt(np.mean(arr_ch_samp.astype(np.float64) ** 2)))
    return db_from_linear(peak), db_from_linear(rms)


def count_scenes_in_clip(scenes: list[dict], start_sec: float, end_sec: float) -> int:
    return sum(
        1 for s in scenes
        if s["end_sec"] > start_sec and s["start_sec"] < end_sec
    )


# ────────────────────────────────────────────────────────
# 핵심 파이프라인
# ────────────────────────────────────────────────────────
def apply_v3_5_5(
    original_wav_path: Path,
    vocals_wav_path: Path,
    no_vocals_wav_path: Path,
    timeline_path: Path,
    clip_start_sec: float,
    clip_end_sec: float,
    output_wav_path: Path,
    eq_intensity: float = 1.0,
    dialogue_protection_no_vocals: bool = True,  # B-1 default
) -> dict:
    """V3.5.5 시변 파이프라인 (B-1: no_vocals에 dialogue protection 적용).

    no_vocals 스템에 v3_3과 동일한 시변 EQ를 적용. vocals 스템은 고정 명료도 EQ.
    """
    # 1) timeline 로드
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    scenes = timeline["scenes"]
    scenes_in_clip = count_scenes_in_clip(scenes, clip_start_sec, clip_end_sec)

    # 2) 스템 로드
    sr_orig, orig = load_wav_float32(original_wav_path)
    sr_v,    voc  = load_wav_float32(vocals_wav_path)
    sr_n,    nov  = load_wav_float32(no_vocals_wav_path)
    if sr_orig != sr_v or sr_orig != sr_n:
        raise ValueError(f"SR mismatch: orig={sr_orig}, voc={sr_v}, nov={sr_n}")
    sr = sr_orig

    L = min(orig.shape[1], voc.shape[1], nov.shape[1])
    drift = max(orig.shape[1], voc.shape[1], nov.shape[1]) - L
    orig = orig[:, :L]
    voc  = voc [:, :L]
    nov  = nov [:, :L]

    # 3) no_vocals 스템 — v3_3과 동일한 시변 EQ (B-1: dialogue_protection=True)
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
        prevent_clipping=False,  # 합산·압축 후에 한꺼번에 관리
    )

    # 4) vocals 스템 — 고정 명료도 EQ (시변 X)
    vocals_gains = VOCALS_CLARITY_EQ * eq_intensity
    voc_chain = build_eq_chain(vocals_gains)
    voc_eq = voc_chain(voc.astype(np.float32, copy=False), sr)

    # 5) 합산
    summed = voc_eq + nov_eq
    pre_peak_db, pre_rms_db = measure(summed)

    # 6) Compressor + Makeup
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

    # 7) 클리핑 방지 (v3_3과 동일 정책)
    final_peak_lin = float(np.abs(out).max())
    clipping_warning = False
    if final_peak_lin > 0.99:
        clipping_warning = True
        scale = 0.95 / final_peak_lin
        out = out * scale
        print(f"    클리핑 방지: peak {final_peak_lin:.3f} → 0.95 (scale {scale:.4f})")

    final_peak_db, final_rms_db = measure(out)

    save_wav_int16(out, sr, output_wav_path)

    orig_peak_db, orig_rms_db = measure(orig)
    voc_peak_db, voc_rms_db   = measure(voc)
    nov_peak_db, nov_rms_db   = measure(nov)

    return {
        "sr": sr,
        "samples": L,
        "duration_sec": L / sr,
        "drift_samples": drift,
        "scenes_in_clip": scenes_in_clip,
        "eq_intensity": eq_intensity,
        "dialogue_protection_no_vocals": dialogue_protection_no_vocals,
        "vocals_gains_db": vocals_gains.tolist(),
        "levels": {
            "original":       {"peak_db": orig_peak_db, "rms_db": orig_rms_db},
            "vocals_in":      {"peak_db": voc_peak_db,  "rms_db": voc_rms_db},
            "no_vocals_in":   {"peak_db": nov_peak_db,  "rms_db": nov_rms_db},
            "pre_compressor": {"peak_db": pre_peak_db,  "rms_db": pre_rms_db},
            "final_output":   {"peak_db": final_peak_db, "rms_db": final_rms_db},
        },
        "clipping_warning": clipping_warning,
    }


# ────────────────────────────────────────────────────────
# 러너
# ────────────────────────────────────────────────────────
def run_one(job: PoCJob) -> dict:
    print(f"\n=== {job.scene_dir_name} (clip {job.clip_start_sec}–{job.clip_end_sec}s) ===")

    scene_dir   = MUSHRA_AUDIO / job.scene_dir_name
    original    = scene_dir / "original.wav"
    out_v3_5_5  = scene_dir / "v3_5_5.wav"
    vocals      = job.stems_dir / "vocals.wav"
    no_vocals   = job.stems_dir / "no_vocals.wav"
    timeline    = JOBS_ROOT / job.job_id / "timeline.json"

    for p in (original, vocals, no_vocals, timeline):
        if not p.exists():
            raise FileNotFoundError(f"{job.scene_dir_name}: 입력 누락 {p}")

    print(f"  timeline   : {timeline.relative_to(REPO)}")
    print(f"  vocals     : {vocals.relative_to(REPO)}")
    print(f"  no_vocals  : {no_vocals.relative_to(REPO)}")
    print(f"  → output   : {out_v3_5_5.relative_to(REPO)}")

    report = apply_v3_5_5(
        original_wav_path=original,
        vocals_wav_path=vocals,
        no_vocals_wav_path=no_vocals,
        timeline_path=timeline,
        clip_start_sec=job.clip_start_sec,
        clip_end_sec=job.clip_end_sec,
        output_wav_path=out_v3_5_5,
        eq_intensity=1.0,
        dialogue_protection_no_vocals=True,
    )

    # v3_3 비교
    v33 = scene_dir / "v3_3.wav"
    v33_levels = None
    if v33.exists():
        _, arr33 = load_wav_float32(v33)
        v33_pk, v33_rms = measure(arr33)
        v33_levels = {"peak_db": v33_pk, "rms_db": v33_rms,
                      "samples": arr33.shape[1]}
    report["v3_3_levels"] = v33_levels

    L = report["levels"]
    print(f"  ✓ 저장 완료 dur={report['duration_sec']:.3f}s  drift={report['drift_samples']}smp  "
          f"scenes_in_clip={report['scenes_in_clip']}")
    print(f"    original       peak={L['original']['peak_db']:+.2f}  rms={L['original']['rms_db']:+.2f}")
    print(f"    vocals_in      peak={L['vocals_in']['peak_db']:+.2f}  rms={L['vocals_in']['rms_db']:+.2f}")
    print(f"    no_vocals_in   peak={L['no_vocals_in']['peak_db']:+.2f}  rms={L['no_vocals_in']['rms_db']:+.2f}")
    print(f"    pre-compressor peak={L['pre_compressor']['peak_db']:+.2f}  rms={L['pre_compressor']['rms_db']:+.2f}")
    print(f"    final v3_5_5   peak={L['final_output']['peak_db']:+.2f}  rms={L['final_output']['rms_db']:+.2f}")
    if v33_levels:
        d_peak = L['final_output']['peak_db'] - v33_levels['peak_db']
        d_rms  = L['final_output']['rms_db']  - v33_levels['rms_db']
        print(f"    v3_3 reference peak={v33_levels['peak_db']:+.2f}  rms={v33_levels['rms_db']:+.2f}  "
              f"(Δpeak={d_peak:+.2f}dB, Δrms={d_rms:+.2f}dB, samples_match={v33_levels['samples']==report['samples']})")
    if report["clipping_warning"]:
        print(f"    ⚠ CLIPPING WARNING — peak normalized to 0.95")

    return {"job": job.scene_dir_name, **report}


def main() -> None:
    print("=" * 64)
    print("V3.5.5 PoC 파이프라인 (Demucs + 시변 EQ + Compressor) — Step 4.6 B-1")
    print("=" * 64)
    reports = [run_one(j) for j in POC_JOBS]

    print("\n" + "=" * 64)
    print("요약")
    print("=" * 64)
    for r in reports:
        L = r["levels"]["final_output"]
        v33 = r.get("v3_3_levels") or {}
        v33_peak = v33.get("peak_db")
        v33_str  = (
            f"  v3_3:peak={v33_peak:+.2f} Δ={L['peak_db']-v33_peak:+.2f}dB"
            if v33_peak is not None else ""
        )
        clip = " [CLIP]" if r["clipping_warning"] else ""
        print(f"  [{r['job']:32s}] scenes={r['scenes_in_clip']}  "
              f"v3_5_5:peak={L['peak_db']:+.2f} rms={L['rms_db']:+.2f}{v33_str}{clip}")


if __name__ == "__main__":
    main()
