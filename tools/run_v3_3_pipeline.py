"""run_v3_3_pipeline.py — V3.3 EQ (±6dB 확대) + Compressor 후처리 파이프라인.

트랙 2 (V3.2 subtlety 근본 해결). 방식 C(하이브리드):
- EQ_PRESETS_V3_3는 eq_engine.py에 추가됐지만 PRESET_VERSIONS에는 등록 안 됨
- 이 wrapper가 기존 job의 timeline.json을 재사용해 V3.3만 새로 생성
- analyzer.py 무수정, 기존 processed_v3_1/v3_2.mp4 영향 없음
- Compressor는 EQ 적용 후 별도 단계 (playback.py 무수정)

출력 두 곳:
  (a) backend/data/jobs/<id>/processed_v3_3.mp4    — 풀 트레일러
  (b) evaluation/webmushra/configs/resources/audio/scene_<video>_<name>/v3_3.wav
      — 큐레이션 6 씬 (길이/포맷을 기존 v3_2.wav와 정확히 일치)

Compressor 사양 (영화 mixing 중간치):
  threshold -12 dB, ratio 3:1, attack 10 ms, release 100 ms
  Makeup gain +3 dB (압축 후 레벨 보상)

실행: PYTHONPATH=. venv/Scripts/python.exe tools/run_v3_3_pipeline.py
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from pedalboard import Compressor, Gain, Pedalboard
from pedalboard.io import AudioFile

from model.autoEQ.inference.eq_engine import EQ_PRESETS_V3_3, compute_effective_eq
from model.autoEQ.inference.mux import mux_video_audio
from model.autoEQ.inference.paths import JOBS_DATA_DIR, MUSHRA_CLIPS_DIR
from model.autoEQ.inference.playback import apply_timevarying_eq
from model.autoEQ.inference.utils import get_audio_info


# ────────────────────────────────────────────────────────
# 작업 대상 job (run_mushra_curated.py와 동일)
# ────────────────────────────────────────────────────────
JOBS = {
    "topgun":   "fe2ecad8-dc25-4131-adfe-ffeea6d977a1",
    "lalaland": "lalaland-demo",
}

# 큐레이션 씬 (run_mushra_curated.py의 CURATED_SCENES와 동일)
CURATED_SCENES: dict[str, list[tuple[str, float, float]]] = {
    "topgun": [
        ("baseline",          8.0, 16.0),
        ("dialogue_protect", 35.0, 47.0),
        ("category_eq",      69.0, 80.0),
    ],
    "lalaland": [
        ("baseline",       0.0,   9.0),
        ("song_dialogue", 86.0,  95.0),
        ("wonder",       100.0, 111.0),
    ],
}

# Compressor + Makeup 사양
COMPRESSOR_THRESHOLD_DB = -12.0
COMPRESSOR_RATIO        = 3.0
COMPRESSOR_ATTACK_MS    = 10.0
COMPRESSOR_RELEASE_MS   = 100.0
MAKEUP_GAIN_DB          = 3.0


def extract_audio_from_video(video_path: Path, wav_path: Path) -> None:
    """analyzer.py와 동일 방식으로 오디오 추출 (기본 SR 유지)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", str(wav_path)],
        check=True, capture_output=True,
    )


def apply_compressor_stage(input_wav: Path, output_wav: Path) -> tuple[float, float]:
    """Compressor + Makeup gain 후처리. playback.py 무수정.

    Returns: (peak, rms) of output in linear [0, 1]
    """
    with AudioFile(str(input_wav)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    board = Pedalboard([
        Compressor(
            threshold_db=COMPRESSOR_THRESHOLD_DB,
            ratio=COMPRESSOR_RATIO,
            attack_ms=COMPRESSOR_ATTACK_MS,
            release_ms=COMPRESSOR_RELEASE_MS,
        ),
        Gain(gain_db=MAKEUP_GAIN_DB),
    ])
    out = board(audio, sr)

    peak = float(np.abs(out).max())
    if peak > 0.99:
        scale = 0.95 / peak
        out = out * scale
        print(f"    클리핑 방지: peak {peak:.3f} → 0.95")
        peak = float(np.abs(out).max())

    rms = float(np.sqrt(np.mean(out ** 2)))

    with AudioFile(str(output_wav), "w", sr, audio.shape[0]) as f:
        f.write(out)

    return peak, rms


def build_scenes_eq_v3_3(scenes: list[dict]) -> list[dict]:
    """timeline.json의 scenes → V3.3 effective_gains 결합."""
    out = []
    for s in scenes:
        probs = s["aggregated"]["mood_probs_mean"]
        density = s["dialogue"]["density"]
        gains = compute_effective_eq(
            probs, density,
            alpha_d=0.5, intensity=1.0,
            confidence_scaling=True,
            presets=EQ_PRESETS_V3_3,
        )
        out.append({
            "start_sec":      s["start_sec"],
            "end_sec":        s["end_sec"],
            "transition_out": s.get("transition_out", "cut"),
            "effective_gains": gains,
        })
    return out


def extract_curated_clip(
    source_mp4: Path, out_wav: Path,
    start_sec: float, end_sec: float,
    target_duration: float | None = None,
) -> float:
    """processed_v3_3.mp4에서 [start, end] 오디오 구간 발췌 → 48kHz stereo s16le wav.

    run_mushra_curated.py의 extract_audio_segment와 동일 포맷.
    target_duration 지정 시 해당 길이로 정확히 trim (다른 v3_X.wav 길이 일치용).
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec
    if target_duration is not None:
        duration = min(duration, target_duration)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(source_mp4),
            "-t", f"{duration:.3f}",
            "-vn",
            "-ar", "48000",
            "-ac", "2",
            "-acodec", "pcm_s16le",
            str(out_wav),
        ],
        check=True, capture_output=True,
    )
    return float(get_audio_info(out_wav)["duration"])


def db_from_linear(x: float) -> float:
    if x <= 1e-10:
        return -float("inf")
    return 20.0 * math.log10(x)


def process_one_job(video_key: str, job_id: str) -> dict:
    job_dir = JOBS_DATA_DIR / job_id
    timeline_path = job_dir / "timeline.json"
    original_mp4  = job_dir / "original.mp4"
    processed_v33 = job_dir / "processed_v3_3.mp4"

    for p in (timeline_path, original_mp4):
        if not p.exists():
            raise FileNotFoundError(f"{video_key}: {p} 없음")

    print(f"\n=== {video_key} (job {job_id}) ===")

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    scenes = timeline["scenes"]
    print(f"  씬 수: {len(scenes)}")

    scenes_eq = build_scenes_eq_v3_3(scenes)

    # 1) 오디오 추출 → 2) 시간축 EQ → 3) Compressor → 4) mux
    audio_orig = job_dir / "audio_original_v3_3_tmp.wav"
    audio_eq   = job_dir / "audio_eq_v3_3_tmp.wav"
    audio_comp = job_dir / "audio_comp_v3_3_tmp.wav"

    try:
        print(f"  [1/4] 오디오 추출")
        extract_audio_from_video(original_mp4, audio_orig)

        print(f"  [2/4] 시간축 EQ (V3.3)")
        apply_timevarying_eq(str(audio_orig), str(audio_eq), scenes_eq)

        print(f"  [3/4] Compressor (-{abs(COMPRESSOR_THRESHOLD_DB):.0f}dB, {COMPRESSOR_RATIO:.0f}:1, atk {COMPRESSOR_ATTACK_MS:.0f}ms, rel {COMPRESSOR_RELEASE_MS:.0f}ms) + makeup {MAKEUP_GAIN_DB:+.0f}dB")
        peak, rms = apply_compressor_stage(audio_eq, audio_comp)

        print(f"  [4/4] Mux → {processed_v33.name}")
        mux_video_audio(str(original_mp4), str(audio_comp), str(processed_v33))

    finally:
        for f in (audio_orig, audio_eq, audio_comp):
            f.unlink(missing_ok=True)

    # 검증
    info = get_audio_info(processed_v33)
    peak_db = db_from_linear(peak)
    rms_db  = db_from_linear(rms)
    crest   = peak_db - rms_db
    print(f"  ✓ processed_v3_3.mp4 생성 (dur={info['duration']:.2f}s, "
          f"peak={peak_db:+.2f} dBFS, rms={rms_db:+.2f} dBFS, crest={crest:.2f} dB)")

    # 큐레이션 씬 wav 발췌
    clip_reports = []
    for scene_name, start, end in CURATED_SCENES[video_key]:
        scene_dir_name = f"scene_{video_key}_{scene_name}"
        scene_dir = MUSHRA_CLIPS_DIR / scene_dir_name
        if not scene_dir.exists():
            print(f"    ⚠ {scene_dir_name} 디렉토리 없음 — run_mushra_curated.py 먼저 실행 필요. 스킵.")
            continue

        # 기존 v3_2.wav 길이에 맞춰 정확히 정렬 (단, 기존 wav가 있을 때만)
        ref_wav = scene_dir / "v3_2.wav"
        target_dur = None
        if ref_wav.exists():
            target_dur = float(get_audio_info(ref_wav)["duration"])

        out_wav = scene_dir / "v3_3.wav"
        dur = extract_curated_clip(processed_v33, out_wav, start, end, target_dur)
        note = f"target {target_dur:.3f}s" if target_dur else "no ref"
        print(f"    ✓ {scene_dir_name}/v3_3.wav  dur={dur:.3f}s  ({note})")
        clip_reports.append({
            "scene": scene_dir_name,
            "duration": round(dur, 3),
            "target": round(target_dur, 3) if target_dur else None,
        })

    return {
        "video_key": video_key,
        "job_id": job_id,
        "processed_mp4": str(processed_v33),
        "peak_dbfs": round(peak_db, 2),
        "rms_dbfs":  round(rms_db, 2),
        "crest_db":  round(crest, 2),
        "curated_clips": clip_reports,
    }


def main() -> None:
    print("=" * 60)
    print("V3.3 파이프라인 (EQ ±6dB + Compressor)")
    print("=" * 60)

    reports = []
    for video_key, job_id in JOBS.items():
        reports.append(process_one_job(video_key, job_id))

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    for r in reports:
        print(f"  [{r['video_key']}] peak {r['peak_dbfs']:+.2f} dBFS, "
              f"rms {r['rms_dbfs']:+.2f}, crest {r['crest_db']:.2f} dB "
              f"— 큐레이션 {len(r['curated_clips'])}/3 씬")


if __name__ == "__main__":
    main()
