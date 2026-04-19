"""run_v3_5_5_full_trailer.py — V3.5.5 풀 트레일러 처리 드라이버 (Phase X1).

트랙 5+ 풀 트레일러 평가 단계. 기존 11s 씬 단위 `run_v3_5_5_pipeline.py`의
`apply_v3_5_5()` 함수를 재사용하여 풀 트레일러 전체 길이에 적용.

처리 흐름:
  1) backend/data/jobs/<job>/original.mp4 → original.wav (ffmpeg, 48k stereo)
  2) backend/data/jobs/<job>/processed_v3_3.mp4 → v3_3.wav (V3.3 reference)
  3) htdemucs 2-stem 분리 (full duration)
     tmp/demucs_test/run_separate.py 재사용
  4) apply_v3_5_5(clip_start=0, clip_end=full_duration) 호출
     → 시변 EQ (42 scenes), dialogue protection, per-stem EQ, compressor

배경: Phase X2 transition crossfade는 apply_timevarying_eq_array → get_eq_at_time
→ sigmoid_crossfade 체인으로 이미 구현됨. 풀 트레일러 처리 결과에 cut 0.3s /
dissolve 2.0s 차등 크로스페이드가 timeline.json의 transition_out 필드 기반으로
자동 적용됨.

실행:
  PYTHONIOENCODING=utf-8 PYTHONPATH=. venv/Scripts/python.exe \\
      tools/run_v3_5_5_full_trailer.py --job topgun
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile


REPO = Path(__file__).resolve().parent.parent

JOB_PATHS = {
    "topgun":   REPO / "backend" / "data" / "jobs" / "fe2ecad8-dc25-4131-adfe-ffeea6d977a1",
    "lalaland": REPO / "backend" / "data" / "jobs" / "lalaland-demo",
}
OUT_BASE = REPO / "tmp" / "full_trailer"


def _run_ffmpeg(input_mp4: Path, output_wav: Path) -> None:
    """48kHz stereo s16le wav 추출."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_mp4),
         "-vn", "-ar", "48000", "-ac", "2", "-acodec", "pcm_s16le",
         str(output_wav)],
        check=True, capture_output=True,
    )


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-10 else -float("inf")


def _measure(path: Path) -> dict:
    sr, a = wavfile.read(str(path))
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    pk = float(np.abs(a).max())
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    return {"sr": sr, "samples": a.shape[0], "peak_db": _db(pk), "rms_db": _db(rms)}


def process_full_trailer(job_key: str, *, skip_separation: bool = False) -> dict:
    """풀 트레일러 V3.5.5 처리. 재실행 시 skip_separation=True로 htdemucs 생략."""
    if job_key not in JOB_PATHS:
        raise ValueError(f"unknown job: {job_key} (options: {list(JOB_PATHS)})")

    job_dir = JOB_PATHS[job_key]
    ft_dir  = OUT_BASE / job_key

    original_mp4    = job_dir / "original.mp4"
    v3_3_mp4        = job_dir / "processed_v3_3.mp4"
    timeline_path   = job_dir / "timeline.json"
    for p in (original_mp4, timeline_path):
        if not p.exists():
            raise FileNotFoundError(f"{p}")

    original_wav  = ft_dir / "original.wav"
    v3_3_wav      = ft_dir / "v3_3.wav"
    stems_dir     = ft_dir / "stems"
    vocals_wav    = stems_dir / "vocals.wav"
    no_vocals_wav = stems_dir / "no_vocals.wav"
    v3_5_5_wav    = ft_dir / "v3_5_5.wav"

    report: dict = {"job": job_key, "out_dir": str(ft_dir), "steps": {}}

    print(f"=== {job_key} full-trailer V3.5.5 ===")

    # 1) ffmpeg 추출
    print("  [1/4] ffmpeg extract (original + V3.3)")
    t = time.perf_counter()
    if not original_wav.exists():
        _run_ffmpeg(original_mp4, original_wav)
    if v3_3_mp4.exists() and not v3_3_wav.exists():
        _run_ffmpeg(v3_3_mp4, v3_3_wav)
    report["steps"]["extract_sec"] = round(time.perf_counter() - t, 2)

    # 2) htdemucs 분리 (stems)
    if vocals_wav.exists() and no_vocals_wav.exists() and skip_separation:
        print(f"  [2/4] stems 존재 — 분리 스킵")
        report["steps"]["separate_sec"] = 0.0
    else:
        print(f"  [2/4] htdemucs separation")
        t = time.perf_counter()
        sep_script = REPO / "tmp" / "demucs_test" / "run_separate.py"
        subprocess.run(
            [str(REPO / "venv" / "Scripts" / "python.exe"),
             str(sep_script), str(original_wav), str(stems_dir)],
            check=True,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
                 "PYTHONPATH": str(REPO)},
        )
        report["steps"]["separate_sec"] = round(time.perf_counter() - t, 2)

    # 3) apply_v3_5_5 (시변 EQ with crossfade built-in)
    print("  [3/4] apply_v3_5_5 (시변 EQ + per-stem + compressor)")
    sys.path.insert(0, str(REPO))
    from tools.run_v3_5_5_pipeline import apply_v3_5_5
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    clip_end = timeline["scenes"][-1]["end_sec"]
    scene_count = len(timeline["scenes"])
    t = time.perf_counter()
    v3_5_5_report = apply_v3_5_5(
        original_wav_path=original_wav,
        vocals_wav_path=vocals_wav,
        no_vocals_wav_path=no_vocals_wav,
        timeline_path=timeline_path,
        clip_start_sec=0.0,
        clip_end_sec=clip_end,
        output_wav_path=v3_5_5_wav,
        eq_intensity=1.0,
        dialogue_protection_no_vocals=True,
    )
    report["steps"]["v3_5_5_sec"] = round(time.perf_counter() - t, 2)
    report["scene_count"]         = scene_count
    report["clip_end_sec"]        = clip_end
    report["v3_5_5"]              = v3_5_5_report

    # 4) 측정
    print("  [4/4] measure all outputs")
    report["levels"] = {
        "original.wav": _measure(original_wav),
        "v3_3.wav":     _measure(v3_3_wav) if v3_3_wav.exists() else None,
        "v3_5_5.wav":   _measure(v3_5_5_wav),
    }

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="topgun", choices=list(JOB_PATHS.keys()))
    ap.add_argument("--skip-separation", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    report = process_full_trailer(args.job, skip_separation=args.skip_separation)
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print(f"Summary  {args.job}  elapsed={elapsed:.1f}s")
    print("=" * 60)
    print(f"  scenes_in_clip : {report['scene_count']}  dur={report['clip_end_sec']:.2f}s")
    for step, sec in report["steps"].items():
        print(f"  {step:16s}: {sec:6.2f}s")
    print()
    print(f"  {'file':16s} {'samples':>8s}  {'peak':>8s}  {'rms':>8s}")
    for name, L in report["levels"].items():
        if L is None:
            continue
        print(f"  {name:16s} {L['samples']:>8d}  {L['peak_db']:+7.2f}  {L['rms_db']:+7.2f}")
    if report["v3_5_5"]["clipping_warning"]:
        print(f"  ⚠ clipping normalized")


if __name__ == "__main__":
    main()
