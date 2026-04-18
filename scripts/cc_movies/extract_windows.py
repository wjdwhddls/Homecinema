"""영화 → 4s window MP4 + WAV 분할.

Usage:
    python -m scripts.cc_movies.extract_windows \\
        --input_dir dataset/autoEQ/CCMovies/films \\
        --output_dir dataset/autoEQ/CCMovies/windows \\
        --window_sec 4 --stride_sec 4

동작:
  1. input_dir의 모든 *.mp4에 대해 ffprobe로 실제 duration, fps, audio_sr 확인
  2. 4s 비중첩 window (stride=window)로 분할 — 시작 0, 마지막 window는 영상 끝을 넘지 않도록 버림
  3. 각 window에 대해:
     - <output_dir>/<film_id>/<film_id>_<win_idx:05d>.mp4 — 비디오+오디오 포함 4s 클립
     - <output_dir>/<film_id>/<film_id>_<win_idx:05d>.wav — 16kHz mono 오디오 (PANNs 입력 규격)
  4. metadata.csv 생성 (film_id, window_id, t0_sec, t1_sec, video_fps, expected_license)
  5. 전체 요약 summary.json

ffmpeg 필요 (macOS: `brew install ffmpeg`).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from scripts.cc_movies.film_list import ALL_FILMS, FilmEntry


def _ffprobe_video(path: Path) -> dict:
    """Return dict with duration_sec, fps, width, height, has_audio, audio_sr."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    out = subprocess.check_output(cmd)
    meta = json.loads(out)
    video_stream = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)
    duration = float(meta["format"]["duration"])

    fps_str = video_stream.get("r_frame_rate", "25/1") if video_stream else "25/1"
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) > 0 else 25.0

    result = {
        "duration_sec": duration,
        "fps": fps,
        "width": int(video_stream["width"]) if video_stream else 0,
        "height": int(video_stream["height"]) if video_stream else 0,
        "has_audio": audio_stream is not None,
        "audio_sr": int(audio_stream["sample_rate"]) if audio_stream else 0,
        "audio_channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
    }
    return result


def _extract_clip(src: Path, dst_mp4: Path, dst_wav: Path, t0: float, duration: float,
                  audio_sr: int = 16000) -> None:
    """ffmpeg로 [t0, t0+duration] 구간 MP4 + 16kHz mono WAV 추출."""
    # MP4 클립 (비디오 + 오디오). -c copy는 키프레임 정렬 이슈가 있어 재인코딩.
    cmd_mp4 = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t0:.3f}", "-t", f"{duration:.3f}",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst_mp4),
    ]
    subprocess.run(cmd_mp4, check=True)

    # 16kHz mono WAV (PANNs 입력)
    cmd_wav = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t0:.3f}", "-t", f"{duration:.3f}",
        "-i", str(src),
        "-ac", "1", "-ar", str(audio_sr),
        "-c:a", "pcm_s16le",
        str(dst_wav),
    ]
    subprocess.run(cmd_wav, check=True)


def _iter_windows(duration: float, window_sec: int, stride_sec: int) -> Iterable[tuple[int, float, float]]:
    """영상 끝을 넘지 않는 범위에서 (idx, t0, t1) 시퀀스 생성."""
    idx = 0
    t0 = 0.0
    while t0 + window_sec <= duration + 1e-6:
        yield idx, t0, t0 + window_sec
        idx += 1
        t0 += stride_sec


def _film_id_for_mp4(mp4_path: Path) -> str:
    return mp4_path.stem


def process_film(film_id: str, mp4_path: Path, output_dir: Path, window_sec: int,
                 stride_sec: int, audio_sr: int) -> dict:
    """한 영화를 처리해 window 생성 + 메타 반환."""
    film_out = output_dir / film_id
    film_out.mkdir(parents=True, exist_ok=True)

    meta = _ffprobe_video(mp4_path)
    duration = meta["duration_sec"]

    rows: list[dict] = []
    t0_start = time.time()
    for idx, t0, t1 in _iter_windows(duration, window_sec, stride_sec):
        win_id = f"{film_id}_{idx:05d}"
        dst_mp4 = film_out / f"{win_id}.mp4"
        dst_wav = film_out / f"{win_id}.wav"
        if dst_mp4.is_file() and dst_wav.is_file():
            rows.append({"film_id": film_id, "window_id": win_id, "t0": t0, "t1": t1})
            continue
        _extract_clip(mp4_path, dst_mp4, dst_wav, t0, window_sec, audio_sr=audio_sr)
        rows.append({"film_id": film_id, "window_id": win_id, "t0": t0, "t1": t1})

    elapsed = time.time() - t0_start
    return {
        "film_id": film_id,
        "src_mp4": str(mp4_path),
        "duration_sec": duration,
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "has_audio": meta["has_audio"],
        "audio_sr": meta["audio_sr"],
        "window_count": len(rows),
        "windows": rows,
        "elapsed_sec": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="영화 → window 분할")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_sec", type=int, default=4)
    parser.add_argument("--stride_sec", type=int, default=4)
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--film_ids", type=str, default="",
                        help="쉼표 구분. 빈 값이면 input_dir의 모든 mp4 처리.")
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(args.input_dir.glob("*.mp4"))
    if args.film_ids:
        wanted = {x.strip() for x in args.film_ids.split(",") if x.strip()}
        mp4s = [p for p in mp4s if _film_id_for_mp4(p) in wanted]
    if not mp4s:
        print(f"[err] {args.input_dir}에서 *.mp4를 찾을 수 없음", file=sys.stderr)
        return 1

    print(f"[info] {len(mp4s)}편 처리. window={args.window_sec}s stride={args.stride_sec}s")

    license_map = {e.film_id: e.license for e in ALL_FILMS}

    all_rows: list[dict] = []
    summary: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "audio_sr": args.audio_sr,
        "films": [],
    }

    for mp4 in mp4s:
        film_id = _film_id_for_mp4(mp4)
        print(f"  [process] {film_id} ({mp4.stat().st_size/1e6:.1f} MB)")
        try:
            result = process_film(
                film_id, mp4, output_dir,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                audio_sr=args.audio_sr,
            )
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: ffmpeg 실패 — {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            continue
        result["license"] = license_map.get(film_id, "UNKNOWN")
        summary["films"].append({k: v for k, v in result.items() if k != "windows"})
        for row in result["windows"]:
            row["license"] = result["license"]
            row["video_fps"] = result["fps"]
            all_rows.append(row)
        print(f"    OK {result['window_count']} windows in {result['elapsed_sec']:.1f}s  audio_sr={result['audio_sr']} has_audio={result['has_audio']}")

    # metadata.csv
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["film_id", "window_id", "t0", "t1", "video_fps", "license"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    # summary.json
    summary["total_windows"] = len(all_rows)
    summary["total_films"] = len(summary["films"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[info] 총 {summary['total_films']}편, {summary['total_windows']} windows")
    print(f"[info] metadata: {csv_path}")
    print(f"[info] summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
