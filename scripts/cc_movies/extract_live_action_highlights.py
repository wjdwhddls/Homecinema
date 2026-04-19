"""Feature film 3편에서 감정 풍부한 highlight 추출.

각 영화에서 **runtime 20% / 50% / 80% 지점에서 각 4분씩** 추출 → 총 12분/영화.
20%: 도입·갈등 시작, 50%: climax, 80%: resolution/긴장 극점.
이렇게 distributed sampling하면 특정 구간 편향 없이 다양한 감정 스펙트럼 확보.

입력 영화:
  - his_girl_friday.mp4  (공공영역 1940 코미디, 92분)
  - doa.mp4              (공공영역 1950 노이어, 83분)
  - valkaama.mp4         (CC-BY-SA 2010 드라마, 93분)

출력: films/ 에 *_highlight.mp4 로 저장 (extract_windows.py 입력용).

Usage:
  python scripts/cc_movies/extract_live_action_highlights.py \\
    --films_dir dataset/autoEQ/CCMovies/films \\
    [--segment_sec 240 --n_segments 3]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


FILMS = [
    ("his_girl_friday", "his_girl_friday.mp4", "public_domain_1940"),
    ("doa",             "doa.mp4",             "public_domain_1950"),
    ("valkaama",        "valkaama.mp4",        "CC-BY-SA-3.0"),
]


def ffprobe_duration(mp4_path: Path) -> float:
    """영상 총 길이 (초) 반환."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp4_path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def extract_segment(src: Path, dst: Path, start_sec: float, length_sec: float) -> None:
    """ffmpeg로 특정 구간 복사 (re-encode, audio 포함)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.2f}",
        "-i", str(src),
        "-t", f"{length_sec:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def concat_segments(segment_paths: list[Path], output_path: Path) -> None:
    """여러 세그먼트를 하나로 합침 (ffmpeg concat demuxer)."""
    list_file = output_path.parent / f".concat_{output_path.stem}.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in segment_paths))
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-loglevel", "error",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink()


def run(films_dir: Path, segment_sec: float, n_segments: int) -> dict:
    report = {"films": []}
    for film_id, filename, license_tag in FILMS:
        src = films_dir / filename
        if not src.is_file():
            print(f"[warn] {src} 없음 — 스킵")
            continue

        duration = ffprobe_duration(src)
        print(f"[info] {film_id}: {duration:.1f}s ({duration/60:.1f} min)")

        # runtime 20%/50%/80% 기준 start time (n_segments 개수 균등)
        positions = [(i + 1) / (n_segments + 1) for i in range(n_segments)]
        segments = []
        tmp_dir = films_dir / f".tmp_{film_id}"
        tmp_dir.mkdir(exist_ok=True)

        for i, pos in enumerate(positions):
            start = duration * pos - segment_sec / 2
            start = max(0, min(start, duration - segment_sec))
            seg_path = tmp_dir / f"seg_{i}.mp4"
            print(f"  → segment {i+1}/{n_segments}: start={start:.1f}s (pos={pos:.0%})")
            extract_segment(src, seg_path, start, segment_sec)
            segments.append(seg_path)

        out_path = films_dir / f"{film_id}_highlight.mp4"
        concat_segments(segments, out_path)
        total_len = segment_sec * n_segments
        print(f"  ✓ {out_path.name} ({total_len:.0f}s)")

        # cleanup
        for p in segments:
            p.unlink()
        tmp_dir.rmdir()

        report["films"].append({
            "film_id": film_id,
            "source_file": filename,
            "source_duration_sec": duration,
            "highlight_file": out_path.name,
            "highlight_duration_sec": total_len,
            "segments": [
                {"start_sec": duration * pos - segment_sec / 2, "length_sec": segment_sec, "pos_ratio": pos}
                for pos in positions
            ],
            "license": license_tag,
        })

    (films_dir / "highlight_manifest.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] manifest → {films_dir / 'highlight_manifest.json'}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--films_dir", type=Path, required=True)
    p.add_argument("--segment_sec", type=float, default=240.0, help="segment 길이 (초). 기본 4분")
    p.add_argument("--n_segments", type=int, default=3, help="영화당 segment 개수. 기본 3 (20%/50%/80%)")
    args = p.parse_args()
    run(args.films_dir, args.segment_sec, args.n_segments)


if __name__ == "__main__":
    main()
