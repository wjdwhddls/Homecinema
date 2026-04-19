"""CC 영화 일괄 다운로드.

Usage:
    python -m scripts.cc_movies.download_films --output_dir dataset/autoEQ/CCMovies/films/
    python -m scripts.cc_movies.download_films --output_dir ... --film_ids sintel,tears_of_steel

`FilmEntry.download_url`이 설정된 엔트리만 처리한다. URL이 없는(TODO) 엔트리는 스킵하고
마지막에 안내 메시지를 출력한다.

기본 다운로드는 `urllib.request`를 사용(표준 라이브러리만). `--use_yt_dlp`를 주면 Vimeo
같은 호스트용으로 yt-dlp subprocess 호출(설치 필요).

각 영화별로:
  - `<output_dir>/<film_id>.mp4` 생성
  - `<output_dir>/manifest.json` 업데이트 (SHA-256, 크기, 실제 duration은 extract_windows 단계에서 ffprobe로 검증)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.cc_movies.film_list import ALL_FILMS, FilmEntry, get_downloadable_films


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_http(url: str, dest: Path) -> None:
    """curl subprocess로 HTTP(S) 다운. macOS 시스템 인증서를 쓰므로 SSL 이슈 회피."""
    cmd = [
        "curl", "-L", "--fail", "--silent", "--show-error",
        "-o", str(dest),
        "-A", "MoodEQ-research/1.0",
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_yt_dlp(url: str, dest: Path) -> None:
    """yt-dlp subprocess. Vimeo, YouTube 등에 사용."""
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "--no-playlist",
        "-o", str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)


def download_film(entry: FilmEntry, output_dir: Path, use_yt_dlp: bool = False) -> dict:
    dest = output_dir / f"{entry.film_id}.mp4"
    if dest.is_file():
        print(f"  [skip] {entry.film_id}: 이미 존재 ({dest.stat().st_size/1e6:.1f} MB)")
        return {
            "film_id": entry.film_id,
            "status": "skipped_existing",
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "sha256": _sha256_file(dest),
        }

    print(f"  [download] {entry.film_id} <- {entry.download_url}")
    t0 = time.time()
    try:
        if use_yt_dlp or entry.source in {"vimeo", "youtube"}:
            _download_yt_dlp(entry.download_url, dest)
        else:
            _download_http(entry.download_url, dest)
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"    ERROR: {e}", file=sys.stderr)
        return {"film_id": entry.film_id, "status": "error", "error": str(e)}

    elapsed = time.time() - t0
    size = dest.stat().st_size
    sha = _sha256_file(dest)
    print(f"    OK {size/1e6:.1f} MB in {elapsed:.1f}s  sha256={sha[:16]}...")
    return {
        "film_id": entry.film_id,
        "status": "downloaded",
        "path": str(dest),
        "size_bytes": size,
        "sha256": sha,
        "elapsed_sec": elapsed,
        "license": entry.license,
        "source": entry.source,
        "expected_duration_sec": entry.duration_sec,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CC 영화 일괄 다운로드")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--film_ids",
        type=str,
        default="",
        help="쉼표 구분 영화 ID. 빈 값이면 다운 가능한 전체.",
    )
    parser.add_argument("--use_yt_dlp", action="store_true",
                        help="모든 소스를 yt-dlp로 다운 (HTTP 소스에도 적용)")
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.film_ids:
        wanted = {x.strip() for x in args.film_ids.split(",") if x.strip()}
        targets = [e for e in ALL_FILMS if e.film_id in wanted]
        missing = wanted - {e.film_id for e in targets}
        if missing:
            print(f"[warn] 알 수 없는 film_id: {sorted(missing)}", file=sys.stderr)
        targets = [e for e in targets if e.download_url]
    else:
        targets = get_downloadable_films()

    print(f"[info] 다운 대상 {len(targets)}편, output_dir={output_dir}")
    results: list[dict] = []
    for entry in targets:
        results.append(download_film(entry, output_dir, use_yt_dlp=args.use_yt_dlp))

    manifest_path = output_dir / "download_manifest.json"
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_attempts": len(results),
        "downloaded": sum(1 for r in results if r["status"] == "downloaded"),
        "skipped_existing": sum(1 for r in results if r["status"] == "skipped_existing"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[info] manifest: {manifest_path}")
    print(f"[info] OK={manifest['downloaded']} skip={manifest['skipped_existing']} err={manifest['errors']}")

    return 0 if manifest["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
