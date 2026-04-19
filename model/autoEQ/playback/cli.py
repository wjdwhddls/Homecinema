"""CLI entry: timeline.json + video → EQ-applied video.

Usage:
    python -m model.autoEQ.playback.cli \\
      --video movie.mp4 \\
      --timeline movie.timeline.json \\
      --output movie_eq.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import apply_eq_to_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply timeline EQ to video")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--timeline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crossfade_ms", type=int, default=300)
    p.add_argument("--audio_bitrate", type=str, default="192k")
    p.add_argument("--work_dir", type=Path, default=None,
                   help="persistent scratch dir (keeps extracted/processed WAV)")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    apply_eq_to_video(
        video_path=args.video,
        timeline_json=args.timeline,
        output_video=args.output,
        crossfade_ms=args.crossfade_ms,
        audio_bitrate=args.audio_bitrate,
        work_dir=args.work_dir,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
