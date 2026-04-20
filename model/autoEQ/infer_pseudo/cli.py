"""CLI entrypoint for video → timeline.json analysis.

Usage:
    python -m model.autoEQ.infer_pseudo.cli \\
      --video path/to/movie.mp4 \\
      --ckpt runs/phase3_v2_gemini_target/best_model.pt \\
      --output movie.timeline.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import analyze_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MoodEQ video analysis → timeline.json")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num_mood_classes", type=int, default=4, choices=[4, 7])
    p.add_argument("--variant", type=str, default="base",
                   choices=["base", "gmu", "ast_gmu"],
                   help="training variant (must match ckpt). ast_gmu is V3.3 official final.")
    p.add_argument("--alpha_d", type=float, default=0.5,
                   help="dialogue protection strength (0.3~0.7 tunable)")
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--include_windows", action="store_true",
                   help="include per-window details in JSON (larger file)")
    p.add_argument("--work_dir", type=Path, default=None,
                   help="persistent scratch dir (keeps 16k WAV for debugging)")
    p.add_argument("--model_version", type=str, default="train_pseudo_v3.3")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    analyze_video(
        video_path=args.video,
        ckpt_path=args.ckpt,
        output_json=args.output,
        num_mood_classes=args.num_mood_classes,
        variant=args.variant,
        alpha_d=args.alpha_d,
        ema_alpha=args.ema_alpha,
        batch_size=args.batch_size,
        include_windows=args.include_windows,
        work_dir=args.work_dir,
        model_version=args.model_version,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
