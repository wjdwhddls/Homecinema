"""End-to-end video → timeline + EQ + EQ+FX 통합 파이프라인.

MoodEQ dual-layer 아키텍처 (BASE_MODEL.md §4d Phase 5-A) 를 새 영상에
한 번에 적용.

단계:
    1. infer_pseudo          → runs/demo_<name>/timeline.json         (scene V/A + mood + EQ preset)
    2. playback (1× EQ)      → runs/demo_<name>/<name>_eq_applied.mp4 (Layer 1 only)
    3. generate_fx_demo      → runs/demo_<name>/<name>_eq_fx.mp4      (Layer 1 + Layer 2)

산출물 구조:
    runs/demo_<name>/
      timeline.json
      <name>_eq_applied.mp4    ← Layer 1 (EQ only, 학술적 contribution)
      <name>_eq_fx.mp4         ← Layer 1 + 2 (EQ + FX, perceptual amplifier)
      run.log

Usage:
    venv/bin/python run_pipeline.py --video my_movie.mp4
    venv/bin/python run_pipeline.py --video my_movie.mp4 --name my_movie
    venv/bin/python run_pipeline.py --video my_movie.mp4 --skip-fx   # Layer 1 만
    venv/bin/python run_pipeline.py --video my_movie.mp4 --force      # 기존 결과 덮어쓰기

주의:
    - infer_pseudo 의 `AutoEQModelCog` 아키텍처 요구 때문에
      기본 ckpt 는 `runs/phase3_v2_gemini_target/best_model.pt` (K=4).
      BASE `runs/phase2a/2a2_A_K7_s2024/best.pt` 는 `AutoEQModelLiris`
      Enhanced arch 라 infer_pseudo 와 호환 안 됨 → 호환성 문제 해결 전엔
      v3 체크포인트 사용.
    - 결과 mp4 는 원본 비디오 스트림 그대로 + 처리된 오디오 재다중화
      (video copy, audio re-encode AAC 192k).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CKPT = Path("runs/phase3_v2_gemini_target/best_model.pt")
DEFAULT_NUM_MOOD = 4      # 위 ckpt 가 K=4 로 학습됨
DEFAULT_MODEL_VERSION = "phase3_v2_gemini_target"


def sanitize_name(video_path: Path) -> str:
    """파일명에서 유효한 디렉토리 이름 추출.

    예: 'KakaoTalk_Video_2026-04-23-00-05-15.mp4' → 'kakaotalk'
        'my_movie.mp4' → 'my_movie'
    """
    stem = video_path.stem.lower()
    # 타임스탬프, 복잡 문자 제거 — 문자/숫자/underscore 만
    cleaned = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    # 앞쪽 키워드만 사용 (너무 긴 이름 방지)
    parts = cleaned.split("_")
    if parts and parts[0] in ("kakaotalk", "video"):
        # 프리픽스 뒤 3단어 이내로
        return "_".join(parts[:min(3, len(parts))]).rstrip("_0123456789-")
    return parts[0] if parts else "demo"


def step_infer_pseudo(
    video: Path, timeline_out: Path, ckpt: Path,
    num_mood_classes: int, model_version: str, verbose: bool,
    alpha_d: float,
) -> None:
    """Step 1: infer_pseudo CLI 호출."""
    cmd = [
        "venv/bin/python", "-m", "model.autoEQ.infer_pseudo.cli",
        "--video", str(video),
        "--ckpt", str(ckpt),
        "--output", str(timeline_out),
        "--num_mood_classes", str(num_mood_classes),
        "--variant", "base",
        "--model_version", model_version,
        "--alpha_d", str(alpha_d),
    ]
    if not verbose:
        cmd.append("--quiet")
    print(f"[step 1] infer_pseudo → {timeline_out.name} (alpha_d={alpha_d})")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[step 1] done in {time.time() - t0:.1f}s")


def step_apply_eq(
    video: Path, timeline: Path, eq_out: Path, verbose: bool,
) -> None:
    """Step 2: playback pipeline (Layer 1 EQ, dialog-protected)."""
    from model.autoEQ.playback.pipeline import apply_eq_to_video

    print(f"[step 2] Layer 1 EQ (1× spec, dialog-protected) → {eq_out.name}")
    t0 = time.time()
    apply_eq_to_video(
        video_path=video,
        timeline_json=timeline,
        output_video=eq_out,
        preset_scale=1.0,
        bypass_dialogue_protection=False,
        verbose=verbose,
    )
    print(f"[step 2] done in {time.time() - t0:.1f}s")


def step_apply_fx(
    eq_video: Path, timeline: Path, fx_out: Path,
) -> None:
    """Step 3: generate_fx_demo 모듈 호출 (Layer 1 → Layer 1 + 2)."""
    from generate_fx_demo import generate_fx_video

    print(f"[step 3] Layer 2 FX (mood shelf + reverb) → {fx_out.name}")
    t0 = time.time()
    generate_fx_video(
        eq_video=eq_video,
        timeline_json=timeline,
        output_video=fx_out,
        crossfade_ms=50,   # in-place 경계 smoothing
    )
    print(f"[step 3] done in {time.time() - t0:.1f}s")


def main() -> int:
    p = argparse.ArgumentParser(
        description="MoodEQ dual-layer pipeline: video → EQ → EQ+FX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", type=Path, required=True, help="입력 영상 파일")
    p.add_argument("--name", type=str, default=None,
                   help="산출 디렉토리 suffix (기본: 파일명에서 추출)")
    p.add_argument("--output-root", type=Path, default=Path("runs"),
                   help="산출물 루트 (기본 runs/)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="infer_pseudo checkpoint (AutoEQModelCog 호환 필요)")
    p.add_argument("--num-mood-classes", type=int, default=DEFAULT_NUM_MOOD,
                   choices=[4, 7])
    p.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION)
    p.add_argument("--alpha-d", type=float, default=0.3,
                   help="대사 보호 강도: α_d 작을수록 강한 보호 "
                        "(g_eff = g_orig × (1 - (1-α_d)·density); α_d=0 완전 감쇠, "
                        "α_d=1 원본 유지). 0.3 권장 (infer_pseudo 기본 0.5 보다 강한 보호)")
    p.add_argument("--skip-fx", action="store_true",
                   help="Step 3 (FX layer) 건너뛰고 EQ-only 까지만 생성")
    p.add_argument("--skip-eq", action="store_true",
                   help="Step 2 건너뛰기 (이미 eq_applied.mp4 있다고 가정)")
    p.add_argument("--skip-timeline", action="store_true",
                   help="Step 1 건너뛰기 (이미 timeline.json 있다고 가정)")
    p.add_argument("--force", action="store_true",
                   help="기존 산출물 덮어쓰기")
    p.add_argument("--quiet", action="store_true", help="verbose 출력 최소")
    args = p.parse_args()

    video_path = args.video.resolve()
    if not video_path.is_file():
        print(f"[error] 영상 없음: {video_path}", file=sys.stderr)
        return 1
    if not args.ckpt.is_file():
        print(f"[error] checkpoint 없음: {args.ckpt}", file=sys.stderr)
        return 1

    name = args.name or sanitize_name(video_path)
    out_dir = args.output_root / f"demo_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = out_dir / "timeline.json"
    eq_path = out_dir / f"{name}_eq_applied.mp4"
    fx_path = out_dir / f"{name}_eq_fx.mp4"
    log_path = out_dir / "run.log"

    print(f"=== MoodEQ Dual-Layer Pipeline ===")
    print(f"  input    : {video_path}")
    print(f"  name     : {name}")
    print(f"  out_dir  : {out_dir}")
    print(f"  ckpt     : {args.ckpt}")
    print(f"  K        : {args.num_mood_classes}")
    print()

    pipeline_t0 = time.time()

    # Step 1: timeline.json
    if args.skip_timeline and timeline_path.exists():
        print(f"[step 1] skipped (existing: {timeline_path.name})")
    elif timeline_path.exists() and not args.force:
        print(f"[step 1] skipped (exists — --force for rebuild)")
    else:
        step_infer_pseudo(
            video_path, timeline_path, args.ckpt,
            args.num_mood_classes, args.model_version,
            verbose=not args.quiet,
            alpha_d=args.alpha_d,
        )

    # Step 2: Layer 1 EQ
    if args.skip_eq and eq_path.exists():
        print(f"[step 2] skipped (existing: {eq_path.name})")
    elif eq_path.exists() and not args.force:
        print(f"[step 2] skipped (exists — --force for rebuild)")
    else:
        step_apply_eq(video_path, timeline_path, eq_path, verbose=not args.quiet)

    # Step 3: Layer 2 FX
    if args.skip_fx:
        print(f"[step 3] skipped (--skip-fx)")
    elif fx_path.exists() and not args.force:
        print(f"[step 3] skipped (exists — --force for rebuild)")
    else:
        step_apply_fx(eq_path, timeline_path, fx_path)

    total = time.time() - pipeline_t0
    print(f"\n=== pipeline complete in {total:.1f}s ===")
    print(f"  timeline : {timeline_path}")
    print(f"  1× EQ    : {eq_path}")
    if not args.skip_fx:
        print(f"  1× EQ+FX : {fx_path}")
    print()
    print("### 다음 단계 ###")
    print(f"  # 3-way 비교 뷰어 실행:")
    print(f"  venv/bin/python live_compare_fx.py \\")
    print(f"      --processed {eq_path} \\")
    print(f"      --demo      {fx_path} \\")
    print(f"      --original  {video_path} \\")
    print(f"      --timeline  {timeline_path}")

    # 간단한 run log
    log_path.write_text(
        f"=== MoodEQ pipeline run ===\n"
        f"video: {video_path}\n"
        f"name: {name}\n"
        f"ckpt: {args.ckpt}\n"
        f"num_mood: {args.num_mood_classes}\n"
        f"elapsed_sec: {total:.1f}\n"
        f"timeline: {timeline_path}\n"
        f"eq_applied: {eq_path}\n"
        f"eq_fx: {fx_path if not args.skip_fx else '(skipped)'}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
