"""MoodEQ dual-layer 파이프라인을 backend job 디렉토리에 맞춰 subprocess 실행.

BackgroundTasks 로 호출되는 진입점 = run_moodeq_pipeline(job_id).

흐름:
  1. status=analyzing (progress 0.05)
  2. run_pipeline.py 실행 — data/jobs/{id}/demo_work/ 에 산출물 생성
       stdout 의 "[step N]" 패턴을 감지해 progress 를 갱신
  3. demo_work/ 아래 파일을 job_dir 직하로 rename:
       timeline.json             → timeline.json
       work_eq_applied.mp4       → eq_applied.mp4
       work_eq_fx.mp4            → processed.mp4
  4. status=completed (progress 1.0, processed_size_bytes 기록)
  실패 시 status=failed + error_message.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from core import storage

logger = logging.getLogger(__name__)

# backend/ 에서 2 단계 위 = 프로젝트 루트 (Homecinema/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_BIN = _PROJECT_ROOT / "venv" / "bin" / "python"
_RUN_PIPELINE = _PROJECT_ROOT / "run_pipeline.py"

# 파이프라인 내부 임시 디렉토리 이름 — run_pipeline.py 가 demo_{name}/ 을 만듬
_WORK_NAME = "work"

# stdout 라인 키워드 → (status, progress) 매핑
_STAGE_MARKERS: list[tuple[str, str, float]] = [
    ("[step 1]", "analyzing", 0.10),
    ("[step 2]", "analyzing", 0.45),
    ("[step 3]", "eq_processing", 0.75),
]


def _build_command(video_path: Path, job_dir: Path) -> list[str]:
    # subprocess cwd 가 프로젝트 루트로 변경되므로 **절대경로** 로 고정.
    # backend 는 backend/ 에서 기동되어 JOBS_DATA_DIR=./data/jobs 를 쓰지만
    # subprocess 의 cwd 가 다르므로 상대경로를 그대로 넘기면 안 됨.
    return [
        str(_PYTHON_BIN),
        str(_RUN_PIPELINE),
        "--video", str(video_path.resolve()),
        "--output-root", str(job_dir.resolve()),
        "--name", _WORK_NAME,
        "--quiet",
    ]


def _tail(text: str, n: int = 10) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:]) if lines else ""


def _move_outputs(work_dir: Path, job_dir: Path) -> Path:
    """demo_work/ 아래 산출물을 job_dir 직하로 이동. processed.mp4 경로 반환."""
    src_timeline = work_dir / "timeline.json"
    src_eq = work_dir / f"{_WORK_NAME}_eq_applied.mp4"
    src_fx = work_dir / f"{_WORK_NAME}_eq_fx.mp4"

    missing = [p.name for p in (src_timeline, src_eq, src_fx) if not p.exists()]
    if missing:
        raise RuntimeError(f"pipeline 산출물 누락: {missing}")

    dst_timeline = job_dir / "timeline.json"
    dst_eq = job_dir / "eq_applied.mp4"
    dst_processed = job_dir / "processed.mp4"

    # 이전 실행 잔재 제거 (force 재처리 대비)
    for p in (dst_timeline, dst_eq, dst_processed):
        p.unlink(missing_ok=True)

    src_timeline.rename(dst_timeline)
    src_eq.rename(dst_eq)
    src_fx.rename(dst_processed)

    # work 디렉토리 전체 정리 (run.log 등 남은 파일 포함)
    shutil.rmtree(work_dir, ignore_errors=True)
    return dst_processed


def run_moodeq_pipeline(job_id: str) -> None:
    """BackgroundTasks 에서 호출되는 진입점. 예외는 먹고 meta 에만 기록."""
    job_dir = storage.get_job_dir(job_id)
    meta = storage.load_job_meta(job_id)
    if meta is None:
        logger.error("[pipeline] meta 없음: %s", job_id)
        return

    video_path = storage.get_original_video_path(job_id)
    if video_path is None or not video_path.exists():
        storage.update_job_status(
            job_id, status="failed", error_message="원본 영상 파일을 찾을 수 없습니다"
        )
        return

    try:
        storage.update_job_status(job_id, status="analyzing", progress=0.05)

        cmd = _build_command(video_path, job_dir)
        logger.info("[pipeline] start job=%s cmd=%s", job_id, " ".join(cmd))

        # run_pipeline.py 내부에서 상대경로 'venv/bin/python' 을 재사용하므로
        # cwd 는 반드시 프로젝트 루트여야 함.
        proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        stdout_buf: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_buf.append(line)
            for marker, status, progress in _STAGE_MARKERS:
                if marker in line:
                    storage.update_job_status(
                        job_id, status=status, progress=progress
                    )
                    logger.info("[pipeline] %s → %s %.0f%%",
                                job_id, status, progress * 100)
                    break

        ret = proc.wait()
        full_output = "".join(stdout_buf)

        if ret != 0:
            # 실패 시 전체 output 을 backend 로그에 덤프 (디버깅용),
            # meta.error_message 에는 tail 40줄 (앱 UI 표시용).
            logger.error(
                "[pipeline] subprocess exit=%d job=%s\n"
                "===== full stdout/stderr =====\n%s\n"
                "===== end =====",
                ret, job_id, full_output,
            )
            raise RuntimeError(
                f"run_pipeline.py exit={ret}\nstdout/stderr tail:\n{_tail(full_output, n=40)}"
            )

        work_dir = job_dir.resolve() / f"demo_{_WORK_NAME}"
        processed = _move_outputs(work_dir, job_dir.resolve())
        size = processed.stat().st_size

        storage.update_job_status(
            job_id,
            status="completed",
            progress=1.0,
            processed_size_bytes=size,
        )
        logger.info("[pipeline] done job=%s size=%d", job_id, size)
    except Exception as e:
        logger.exception("[pipeline] failed job=%s", job_id)
        storage.update_job_status(
            job_id, status="failed", error_message=str(e)
        )
