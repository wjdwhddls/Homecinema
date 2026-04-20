"""
최적 스피커 배치 FastAPI 라우터 (xRIR 버전)

POST /api/xrir/initial-position — 임시 스피커 위치 계산
POST /api/xrir/speakers         — roomplan JSON + ref_rir.wav → 최적 위치 계산
GET  /api/xrir/status/{job_id}  — 결과 확인
"""

from __future__ import annotations

import json
import logging

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form

from core.optimization_job_store import OptimizationJobStore
from core.xrir_pipeline import run_xrir_pipeline
from core.initial_speaker_position import compute_initial_speaker_position
from core.roomplan_to_numpy import compute_listener_position

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/xrir", tags=["xrir"])

_job_store = OptimizationJobStore()


# ── 임시 스피커 위치 계산 ────────────────────────────────────────

@router.post("/initial-position")
async def get_initial_speaker_position(
    roomplan_scan: str = Form(...),
    listener_height_m: float = Form(1.2),
    speaker_height_m: float = Form(1.2),
) -> dict:
    """
    스캔 직후 sweep 측정을 위한 임시 스피커 위치 반환
    청취자(원점) → 정면 방향 벽 사이 중간 지점
    """
    try:
        roomplan_json = json.loads(roomplan_scan)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {e}")

    walls = roomplan_json.get("walls", [])
    if len(walls) < 3:
        raise HTTPException(status_code=400, detail="벽이 최소 3개 필요합니다")

    listener_pos = compute_listener_position(walls, listener_height_m)
    initial_pos = compute_initial_speaker_position(
        walls=walls,
        listener_pos=listener_pos,
        speaker_height=speaker_height_m,
    )

    return {
        "listener_position": {
            "x": float(listener_pos[0]),
            "y": float(listener_pos[1]),
            "z": float(listener_pos[2]),
        },
        "initial_speaker_position": {
            "x": float(initial_pos[0]),
            "y": float(initial_pos[1]),
            "z": float(initial_pos[2]),
        },
    }


# ── 최적 스피커 위치 계산 ────────────────────────────────────────

@router.post("/speakers")
async def start_optimization(
    background_tasks: BackgroundTasks,
    roomplan_scan: str = Form(...),
    ref_rir: UploadFile = File(...),
    listener_height_m: float = Form(1.2),
    speaker_height_m: float = Form(1.2),
    top_k: int = Form(5),
) -> dict:
    try:
        roomplan_json = json.loads(roomplan_scan)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {e}")

    ref_rir_bytes = await ref_rir.read()

    job_id = _job_store.create_job()
    background_tasks.add_task(
        _run_task, job_id, roomplan_json, ref_rir_bytes,
        listener_height_m, speaker_height_m, top_k,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "estimated_seconds": 60,
        "poll_url": f"/api/xrir/status/{job_id}",
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    status = _job_store.get_status(job_id)
    if status is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return status


def _run_task(
    job_id: str,
    roomplan_json: dict,
    ref_rir_bytes: bytes,
    listener_height: float,
    speaker_height: float,
    top_k: int,
) -> None:
    try:
        _job_store.update_status(job_id, "processing", progress=10)

        results = run_xrir_pipeline(
            roomplan_json=roomplan_json,
            ref_rir_bytes=ref_rir_bytes,
            top_k=top_k,
            listener_height=listener_height,
            speaker_height=speaker_height,
        )

        _job_store.update_status(job_id, "processing", progress=95)

        _job_store.save_result(job_id, {
            "status": "success",
            "job_id": job_id,
            "best": results[0] if results else None,
            "top_alternatives": results[1:],
            "room_summary": None,
            "computation_time_seconds": 0.0,
            "warnings": [],
            "error_message": None,
        })
        _job_store.update_status(job_id, "completed", progress=100)
        logger.info("최적화 완료: job=%s", job_id)

    except Exception as e:
        logger.exception("최적화 실패: job=%s", job_id)
        _job_store.save_result(job_id, {"status": "error", "message": str(e)})
        _job_store.update_status(job_id, "failed", progress=100)
