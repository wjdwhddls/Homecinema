"""
최적 스피커 배치 FastAPI 라우터.

POST /api/optimize/speakers — 비동기 잡 시작, job_id 반환
GET  /api/optimize/status/{job_id} — 상태 + 결과 조회
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from core.absorption_db import HybridAbsorptionDatabase
from core.acoustic_simulator import AcousticSimulator
from core.optimization_job_store import OptimizationJobStore
from core.room_converter import (
    InsufficientWallsError,
    InvalidRoomGeometryError,
    convert_roomplan_to_pra_input,
)
from core.speaker_optimizer import EvaluatedCandidate, SpeakerOptimizer
from models.optimization_schemas import (
    AcousticMetrics,
    JobStatusResponse,
    OptimalResult,
    OptimizeRequest,
    OptimizeResponse,
    RoomSummary,
    SpeakerPosition,
    StereoPlacement,
    UserMaterialSelection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimize", tags=["optimization"])

_db: Optional[HybridAbsorptionDatabase] = None
_job_store = OptimizationJobStore()


def _get_db() -> HybridAbsorptionDatabase:
    global _db
    if _db is None:
        _db = HybridAbsorptionDatabase()
    return _db


@router.post("/speakers")
async def start_optimization(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    job_id = _job_store.create_job()
    background_tasks.add_task(_run_optimization_task, job_id, request)
    return {
        "job_id": job_id,
        "status": "pending",
        "estimated_seconds": 120,
        "poll_url": f"/api/optimize/status/{job_id}",
    }


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    status = _job_store.get_status(job_id)
    if status is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return status


def _run_optimization_task(job_id: str, request: OptimizeRequest) -> None:
    try:
        _job_store.update_status(job_id, "processing", progress=5)

        db = _get_db()
        selections_map = {
            sel.object_id: sel.material_key
            for sel in (request.user_material_selections or [])
        }
        converted = convert_roomplan_to_pra_input(
            request.roomplan_scan,
            db,
            listener_height=request.listener_height_m,
            user_material_selections=selections_map,
        )
        _job_store.update_status(job_id, "processing", progress=15)

        simulator = AcousticSimulator(converted)
        optimizer = SpeakerOptimizer(
            simulator,
            config_type=request.config_type,
            time_budget_seconds=180.0,
            speaker_dimensions={
                "width_m": request.speaker_dimensions.width_m,
                "height_m": request.speaker_dimensions.height_m,
                "depth_m": request.speaker_dimensions.depth_m,
            },
        )

        def progress_cb(done: int, total: int) -> None:
            if total <= 0:
                return
            pct = 15 + int(75 * done / total)
            _job_store.update_status(job_id, "processing", progress=pct)

        _job_store.update_status(job_id, "processing", progress=20)
        result = optimizer.find_optimal(progress_callback=progress_cb)
        _job_store.update_status(job_id, "processing", progress=95)

        response = _build_response(job_id, result, converted, request)
        _job_store.save_result(job_id, response)
        _job_store.update_status(job_id, "completed", progress=100)
        logger.info("최적화 완료: job=%s elapsed=%.1fs",
                    job_id, result["elapsed_seconds"])

    except (InsufficientWallsError, InvalidRoomGeometryError) as e:
        logger.warning("최적화 실패 (데이터 부족): %s", e)
        response = _build_error_response(job_id, "insufficient_data", str(e))
        _job_store.save_result(job_id, response)
        _job_store.update_status(job_id, "failed", progress=100)
    except Exception as e:
        logger.exception("최적화 실패: job=%s", job_id)
        response = _build_error_response(job_id, "error", str(e))
        _job_store.save_result(job_id, response)
        _job_store.update_status(job_id, "failed", progress=100)


def _build_response(
    job_id: str,
    result: dict,
    converted,
    request: OptimizeRequest,
) -> OptimizeResponse:
    listener = SpeakerPosition(
        x=float(converted.listener_position[0]),
        y=float(converted.listener_position[1]),
        z=float(converted.listener_position[2]),
    )

    def to_result(ec: EvaluatedCandidate) -> OptimalResult:
        if ec.candidate.is_stereo:
            left = _to_position(ec.candidate.left)
            right = _to_position(ec.candidate.right)
        else:
            left = right = _to_position(ec.candidate.left)
        placement = StereoPlacement(left=left, right=right, listener=listener)
        return OptimalResult(
            placement=placement,
            score=float(ec.score),
            metrics=AcousticMetrics(**ec.metrics) if ec.metrics else _zero_metrics(),
            rank=ec.rank,
        )

    best = to_result(result["best"])
    top_alts = [to_result(ec) for ec in result["top_alts"][1:]]

    warnings = list(result.get("warnings", []))
    if len(converted.floor_corners_2d) < 4:
        warnings.append("벽이 적어 방 형태 추정이 부정확할 수 있습니다.")
    if result["elapsed_seconds"] > 150.0:
        warnings.append("계산 시간이 길어 일부 후보 평가가 생략됐을 수 있습니다.")

    summary = RoomSummary(
        wall_count=len(request.roomplan_scan.get("walls", []) or []),
        object_count=len(converted.objects),
        floor_area_m2=float(converted.floor_area_m2),
        height_m=float(converted.height),
        volume_m3=float(converted.volume_m3),
    )

    return OptimizeResponse(
        status="success",
        job_id=job_id,
        best=best,
        top_alternatives=top_alts,
        room_summary=summary,
        computation_time_seconds=float(result["elapsed_seconds"]),
        warnings=warnings,
    )


def _to_position(arr) -> SpeakerPosition:
    return SpeakerPosition(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


def _zero_metrics() -> AcousticMetrics:
    return AcousticMetrics(
        rt60_seconds=0.0, rt60_low=0.0, rt60_mid=0.0,
        standing_wave_severity_db=0.0, flatness_db=0.0,
        early_reflection_ratio=0.0, direct_to_reverb_ratio_db=0.0,
    )


def _build_error_response(job_id: str, status: str, message: str) -> OptimizeResponse:
    return OptimizeResponse(
        status=status,
        job_id=job_id,
        computation_time_seconds=0.0,
        error_message=message,
    )
