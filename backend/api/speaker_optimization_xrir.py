"""
최적 스피커 배치 FastAPI 라우터 (xRIR 버전)

POST /api/xrir/initial-position — 임시 스피커 위치 계산
POST /api/xrir/speakers         — roomplan JSON + recorded.wav + sweep.wav + mesh.bin → 최적 위치
GET  /api/xrir/status/{job_id}  — 결과 확인
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form

from core.optimization_job_store import OptimizationJobStore
from core.xrir_pipeline import run_xrir_pipeline
from core.initial_speaker_position import compute_initial_speaker_position
from core.roomplan_to_numpy import compute_listener_position
from core.sweep_deconvolution import deconvolve_sweep
from core.topview_generator import generate_topview
import time

DATA_SAVE_DIR = Path("./data/roomplan_scans")
DATA_SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
    
    # 데이터 수집 저장
    save_path = DATA_SAVE_DIR / f"scan_{int(time.time())}.json"
    save_path.write_text(json.dumps(roomplan_json, ensure_ascii=False, indent=2))
    logger.info("roomplan 저장: %s", save_path)

    walls = roomplan_json.get("walls", [])
    if len(walls) < 3:
        raise HTTPException(status_code=400, detail="벽이 최소 3개 필요합니다")

    listener_pos = compute_listener_position(walls, listener_height_m)
    initial_pos = compute_initial_speaker_position(
        walls=walls,
        listener_pos=listener_pos,
        speaker_height=speaker_height_m,
    )

    listener_dict = {
        "x": float(listener_pos[0]),
        "y": float(listener_pos[1]),
        "z": float(listener_pos[2]),
    }
    initial_dict = {
        "x": float(initial_pos[0]),
        "y": float(initial_pos[1]),
        "z": float(initial_pos[2]),
    }

    topview = generate_topview(
        roomplan_json=roomplan_json,
        listener_pos=listener_dict,
        speaker_positions={"initial": initial_dict},
    )

    return {
        "listener_position": listener_dict,
        "initial_speaker_position": initial_dict,
        "topview_image": topview,
    }


# ── 최적 스피커 위치 계산 ────────────────────────────────────────

@router.post("/speakers")
async def start_optimization(
    background_tasks: BackgroundTasks,
    roomplan_scan: str = Form(...),
    recorded: UploadFile = File(...),           # 마이크 녹음 wav
    sweep: UploadFile = File(...),              # sweep 원본 wav
    mesh: Optional[UploadFile] = File(None),    # LiDAR 메쉬 (없으면 fallback)
    listener_height_m: float = Form(1.2),
    speaker_height_m: float = Form(1.2),
    top_k: int = Form(5),
    initial_speaker_x: float = Form(...),
    initial_speaker_y: float = Form(...),
    initial_speaker_z: float = Form(...),
) -> dict:
    try:
        roomplan_json = json.loads(roomplan_scan)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {e}")

    recorded_bytes = await recorded.read()
    sweep_bytes    = await sweep.read()
    mesh_bytes     = await mesh.read() if mesh else None

    job_id = _job_store.create_job()
    ref_src_pos = np.array([initial_speaker_x, initial_speaker_y, initial_speaker_z], dtype=np.float32)
    background_tasks.add_task(
        _run_task, job_id, roomplan_json,
        recorded_bytes, sweep_bytes, mesh_bytes,
        listener_height_m, speaker_height_m, top_k,
        ref_src_pos,
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


# ── 백그라운드 작업 ──────────────────────────────────────────────

def _run_task(
    job_id: str,
    roomplan_json: dict,
    recorded_bytes: bytes,
    sweep_bytes: bytes,
    mesh_bytes: Optional[bytes],
    listener_height: float,
    speaker_height: float,
    top_k: int,
    ref_src_pos: np.ndarray
) -> None:
    try:
        start_time = time.time()
        _job_store.update_status(job_id, "processing", progress=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # ── 1. 파일 저장 ────────────────────────────────────
            recorded_path = tmpdir / "recorded.wav"
            sweep_path    = tmpdir / "sweep.wav"
            recorded_path.write_bytes(recorded_bytes)
            sweep_path.write_bytes(sweep_bytes)

            mesh_bin_path = None
            if mesh_bytes:
                mesh_bin_path = str(tmpdir / "mesh.bin")
                (tmpdir / "mesh.bin").write_bytes(mesh_bytes)
                logger.info("mesh.bin 수신: %d bytes", len(mesh_bytes))
            else:
                logger.info("mesh.bin 없음 → roomplan JSON fallback 사용")

            _job_store.update_status(job_id, "processing", progress=20)

            # roomplan JSON 저장
            scan_path = DATA_SAVE_DIR / f"scan_{job_id}.json"
            scan_path.write_text(json.dumps(roomplan_json, ensure_ascii=False, indent=2))

            # recorded.wav, sweep.wav도 영구 저장
            perm_dir = DATA_SAVE_DIR / job_id
            perm_dir.mkdir(parents=True, exist_ok=True)
            (perm_dir / "recorded.wav").write_bytes(recorded_bytes)
            (perm_dir / "sweep.wav").write_bytes(sweep_bytes)
            if mesh_bytes:
                (perm_dir / "mesh.bin").write_bytes(mesh_bytes)

            # ── 2. Deconvolution → ref_rir.wav ──────────────────
            ref_rir_path = str(tmpdir / "ref_rir.wav")
            deconvolve_sweep(
                recorded_path=str(recorded_path),
                sweep_path=str(sweep_path),
                output_path=ref_rir_path,
            )

            _job_store.update_status(job_id, "processing", progress=40)

            # ── 3. xRIR 추론 ────────────────────────────────────
            import soundfile as sf
            ref_rir_bytes = Path(ref_rir_path).read_bytes()

            results = run_xrir_pipeline(
                roomplan_json=roomplan_json,
                ref_rir_bytes=ref_rir_bytes,
                ref_src_pos=ref_src_pos,
                top_k=top_k,
                listener_height=listener_height,
                speaker_height=speaker_height,
                mesh_bin_path=mesh_bin_path,
            )

        _job_store.update_status(job_id, "processing", progress=95)
        best = results[0] if results else None

        topview = None
        if best:
            listener_dict = {
                "x": float(best["placement"]["listener"]["x"]),
                "y": float(best["placement"]["listener"]["y"]),
                "z": float(best["placement"]["listener"]["z"]),
            }
            topview = generate_topview(
                roomplan_json=roomplan_json,
                listener_pos=listener_dict,
                speaker_positions={
                    "left":  best["placement"]["left"],
                    "right": best["placement"]["right"],
                },
            )

        _job_store.save_result(job_id, {
            "status": "success",
            "job_id": job_id,
            "best": best,
            "top_alternatives": results[1:],
            "room_summary": None,
            "computation_time_seconds": time.time() - start_time,
            "warnings": [],
            "error_message": None,
            "topview_image": topview,
        })
        _job_store.update_status(job_id, "completed", progress=100)
        logger.info("최적화 완료: job=%s", job_id)

    except Exception as e:
        logger.exception("최적화 실패: job=%s", job_id)
        _job_store.save_result(job_id, {"status": "error", "message": str(e)})
        _job_store.update_status(job_id, "failed", progress=100)
