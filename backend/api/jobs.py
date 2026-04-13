# api/jobs.py — job 상태 조회, 타임라인, 다운로드, 삭제 엔드포인트
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import settings
from core import storage

router = APIRouter()


# --- GET /api/jobs/{job_id}/status ---
# 향후 확장:
#   status: "uploaded" | "queued" | "analyzing" | "eq_processing" | "completed" | "failed"
#   progress: 0.0~1.0, 현재는 항상 0.0
#   processed_size_bytes: completed 상태일 때만 값 존재
#   향후 작업 큐와 연결될 때 실제 상태 반영
@router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    if not storage.job_exists(job_id):
        raise HTTPException(status_code=404, detail="해당 job을 찾을 수 없습니다")

    meta = storage.load_job_meta(job_id)

    # 상태별 메시지 매핑
    status_messages = {
        "uploaded": "업로드 완료. 분석 기능은 추후 제공됩니다.",
        "queued": "분석 작업 큐에 등록되었습니다.",
        "analyzing": "영상을 분석하고 있습니다...",
        "eq_processing": "EQ를 적용하고 있습니다...",
        "completed": "모든 처리가 완료되었습니다.",
        "failed": "처리 중 오류가 발생했습니다.",
    }

    current_status = meta.get("status", "uploaded")

    return {
        "job_id": meta["job_id"],
        "status": current_status,
        "progress": meta.get("analysis_progress", 0.0),
        "message": status_messages.get(current_status, ""),
        "created_at": meta["created_at"],
        "updated_at": meta["updated_at"],
        "original_size_bytes": meta.get("original_size_bytes", 0),
        "processed_size_bytes": meta.get("processed_size_bytes"),
        "error_message": meta.get("error_message"),
    }


# --- GET /api/jobs/{job_id}/timeline ---
# 분석 완료된 job의 JSON 타임라인 반환 (명세서 V3.2 부록 C 스키마)
# 주 용도: 학술 분석, 디버깅, 내부 검증
@router.get("/{job_id}/timeline")
async def get_job_timeline(job_id: str):
    if not storage.job_exists(job_id):
        raise HTTPException(status_code=404, detail="해당 job을 찾을 수 없습니다")

    timeline = storage.load_timeline(job_id)
    if timeline is None:
        raise HTTPException(
            status_code=409,
            detail=f"아직 분석이 완료되지 않았습니다. 상태는 /api/jobs/{job_id}/status로 확인하세요",
        )

    return timeline


# --- GET /api/jobs/{job_id}/download/original ---
# 원본 영상 파일 다운로드 (A/B 비교를 위해 앱에서 필요)
@router.get("/{job_id}/download/original")
async def download_original(job_id: str):
    if not storage.job_exists(job_id):
        raise HTTPException(status_code=404, detail="해당 job을 찾을 수 없습니다")

    video_path = storage.get_original_video_path(job_id)
    if video_path is None or not video_path.exists():
        raise HTTPException(status_code=404, detail="원본 영상 파일을 찾을 수 없습니다")

    meta = storage.load_job_meta(job_id)
    ext = meta.get("original_ext", "mp4")

    # mov 확장자는 video/quicktime MIME 타입 사용
    media_type = "video/quicktime" if ext == "mov" else f"video/{ext}"

    return FileResponse(
        path=video_path,
        media_type=media_type,
        filename=f"original.{ext}",
        headers={"Cache-Control": "no-store"},
    )


# --- GET /api/jobs/{job_id}/download/processed ---
# EQ 적용된 영상 파일 다운로드
# DEV_FAKE_PROCESSED=true일 때는 status 체크를 bypass하고 원본을 반환
@router.get("/{job_id}/download/processed")
async def download_processed(job_id: str):
    if not storage.job_exists(job_id):
        raise HTTPException(status_code=404, detail="해당 job을 찾을 수 없습니다")

    meta = storage.load_job_meta(job_id)

    # DEV_FAKE_PROCESSED: 개발/테스트용 — 원본을 processed로 간주하여 반환
    if settings.DEV_FAKE_PROCESSED:
        video_path = storage.get_original_video_path(job_id)
        if video_path is None or not video_path.exists():
            raise HTTPException(status_code=404, detail="원본 영상 파일을 찾을 수 없습니다")
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename="processed.mp4",
            headers={"Cache-Control": "no-store"},
        )

    # 일반 모드: status가 completed인지 확인
    if meta.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"EQ 처리가 완료되지 않았습니다. 상태는 /api/jobs/{job_id}/status로 확인하세요",
        )

    video_path = storage.get_processed_video_path(job_id)
    if video_path is None or not video_path.exists():
        raise HTTPException(
            status_code=500,
            detail="처리된 영상 파일이 누락되었습니다",
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename="processed.mp4",
        headers={"Cache-Control": "no-store"},
    )


# --- DELETE /api/jobs/{job_id} ---
# Job과 관련된 모든 파일 삭제
@router.delete("/{job_id}")
async def delete_job(job_id: str):
    if not storage.job_exists(job_id):
        raise HTTPException(status_code=404, detail="해당 job을 찾을 수 없습니다")

    try:
        storage.delete_job(job_id)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="삭제 중 오류가 발생했습니다",
        )

    return {
        "status": "success",
        "job_id": job_id,
        "message": "job이 삭제되었습니다",
    }
