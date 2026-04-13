# api/upload.py — 영상 파일 업로드 엔드포인트
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from config import settings
from core import storage

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1MB


@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
):
    # 1. 파일명 확인
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 제공되지 않았습니다")

    # 2. 확장자 검증
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다 (허용: {allowed})",
        )

    # 3. Content-Length 헤더 1차 체크
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"파일이 너무 큽니다 (최대 {settings.MAX_UPLOAD_SIZE_MB}MB)",
        )

    # 4. UUID 생성 + 임시 파일 경로
    job_id = str(uuid.uuid4())
    tmp_path = storage.get_tmp_upload_path(job_id, ext)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Streaming 저장 + 실시간 크기 체크
    total_size = 0
    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > settings.max_upload_size_bytes:
                    # context manager가 파일을 닫은 후 cleanup
                    raise HTTPException(
                        status_code=413,
                        detail=f"파일이 너무 큽니다 (최대 {settings.MAX_UPLOAD_SIZE_MB}MB)",
                    )
                await f.write(chunk)

        # 6. Job 디렉토리로 이동 + meta.json 생성
        await storage.finalize_upload(
            job_id=job_id,
            tmp_path=tmp_path,
            original_filename=file.filename,
            ext=ext,
            size_bytes=total_size,
        )
    except HTTPException:
        # HTTPException은 그대로 전파, 임시 파일만 정리
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        # 예상치 못한 에러 시 임시 파일 정리
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="파일 저장 중 오류가 발생했습니다",
        )

    return {
        "status": "success",
        "job_id": job_id,
        "original_filename": file.filename,
        "saved_filename": f"original.{ext}",
        "size_bytes": total_size,
        "message": "영상 업로드가 완료되었습니다",
    }
