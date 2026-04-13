# main.py — FastAPI 앱 진입점
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api import health, upload, jobs

# 서버 시작 시간 기록
SERVER_STARTED_AT = datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 이벤트"""
    # Startup: 필요한 디렉토리 생성
    settings.upload_tmp_path.mkdir(parents=True, exist_ok=True)
    settings.jobs_data_path.mkdir(parents=True, exist_ok=True)

    # health 엔드포인트에 시작 시간 설정
    health.set_started_at(SERVER_STARTED_AT)

    print(f"[Mood EQ Backend] 서버 시작: {SERVER_STARTED_AT}")
    print(f"[Mood EQ Backend] 업로드 임시 디렉토리: {settings.upload_tmp_path}")
    print(f"[Mood EQ Backend] Job 데이터 디렉토리: {settings.jobs_data_path}")
    print(f"[Mood EQ Backend] 최대 업로드 크기: {settings.MAX_UPLOAD_SIZE_MB}MB")
    print(f"[Mood EQ Backend] DEV_FAKE_PROCESSED: {settings.DEV_FAKE_PROCESSED}")

    yield  # 앱 실행 중

    # Shutdown
    print("[Mood EQ Backend] 서버 종료")


app = FastAPI(
    title="Mood EQ Backend",
    description="영화 분위기 자동 EQ 적용 백엔드 서버",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 설정 — allow_origins=["*"] 금지, 명시적 origin 사용
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
