# api/health.py — 서버 헬스 체크 엔드포인트
from fastapi import APIRouter
from models.schemas import HealthResponse

router = APIRouter()

# 서버 시작 시간은 main.py에서 설정
_started_at: str = ""


def set_started_at(ts: str) -> None:
    global _started_at
    _started_at = ts


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        service="mood-eq-backend",
        version="0.1.0",
        started_at=_started_at,
    )
