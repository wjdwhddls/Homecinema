"""
EQ 분석 FastAPI 라우터

POST /api/eq/analyze
  입력: sweep.wav (원본), recorded.wav (최적 위치에서 녹음)
  출력: 23밴드 보정값, Bass/Mid/Treble 요약, Parametric EQ 필터
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from core.eq_analyzer import run_eq_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eq", tags=["eq"])


@router.post("/analyze")
async def analyze_eq(
    sweep:    UploadFile = File(...),     # sweep 원본 wav
    recorded: UploadFile = File(...),     # 최적 위치에서 마이크 녹음 wav
    sr:       int        = Form(44100),   # 처리 샘플레이트
) -> dict:
    """
    sweep + 녹음 → EQ 보정값 반환

    흐름:
      1. 파일 수신
      2. 시간 정렬 (cross-correlation)
      3. Transfer Function 계산 (ESS deconvolution)
      4. 23밴드 보정값 + Bass/Mid/Treble + Parametric EQ 반환
    """
    try:
        sweep_bytes    = await sweep.read()
        recorded_bytes = await recorded.read()

        if len(sweep_bytes) == 0:
            raise HTTPException(400, "sweep 파일이 비어 있습니다.")
        if len(recorded_bytes) == 0:
            raise HTTPException(400, "recorded 파일이 비어 있습니다.")

        logger.info(
            "EQ 분석 시작: sweep=%d bytes, recorded=%d bytes",
            len(sweep_bytes), len(recorded_bytes),
        )

        result = run_eq_pipeline(
            sweep_bytes=sweep_bytes,
            recorded_bytes=recorded_bytes,
            sr=sr,
        )

        logger.info("EQ 분석 완료: %d밴드", len(result["bands"]))
        return {"status": "success", **result}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("EQ 분석 실패")
        raise HTTPException(500, f"EQ 분석 중 오류가 발생했습니다: {e}")
