"""
Phase 3 최적화 API 요청/응답 스키마.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UserMaterialSelection(BaseModel):
    object_id: str
    material_key: str


class SpeakerDimensions(BaseModel):
    """사용자 스피커 물리 치수 (meter 단위). 배치 가능성 검사에 사용."""

    width_m: float = Field(gt=0, le=2.0)
    height_m: float = Field(gt=0, le=2.5)
    depth_m: float = Field(gt=0, le=2.0)


class OptimizeRequest(BaseModel):
    roomplan_scan: Dict[str, Any]
    speaker_dimensions: SpeakerDimensions
    listener_height_m: float = Field(default=1.2, ge=0.5, le=2.0)
    config_type: Literal["single", "stereo"] = "stereo"
    user_material_selections: Optional[List[UserMaterialSelection]] = None


class AcousticMetrics(BaseModel):
    rt60_seconds: float
    rt60_low: float
    rt60_mid: float
    standing_wave_severity_db: float
    flatness_db: float
    early_reflection_ratio: float
    direct_to_reverb_ratio_db: float


class SpeakerPosition(BaseModel):
    x: float
    y: float
    z: float


class StereoPlacement(BaseModel):
    left: SpeakerPosition
    right: SpeakerPosition
    listener: SpeakerPosition


class OptimalResult(BaseModel):
    placement: StereoPlacement
    score: float
    metrics: AcousticMetrics
    rank: int


class RoomSummary(BaseModel):
    wall_count: int
    object_count: int
    floor_area_m2: float
    height_m: float
    volume_m3: float


class OptimizeResponse(BaseModel):
    status: Literal["success", "insufficient_data", "error"]
    job_id: str
    best: Optional[OptimalResult] = None
    top_alternatives: List[OptimalResult] = []
    room_summary: Optional[RoomSummary] = None
    computation_time_seconds: float = 0.0
    warnings: List[str] = []
    error_message: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress_percent: Optional[int] = None
    estimated_remaining_seconds: Optional[int] = None
    result: Optional[OptimizeResponse] = None
