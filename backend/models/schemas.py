# models/schemas.py — API 응답 스키마
from pydantic import BaseModel
from typing import Literal


# Job 상태 타입
JobStatus = Literal[
    "uploaded",
    "queued",
    "analyzing",
    "eq_processing",
    "completed",
    "failed",
]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    started_at: str


class UploadResponse(BaseModel):
    status: str
    job_id: str
    original_filename: str
    saved_filename: str
    size_bytes: int
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    message: str | None
    created_at: str
    updated_at: str
    original_size_bytes: int
    processed_size_bytes: int | None
    error_message: str | None


class DeleteResponse(BaseModel):
    status: str
    job_id: str
    message: str
