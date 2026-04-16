"""
파일 기반 최적화 Job 저장소.

tmp/optimization_results/{job_id}/{status,result}.json
디스크 경합을 줄이기 위해 temp write → rename 패턴 사용.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models.optimization_schemas import JobStatusResponse, OptimizeResponse

logger = logging.getLogger(__name__)


class OptimizationJobStore:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        default = Path(__file__).resolve().parent.parent / "tmp" / "optimization_results"
        self.base_dir = base_dir or default
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _result_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "result.json"

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(
            self._status_path(job_id),
            {
                "job_id": job_id,
                "status": "pending",
                "progress_percent": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return job_id

    def update_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[int] = None,
    ) -> None:
        current = self._read_json(self._status_path(job_id)) or {"job_id": job_id}
        current["status"] = status
        if progress is not None:
            current["progress_percent"] = int(progress)
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write_json(self._status_path(job_id), current)

    def save_result(self, job_id: str, response: OptimizeResponse) -> None:
        self._atomic_write_json(
            self._result_path(job_id), response.model_dump(mode="json")
        )

    def get_status(self, job_id: str) -> Optional[JobStatusResponse]:
        status_data = self._read_json(self._status_path(job_id))
        if status_data is None:
            return None
        result_data = self._read_json(self._result_path(job_id))
        result_model = (
            OptimizeResponse.model_validate(result_data)
            if result_data is not None
            else None
        )
        return JobStatusResponse(
            job_id=status_data.get("job_id", job_id),
            status=status_data.get("status", "pending"),
            progress_percent=status_data.get("progress_percent"),
            result=result_model,
        )

    def get_result(self, job_id: str) -> Optional[OptimizeResponse]:
        data = self._read_json(self._result_path(job_id))
        if data is None:
            return None
        return OptimizeResponse.model_validate(data)

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("JSON read 실패 (%s): %s", path, e)
            return None
