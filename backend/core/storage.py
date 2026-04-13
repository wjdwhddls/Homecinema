# core/storage.py — job 디렉토리 관리 헬퍼
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

from config import settings


def get_job_dir(job_id: str) -> Path:
    """data/jobs/{job_id}/ 경로 반환 (생성은 하지 않음)"""
    return settings.jobs_data_path / job_id


def create_job_dir(job_id: str) -> Path:
    """data/jobs/{job_id}/ 디렉토리 생성 후 경로 반환"""
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def get_tmp_upload_path(job_id: str, ext: str) -> Path:
    """tmp/uploads/{job_id}.partial 경로 반환"""
    return settings.upload_tmp_path / f"{job_id}.partial"


async def finalize_upload(
    job_id: str, tmp_path: Path, original_filename: str, ext: str, size_bytes: int
) -> Path:
    """tmp 파일을 data/jobs/{job_id}/original.{ext}로 이동 + meta.json 생성"""
    job_dir = create_job_dir(job_id)
    dest = job_dir / f"original.{ext}"

    # 임시 파일을 job 디렉토리로 이동
    shutil.move(str(tmp_path), str(dest))

    # meta.json 생성
    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "job_id": job_id,
        "status": "uploaded",
        "created_at": now,
        "updated_at": now,
        "original_filename": original_filename,
        "original_ext": ext,
        "original_size_bytes": size_bytes,
        "processed_size_bytes": None,
        "analysis_progress": 0.0,
        "error_message": None,
    }
    save_job_meta(job_id, meta)

    return dest


def load_job_meta(job_id: str) -> dict | None:
    """meta.json 읽어서 dict 반환, 없으면 None"""
    meta_path = get_job_dir(job_id) / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_job_meta(job_id: str, meta: dict) -> None:
    """meta.json 저장 (updated_at 자동 갱신)"""
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta_path = get_job_dir(job_id) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def job_exists(job_id: str) -> bool:
    """data/jobs/{job_id}/ 디렉토리 + meta.json 존재 여부 확인"""
    job_dir = get_job_dir(job_id)
    return job_dir.exists() and (job_dir / "meta.json").exists()


def load_timeline(job_id: str) -> dict | None:
    """timeline.json 읽어서 dict 반환, 없으면 None"""
    timeline_path = get_job_dir(job_id) / "timeline.json"
    if not timeline_path.exists():
        return None
    with open(timeline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_original_video_path(job_id: str) -> Path | None:
    """original.{ext} 경로 반환. 없으면 None. 확장자는 meta.json에서 조회."""
    meta = load_job_meta(job_id)
    if meta is None:
        return None
    ext = meta.get("original_ext", "mp4")
    video_path = get_job_dir(job_id) / f"original.{ext}"
    if video_path.exists():
        return video_path
    return None


def get_processed_video_path(job_id: str) -> Path | None:
    """processed.mp4 경로 반환. 없으면 None."""
    video_path = get_job_dir(job_id) / "processed.mp4"
    if video_path.exists():
        return video_path
    return None


def delete_job(job_id: str) -> bool:
    """data/jobs/{job_id}/ 전체 삭제. 성공 시 True, 없으면 False."""
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        return False
    shutil.rmtree(str(job_dir))
    return True
