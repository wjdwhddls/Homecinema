# config.py — pydantic-settings 기반 환경 변수 관리
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_UPLOAD_SIZE_MB: int = 500
    UPLOAD_TMP_DIR: str = "./tmp/uploads"
    JOBS_DATA_DIR: str = "./data/jobs"
    CORS_ORIGINS: str = "http://localhost:8081,http://localhost:3000"
    ALLOWED_EXTENSIONS: str = "mp4,mov,mkv,avi,webm"
    DEV_FAKE_PROCESSED: bool = False

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()}

    @property
    def upload_tmp_path(self) -> Path:
        return Path(self.UPLOAD_TMP_DIR)

    @property
    def jobs_data_path(self) -> Path:
        return Path(self.JOBS_DATA_DIR)


settings = Settings()
