# Mood EQ Backend

영화 분위기 자동 EQ 적용 프로젝트의 백엔드 서버입니다. FastAPI 기반으로 영상 파일 업로드, job 상태 관리, 영상 다운로드(원본/EQ 적용본) 기능을 제공합니다. 현재는 ML 분석과 EQ 처리 없이 업로드와 파일 관리만 동작하는 skeleton 버전입니다.

## 사전 요구사항

- Python 3.10 이상
- 설치 확인: `python --version`

## 설치 방법

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 실행 방법

```bash
python main.py
```

서버가 `http://0.0.0.0:8000`에서 실행됩니다. `http://localhost:8000/docs`에서 Swagger UI에 접근할 수 있습니다.

## API 사용 예시 (curl)

### 헬스 체크
```bash
curl http://localhost:8000/api/health
```

### 영상 업로드
```bash
curl -X POST -F "file=@test.mp4" http://localhost:8000/api/upload
```

### Job 상태 조회
```bash
curl http://localhost:8000/api/jobs/<job_id>/status
```

### 원본 다운로드
```bash
curl -o original.mp4 http://localhost:8000/api/jobs/<job_id>/download/original
```

### 처리된 영상 다운로드 (현재는 409, DEV_FAKE_PROCESSED=true 시 원본 반환)
```bash
curl -o processed.mp4 http://localhost:8000/api/jobs/<job_id>/download/processed
```

### Job 삭제
```bash
curl -X DELETE http://localhost:8000/api/jobs/<job_id>
```

## 환경 변수 설명

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 서버 바인딩 호스트 |
| `PORT` | `8000` | 서버 포트 |
| `MAX_UPLOAD_SIZE_MB` | `500` | 최대 업로드 파일 크기 (MB) |
| `UPLOAD_TMP_DIR` | `./tmp/uploads` | 업로드 중 임시 파일 저장 경로 |
| `JOBS_DATA_DIR` | `./data/jobs` | Job 데이터 영구 저장 경로 |
| `CORS_ORIGINS` | `http://localhost:8081,http://localhost:3000` | 허용할 CORS origin 목록 (쉼표 구분). 와일드카드 `*` 사용 금지 |
| `ALLOWED_EXTENSIONS` | `mp4,mov,mkv,avi,webm` | 허용할 영상 확장자 (쉼표 구분) |
| `DEV_FAKE_PROCESSED` | `false` | `true` 설정 시 `/download/processed`가 원본을 반환. **개발/테스트 전용, 운영 배포 시 반드시 `false`** |

## 폴더 구조 설명

```
backend/
├── main.py              # FastAPI 앱 진입점
├── config.py            # pydantic-settings 환경 변수 관리
├── api/
│   ├── health.py        # GET /api/health
│   ├── upload.py        # POST /api/upload
│   └── jobs.py          # job 상태/타임라인/다운로드/삭제
├── core/
│   └── storage.py       # job 디렉토리 관리 헬퍼 함수
├── models/
│   └── schemas.py       # API 응답 Pydantic 스키마
├── tmp/uploads/         # 업로드 중 임시 파일. 완료 시 자동 이동.
└── data/jobs/{job_id}/  # Job 단위 영구 저장소
    ├── original.{ext}   # 업로드 완료 시 생성
    ├── meta.json        # 상태와 메타데이터
    ├── timeline.json    # 분석 완료 후 (현재는 없음)
    └── processed.mp4    # EQ 처리 완료 후 (현재는 없음)
```

## 자주 발생하는 문제

- **포트 8000 충돌**: `.env`의 `PORT` 값을 다른 포트로 변경하세요.
- **파일 업로드 실패**: `tmp/uploads/`, `data/jobs/` 폴더 권한을 확인하세요.
- **`pydantic-settings` 설치 오류**: pydantic 2.x 버전이 설치되어 있는지 확인하세요.
- **`/download/processed`가 409 반환**: 정상 동작입니다. 분석 기능이 아직 통합되지 않은 상태입니다. 테스트용으로 원본을 반환받고 싶다면 `.env`의 `DEV_FAKE_PROCESSED=true`로 설정하세요.
