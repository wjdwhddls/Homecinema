"""paths.py — 프로젝트 전반의 절대 경로 상수.

모든 모듈은 경로가 필요할 때 이 모듈을 import하여 사용합니다.
경로를 바꾸고 싶으면 이 파일만 수정하면 됩니다.
"""

from __future__ import annotations

import os
from pathlib import Path

# ────────────────────────────────────────────────────────
# 프로젝트 루트
# 이 파일 위치: <project_root>/model/autoEQ/inference/paths.py
# parents[3] = <project_root>
# ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ────────────────────────────────────────────────────────
# 모델 자산 (가중치, 외부 레포)
# ────────────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "model" / "autoEQ"
ASSETS_DIR = MODEL_DIR / "assets"

# PANNs
PANNS_WEIGHTS = ASSETS_DIR / "Cnn14_mAP=0.431.pth"
PANNS_CODE_DIR = ASSETS_DIR / "audioset_tagging_cnn"
PANNS_PYTORCH_DIR = PANNS_CODE_DIR / "pytorch"
PANNS_LABELS_CSV = PANNS_CODE_DIR / "metadata" / "class_labels_indices.csv"

# ────────────────────────────────────────────────────────
# 입력 데이터
# ────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
TRAILERS_DIR = DATA_DIR / "trailers"

# ────────────────────────────────────────────────────────
# 산출물 (분석 결과, EQ 적용 영상, MUSHRA 클립 등)
# ────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCENES_DIR = OUTPUTS_DIR / "scenes"
AUDIO_FEATURES_DIR = OUTPUTS_DIR / "audio_features"
EQ_VIZ_DIR = OUTPUTS_DIR / "eq_viz"
SCENE_LABELS_DIR = OUTPUTS_DIR / "scene_labels"
PROCESSED_VIDEOS_DIR = OUTPUTS_DIR / "processed_videos"

# ────────────────────────────────────────────────────────
# 백엔드 jobs (워커 입출력)
# 환경변수 JOBS_DATA_DIR이 있으면 그것을 사용, 없으면 기본값
# ────────────────────────────────────────────────────────
JOBS_DATA_DIR = Path(
    os.environ.get("JOBS_DATA_DIR", str(PROJECT_ROOT / "backend" / "data" / "jobs"))
)

# ────────────────────────────────────────────────────────
# webMUSHRA 평가 (evaluation/ 디렉토리, 백엔드와 분리)
# webMUSHRA 자체는 evaluation/webmushra/ 에 별도 클론 (gitignore됨)
# 클립 wav는 webMUSHRA가 인식하는 표준 위치에 직접 생성
# ────────────────────────────────────────────────────────
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
WEBMUSHRA_DIR = EVALUATION_DIR / "webmushra"
WEBMUSHRA_CONFIG_DIR = WEBMUSHRA_DIR / "configs"
MUSHRA_CLIPS_DIR = WEBMUSHRA_CONFIG_DIR / "resources" / "audio"
EVALUATION_RESULTS_DIR = EVALUATION_DIR / "results"


def ensure_dirs():
    """산출물 디렉토리들을 미리 생성."""
    for d in [
        OUTPUTS_DIR,
        SCENES_DIR,
        AUDIO_FEATURES_DIR,
        EQ_VIZ_DIR,
        SCENE_LABELS_DIR,
        PROCESSED_VIDEOS_DIR,
        EVALUATION_DIR,
        MUSHRA_CLIPS_DIR,
        EVALUATION_RESULTS_DIR,
        TRAILERS_DIR,
        ASSETS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def trailer_path(name: str) -> Path:
    """트레일러 영상 경로. name='topgun' → data/trailers/trailer_topgun.mp4"""
    return TRAILERS_DIR / f"trailer_{name}.mp4"


def audio_path(name: str, kind: str = "32k") -> Path:
    """오디오 추출 결과 경로. kind='32k' or 'original'"""
    return TRAILERS_DIR / f"audio_{name}_{kind}.wav"


def scenes_json(name: str) -> Path:
    return SCENES_DIR / f"merged_scenes_{name}.json"


def audio_features_json(name: str) -> Path:
    return AUDIO_FEATURES_DIR / f"audio_features_{name}.json"


def scene_labels_json(name: str, stage: str = "auto") -> Path:
    """stage='auto' (CLIP 결과) or 'final' (수동 검수 후)"""
    return SCENE_LABELS_DIR / f"scene_labels_{name}_{stage}.json"


def processed_video(name: str, version: str = "v3_1") -> Path:
    return PROCESSED_VIDEOS_DIR / f"processed_{name}_{version}.mp4"


def mushra_clip_dir(scene_dir_name: str) -> Path:
    """씬별 MUSHRA 클립 디렉토리. scene_dir_name = 'trailer_topgun_scene00' 형식."""
    return MUSHRA_CLIPS_DIR / scene_dir_name


# ────────────────────────────────────────────────────────
# 직접 실행 시 경로 점검
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("프로젝트 경로 구조")
    print("=" * 60)
    print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"MODEL_DIR         : {MODEL_DIR}")
    print(f"  ASSETS_DIR      : {ASSETS_DIR}")
    print(f"    PANNS_WEIGHTS : {PANNS_WEIGHTS}  exists={PANNS_WEIGHTS.exists()}")
    print(f"    PANNS_CODE    : {PANNS_CODE_DIR}  exists={PANNS_CODE_DIR.exists()}")
    print(f"DATA_DIR          : {DATA_DIR}")
    print(f"  TRAILERS_DIR    : {TRAILERS_DIR}  exists={TRAILERS_DIR.exists()}")
    print(f"OUTPUTS_DIR       : {OUTPUTS_DIR}  exists={OUTPUTS_DIR.exists()}")
    print(f"JOBS_DATA_DIR     : {JOBS_DATA_DIR}  exists={JOBS_DATA_DIR.exists()}")
    print()
    print("→ ensure_dirs() 호출하여 산출물 디렉토리 생성")
    ensure_dirs()
    print("✓ 산출물 디렉토리 생성 완료")
