# Mood-EQ Worker (B 역할) — Ubuntu 진행 가이드

영상 분위기 자동 EQ 적용 시스템의 추론/EQ 워커.

---

## 디렉토리 구조

설치 후 프로젝트는 다음 구조가 되어야 합니다:

```
project_root/
├── backend/                            # 이미 있음 (FastAPI)
├── mobile/                             # 이미 있음 (React Native)
├── model/autoEQ/
│   ├── train/                          # A의 학습 코드 (변경 X)
│   ├── inference/                      # ⭐ B의 워커 (이 패키지)
│   │   ├── paths.py                    # 모든 경로 관리
│   │   ├── utils.py                    # ffprobe 헬퍼
│   │   ├── scene_splitter.py           # 씬 분할 + 전환 분류
│   │   ├── audio_extractor.py          # ffmpeg 오디오 추출
│   │   ├── audio_analyzer.py           # PANNs + Silero VAD + density
│   │   ├── eq_engine.py                # 10밴드 EQ 프리셋 + 블렌딩
│   │   ├── smoothing.py                # EMA + 크로스페이드
│   │   ├── playback.py                 # pedalboard 시간축 EQ
│   │   ├── mux.py                      # 영상-오디오 합성
│   │   ├── scene_labeler.py            # CLIP + PANNs 자동 라벨링
│   │   ├── eq_visualizer.py            # 시각화 4종
│   │   ├── analyzer.py                 # ⭐ 백엔드 진입점
│   │   ├── mushra_generator.py         # webMUSHRA 클립 생성
│   │   ├── mushra_analyzer.py          # 평가 결과 분석
│   │   └── vad_evaluator.py            # VAD F1 측정
│   └── assets/                         # 모델 가중치
│       ├── Cnn14_mAP=0.431.pth         # PANNs (다운로드 필요)
│       └── audioset_tagging_cnn/       # PANNs 코드 (clone 필요)
├── data/trailers/
│   ├── trailer_topgun.mp4              # 다운로드 필요
│   └── trailer_lalaland.mp4            # 다운로드 필요
├── evaluation/                         # ⭐ 청취 평가 (백엔드와 분리)
│   ├── README.md                       # 평가 진행 가이드
│   ├── webmushra/                      # webMUSHRA 별도 클론 (gitignore)
│   └── results/                        # 분석 결과 (요약/그래프)
├── outputs/                            # 산출물 (자동 생성)
└── requirements.txt
```

---

## 1주차 — 환경 세팅 + 분석 파이프라인

### Step 1. 시스템 패키지

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv python3-pip git curl
sudo apt install -y fonts-nanum   # 한글 시각화용
```

### Step 2. 프로젝트 셋업

이미 백엔드/모바일이 있는 프로젝트 루트에서:

```bash
# 1. 디렉토리 생성
mkdir -p model/autoEQ/inference model/autoEQ/assets data/trailers outputs

# 2. inference 모듈 파일들을 model/autoEQ/inference/ 에 복사
#    (이 README와 함께 받은 *.py 파일들)

# 3. requirements.txt를 프로젝트 루트에 두기
```

### Step 3. Python 가상환경

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4. PANNs 자산 다운로드

```bash
# 가중치 (~340MB, Zenodo)
cd model/autoEQ/assets
curl -L -o Cnn14_mAP=0.431.pth \
  "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"

# 코드 레포
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git
cd ../../..  # 프로젝트 루트로 복귀

# 경로 점검
python -m model.autoEQ.inference.paths
```

`PANNS_WEIGHTS exists=True`, `PANNS_CODE exists=True`가 떠야 정상.

### Step 5. 트레일러 다운로드

```bash
# yt-dlp로 다운로드 (URL은 본인이 사용할 트레일러로 교체)
cd data/trailers
yt-dlp -f "bv*[height<=1080]+ba/b" -o "trailer_topgun.%(ext)s" \
  "https://www.youtube.com/watch?v=..."
yt-dlp -f "bv*[height<=1080]+ba/b" -o "trailer_lalaland.%(ext)s" \
  "https://www.youtube.com/watch?v=..."

# mp4가 아니면 ffmpeg로 변환
# ffmpeg -i trailer_topgun.webm -c:v copy -c:a aac trailer_topgun.mp4

cd ../..

# 메타데이터 점검
python -m model.autoEQ.inference.utils data/trailers/trailer_topgun.mp4
python -m model.autoEQ.inference.utils data/trailers/trailer_lalaland.mp4
```

### Step 6. 1주차 파이프라인 실행

```bash
# 6-1. 씬 분할 + 전환 분류 + 짧은 씬 병합
python -m model.autoEQ.inference.scene_splitter
# → outputs/scenes/merged_scenes_{topgun,lalaland}.json

# 6-2. 오디오 추출 (32kHz mono + 원본 품질)
python -m model.autoEQ.inference.audio_extractor
# → data/trailers/audio_{name}_{32k,original}.wav

# 6-3. PANNs + VAD + density
python -m model.autoEQ.inference.audio_analyzer
# → outputs/audio_features/audio_features_{name}.json

# 6-4. EQ 엔진 검증 (V3.1 + V3.2 프리셋 무결성)
python -m model.autoEQ.inference.eq_engine

# 6-5. Smoothing 검증
python -m model.autoEQ.inference.smoothing

# 6-6. Playback 검증 (시간축 EQ + 엣지 케이스)
python -m model.autoEQ.inference.playback
```

각 명령은 검증을 자동 수행하고 ✓ 마크와 함께 진행 상태를 출력합니다.

---

## 2주차 — 자동 라벨링 + 시각화 + 워커 통합

### Step 7. 씬 자동 라벨링 (CLIP + PANNs)

```bash
python -m model.autoEQ.inference.scene_labeler
# → outputs/scene_labels/scene_labels_{name}_auto.json
```

출력에서 `?` 또는 `✗` 마크가 붙은 씬은 수동 검수 권장.

수동 수정 적용:
```python
from model.autoEQ.inference.scene_labeler import apply_manual_corrections
apply_manual_corrections("trailer_topgun", {3: "Cockpit", 7: "Dialogue"})
# → scene_labels_topgun_final.json
```

### Step 8. 시각화 4종

`MANUAL_MOOD_LABELS`를 채운 뒤 (자동 라벨링 결과 기반):

```python
from model.autoEQ.inference.eq_visualizer import visualize_all

MANUAL_MOOD_LABELS = {
    "trailer_topgun": [
        {"start": 0.0, "end": 8.0, "scene_name": "Title",
         "mood": "Power", "prob": 0.75, "density": 0.0},
        # ... (자동 라벨링의 auto_labels_to_manual_template() 결과 사용 가능)
    ],
    "trailer_lalaland": [...],
}

visualize_all(MANUAL_MOOD_LABELS)
# → outputs/eq_viz/eq_{heatmap,timeline,protection,comparison}_*.png
```

발표 핵심: `eq_comparison_*.png` (V3.1 vs V3.2)와 `eq_heatmap_*.png`.

### Step 9. 워커 통합 테스트 (백엔드 없이)

백엔드 없이 워커만 단독 실행:

```bash
# 임시 job 디렉토리 만들기
JOB_ID="test_local"
mkdir -p data/jobs/$JOB_ID
cp data/trailers/trailer_topgun.mp4 data/jobs/$JOB_ID/original.mp4
echo '{"status":"uploaded"}' > data/jobs/$JOB_ID/meta.json

# 환경변수로 jobs 디렉토리 지정
export JOBS_DATA_DIR=$(pwd)/data/jobs

# 워커 실행
python -m model.autoEQ.inference.analyzer $JOB_ID
```

산출물:
- `data/jobs/test_local/timeline.json`
- `data/jobs/test_local/processed.mp4` (V3.1 복사본)
- `data/jobs/test_local/processed_v3_1.mp4`
- `data/jobs/test_local/processed_v3_2.mp4`

---

## 3주차 — 평가

청취 평가는 백엔드와 분리된 `evaluation/` 디렉토리에서 진행합니다. 전체 셋업/실행은 `evaluation/README.md`에 정리되어 있고, 여기서는 순서만 요약합니다.

### Step 10. webMUSHRA 클론 + PHP 설치 (한 번만)

```bash
cd evaluation
git clone https://github.com/audiolabs/webMUSHRA.git webmushra
sudo apt install -y php-cli
cd ..   # 프로젝트 루트로 복귀
```

`evaluation/webmushra/`는 `.gitignore`에 등록되어 본 레포에는 올라가지 않습니다.

### Step 11. 평가 클립 + YAML 자동 생성

```python
# 프로젝트 루트에서
from model.autoEQ.inference.mushra_generator import generate_all_clips

EVALUATION_SET = {
    "trailer_topgun": [
        # (scene_idx, start, end, mood, prob, density)
        (0, 0.0, 8.0, "Power", 0.75, 0.0),       # Title
        (5, 25.3, 38.1, "Tension", 0.80, 0.2),   # Dogfight
        (12, 65.0, 75.5, "Tenderness", 0.70, 0.4),  # Dialogue
        # ... 영상당 4~6개
    ],
    "trailer_lalaland": [
        (8, 35.2, 47.8, "Joyful Activation", 0.80, 0.3),  # Dance
        # ...
    ],
}
generate_all_clips(EVALUATION_SET)
```

자동 출력 위치 (별도 복사 단계 없음):
- 클립: `evaluation/webmushra/configs/resources/audio/<scene_dir>/{original,v3_1,v3_2,anchor}.wav`
- YAML: `evaluation/webmushra/configs/mood_eq_test.yaml`

### Step 12. webMUSHRA 서버 실행

```bash
cd evaluation/webmushra
php -S localhost:8080
```

> 백엔드(8000번)와 충돌 안 나도록 webMUSHRA는 **8080**번 사용. 두 서버 동시 실행 OK.

브라우저: `http://localhost:8080/?config=mood_eq_test.yaml`

### Step 13. 평가 결과 분석

```bash
cd ../..   # 프로젝트 루트
python -m model.autoEQ.inference.mushra_analyzer
# 인자 없으면 evaluation/webmushra/results/ 안의 가장 최근 CSV 자동 분석
```

산출물:
- `evaluation/results/mushra_summary.csv` — 조건별 평균/CI
- `evaluation/results/mushra_bars.png`, `mushra_boxplot.png`
- 콘솔: V3.1 vs V3.2 paired t-test 결과

### Step 14. VAD F1 측정

Audacity로 트레일러 wav를 열고 대사 구간을 라벨링 → Export Labels:

```bash
python -m model.autoEQ.inference.vad_evaluator \
  data/trailers/audio_topgun_32k.wav labels_topgun.txt

# → outputs/vad_eval/vad_metrics.json
# 목표: macro F1 ≥ 0.85
```

---

## 트러블슈팅

### `ModuleNotFoundError: No module named 'model'`
프로젝트 루트에서 `python -m ...`으로 실행해야 합니다. `cd model/autoEQ/inference && python scene_splitter.py` 같은 방식은 상대 import 때문에 동작하지 않습니다.

### `FileNotFoundError: PANNs 가중치 없음`
`model/autoEQ/assets/Cnn14_mAP=0.431.pth`가 있는지 확인. 파일 크기 ~340MB.

### `FileNotFoundError: AudioSet 라벨 메타 없음`
`model/autoEQ/assets/audioset_tagging_cnn/metadata/class_labels_indices.csv`가 있는지 확인. git clone이 정상 완료됐는지.

### 한글 폰트가 깨져서 시각화에 □□□ 표시
```bash
sudo apt install fonts-nanum
rm -rf ~/.cache/matplotlib   # 폰트 캐시 초기화
```

### `cv2 is not defined` 또는 OpenCV 충돌
```bash
pip uninstall opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python
```

### Silero VAD 다운로드 실패
인터넷 연결 확인. 처음 실행 시 ~50MB 모델을 ~/.cache에 받습니다.

### `librosa.load`가 느리거나 에러
```bash
pip install soundfile  # librosa의 백엔드
```

---

## 다음 단계

- **백엔드 통합**: `analyzer.process_job(job_id)`를 백엔드 큐 워커에서 호출하면 끝.
- **A 모델 통합 (Day 10)**: `analyzer.py`의 `dummy_a_model_output()`을 A의 Gate Network 호출로 교체.
- **모바일 연동**: 모바일은 백엔드 API만 호출하므로 워커 변경과 무관.

진행 중 막히면 에러 메시지 + 실행한 명령 + Python 버전을 함께 공유해 주세요.
