# Mood-EQ 청취 평가 (webMUSHRA)

이 디렉토리는 본 프로젝트의 **청취 평가**에만 사용됩니다. 백엔드/모바일과 완전히 분리되어 있어, 평가자가 자기 컴퓨터에서 독립적으로 띄울 수 있습니다.

---

## 디렉토리 구조

```
evaluation/
├── README.md                      ← 지금 이 파일
├── .gitignore                     ← webmushra/ 와 raw 결과 제외
├── webmushra/                     ⭐ 별도 클론 (gitignore됨)
│   ├── configs/
│   │   ├── mood_eq_test.yaml      ← 워커가 자동 생성
│   │   └── resources/audio/
│   │       ├── trailer_topgun_scene00/
│   │       │   ├── original.wav
│   │       │   ├── v3_1.wav
│   │       │   ├── v3_2.wav
│   │       │   └── anchor.wav
│   │       └── trailer_topgun_scene05/...
│   └── results/                   ← 평가 결과 자동 저장
└── results/                       ← 분석 결과 (요약 CSV, 그래프) 보관
```

---

## 1단계 — webMUSHRA 클론 (한 번만)

```bash
cd evaluation
git clone https://github.com/audiolabs/webMUSHRA.git webmushra
```

`webmushra/`는 `.gitignore`에 등록되어 있어 본 프로젝트 git에는 올라가지 않습니다.

## 2단계 — PHP 설치

webMUSHRA는 결과 저장을 위해 PHP가 필요합니다.

```bash
sudo apt install -y php-cli
php --version    # 8.x 권장
```

## 3단계 — 평가 클립 생성

`evaluation/`이 아니라 **프로젝트 루트**에서 워커를 실행해 클립을 만듭니다. 클립은 자동으로 `evaluation/webmushra/configs/resources/audio/` 안에 떨어지고, YAML도 `evaluation/webmushra/configs/mood_eq_test.yaml` 위치에 자동 저장됩니다.

```bash
cd ..    # 프로젝트 루트
source venv/bin/activate

python -c "
from model.autoEQ.inference.mushra_generator import generate_all_clips

EVALUATION_SET = {
    'trailer_topgun': [
        # (scene_idx, start, end, mood, prob, density)
        (0, 0.0, 8.0, 'Power', 0.75, 0.0),
        (5, 25.3, 38.1, 'Tension', 0.80, 0.2),
        (12, 65.0, 75.5, 'Tenderness', 0.70, 0.4),
        # 4~6개 권장
    ],
    'trailer_lalaland': [
        (8, 35.2, 47.8, 'Joyful Activation', 0.80, 0.3),
        # 4~6개
    ],
}
generate_all_clips(EVALUATION_SET)
"
```

각 씬마다 4개 wav (원본/V3.1/V3.2/Anchor)가 생성됩니다. 총 8~12씬이면 32~48개 클립.

## 4단계 — webMUSHRA 서버 실행

```bash
cd evaluation/webmushra
php -S localhost:8080
```

> **포트 주의**: 백엔드(8000번)와 분리하기 위해 webMUSHRA는 **8080**번을 사용합니다. 두 서버를 동시에 띄워도 충돌 없음.

## 5단계 — 브라우저에서 평가

```
http://localhost:8080/?config=mood_eq_test.yaml
```

평가 흐름:
1. 첫 페이지 안내 읽기
2. 각 씬마다 4개 슬라이더(원본/V3.1/V3.2/Anchor)를 0~100점으로 평가
   - **평가 기준: "영상의 분위기와 가장 잘 맞는다고 느낀 정도"**
   - 클립을 자유롭게 반복 재생 가능
   - 같은 슬라이더에 여러 번 점수 변경 OK
3. 모든 씬 완료 → 결과 자동 저장

## 6단계 — 결과 분석

웹 평가가 끝나면 `evaluation/webmushra/results/mood_eq_v1/`에 평가자별 CSV가 저장됩니다.

```bash
cd ../..   # 프로젝트 루트로
python -m model.autoEQ.inference.mushra_analyzer
# → 가장 최근 CSV 자동 분석
# → evaluation/results/mushra_{bars,boxplot}.png
# → evaluation/results/mushra_summary.csv
# → V3.1 vs V3.2 paired t-test 출력
```

특정 CSV를 지정하고 싶으면:
```bash
python -m model.autoEQ.inference.mushra_analyzer \
  evaluation/webmushra/results/mood_eq_v1/<session>/results.csv
```

---

## 평가자에게 안내하는 법

외부 평가자(친구, 동료, 청취 실험 참가자)에게 평가를 부탁할 때:

**Option A. 본인 컴퓨터에서 직접 띄우기** (가장 간단)
- 위 1~5단계를 본인이 진행
- 평가자에게 본인 컴퓨터 앞에 앉혀서 평가
- 헤드폰 권장 (스피커는 룸 음향 영향)

**Option B. 평가자가 자기 컴퓨터에서 띄우기**
- 본 레포 클론 → 위 단계 그대로 진행
- 단점: PHP 설치 부담, Python 환경 필요

**Option C. 호스팅** (대규모 평가용)
- `evaluation/webmushra/` 디렉토리를 임의 PHP 호스팅(저렴한 VPS, Heroku 등)에 업로드
- 평가자에게 URL 공유
- webMUSHRA 결과는 서버에 자동 저장

학사논문/발표 정도 규모라면 **Option A가 가장 합리적**입니다 (5~10명, 본인 입회 하 30분).

---

## 평가 시 주의사항

- **헤드폰 권장**: 스피커는 룸 음향이 EQ 차이를 가려서 평가가 어려움
- **음량 통일**: 첫 씬 시작 전 본인 시스템 볼륨을 한 번 정해두고 평가 중 변경 X
- **순서 무작위화**: webMUSHRA는 stimuli 순서를 자동 랜덤화 (조건 위치 편향 제거)
- **Anchor 점수 확인**: 모든 평가자의 Anchor 평균이 50점 이상이면 → 평가가 무작위였거나 Anchor 설계 문제. 분석 단계에서 `mushra_analyzer`가 자동 경고
- **휴식**: 10씬 이상이면 중간에 5분 휴식 (청각 피로)

---

## 트러블슈팅

### "404 Not Found" 오디오 파일
브라우저 개발자도구(F12) → Network 탭에서 어느 파일이 404인지 확인. `evaluation/webmushra/configs/resources/audio/<scene_dir>/v3_1.wav` 같은 경로가 실제로 있는지 점검. 없으면 3단계 다시 실행.

### "Cannot connect to localhost:8080"
PHP 서버가 안 켜졌거나 다른 포트를 쓰는 중. `lsof -i :8080`으로 포트 점유 확인.

### 결과가 results/에 안 생김
webMUSHRA는 `service/write.php`를 통해 결과를 저장하는데 PHP 서버 없이 정적 파일 서버(예: `python -m http.server`)로 띄우면 결과 저장이 동작하지 않음. **반드시 `php -S` 사용**.

### Anchor가 너무 비슷하게 들림
`mushra_generator.py`의 `make_anchor_eq()`는 저역(B1~B3)에 -6dB cut을 적용. 더 명확한 차이가 필요하면 -10dB로 늘리거나 고역도 같이 cut.
