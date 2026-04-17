# Mood-EQ webMUSHRA 시연 — 리허설 체크리스트 & 플레이북

생성일: 2026-04-17

## 사전 준비 (강사 오기 전, 5~10분)

### 1. PHP 서버 띄우기 — 별도 터미널 열어 유지

**Git Bash에서 `php`가 PATH에 있는지부터 확인:**

```bash
php -v
```

**Case A — `php -v` 정상 출력**:

```bash
cd /d/Homecinema/evaluation/webmushra
php -S localhost:8080 -t .
```

**Case B — `command not found` (현재 로컬 상태 ← 이쪽)**:

PATH 등록이 안 된 상태라 winget 설치 경로를 직접 사용.

```bash
cd /d/Homecinema/evaluation/webmushra
/c/Users/yun72_92xubzr/AppData/Local/Microsoft/WinGet/Packages/PHP.PHP.8.3_Microsoft.Winget.Source_8wekyb3d8bbwe/php.exe -S localhost:8080 -t .
```

서버 기동 시 출력 예시:
```
[Fri Apr 17 12:42:26 2026] PHP 8.3.30 Development Server (http://localhost:8080) started
```

### 2. 브라우저 접속 (Chrome 권장)

```
http://localhost:8080/?config=mood_eq_demo.yaml
```

환영 페이지(`<h1>Mood-EQ 청취 평가</h1>`)가 뜨면 OK.

### 3. 결과 저장 경로 사전 확인

평가 제출 시 자동 생성되는 위치:

```
/d/Homecinema/evaluation/webmushra/results/mood_eq_v1/mushra.csv
```

(`mood_eq_v1`은 YAML의 `testId`에서 온 이름. `sessionXX.xml`도 같이 생길 수 있음.)

---

## 리허설 (본인이 혼자 10분)

### 기본 동작
- [ ] `http://localhost:8080/?config=mood_eq_demo.yaml` 접속 시 환영 페이지 로딩
- [ ] "Next" 버튼으로 1번째 trial(탑건) 페이지 진행

### 1번째 trial — 탑건 (2분 30초)
- [ ] Reference 재생 버튼 동작 (파형 표시)
- [ ] 4개 stimuli(`original_hidden`, `v3_1`, `v3_2`, `anchor`) 재생 버튼 동작
- [ ] anchor 재생 시 저음이 확연히 빠진 소리 들림 (정상)
- [ ] 슬라이더 0~100 움직임
- [ ] 모든 stimulus를 한 번 이상 재생하지 않으면 Next 비활성화 (MUSHRA 표준)
- [ ] 점수 매기고 Next 클릭 → 2번째 trial로

### 2번째 trial — 라라랜드 (2분 10초)
- [ ] 동일 흐름 동작
- [ ] Next 클릭 → 완료 페이지

### 종료 확인
- [ ] "평가 완료" 화면 나옴
- [ ] `/d/Homecinema/evaluation/webmushra/results/mood_eq_v1/mushra.csv` 파일 생성됨
- [ ] CSV 열어서 `trial_id, rating_stimulus, rating_score` 컬럼에 점수 기록 확인
- [ ] F12 콘솔에서 `POST /service/write.php` 200 응답 확인

---

## 시연 플레이북 (강사 앞에서)

### 오프닝 (1분)

> "Mood-EQ는 영상 분위기를 자동으로 분석해서 EQ를 동적으로 조정하는 시스템입니다.
> 오늘은 webMUSHRA라는 청취 평가 도구로 직접 들어보시고 평가해주시면 됩니다.
> 트레일러 두 개(탑건, 라라랜드)를 들으실 텐데, 각각 원본과 두 가지 처리본,
> 그리고 한 개의 참고용 품질 손상 버전(anchor) 총 4개를
> **어느 게 어느 버전인지 모르는 상태(blind)** 로 비교 평가합니다.
> 이어폰 사용 권장합니다."

### 평가 진행 (5~10분)

- 노트북 화면 공유 또는 강사가 직접 브라우저 조작
- 각 trial마다 reference 1번 + 4 stimuli 청취 (각 2~2.5분)
- 슬라이더로 점수 (0~100, 100이 reference와 가장 유사)
- 자연스럽게: "평가 진행 중에는 말 걸지 않을게요"

### 마무리 (1분)

> "결과는 자동 저장됐습니다. 오늘 강사님 점수 기준으로
> V3.1(기본 EQ) vs V3.2(대사 보호 추가) 선호도를 확인해보고,
> Day 11~14에 5~10명 정식 평가에 반영할 예정입니다."

---

## 예상 질문 대응

### Q1. "왜 트레일러 통째로? 짧은 클립으로 비교하는 게 아닌가?"

> 정식 MUSHRA는 보통 10초 이내 짧은 클립을 사용합니다. 오늘은 비공식 데모이고
> 강사님께 자연스러운 흐름으로 들려드리고 싶어서 트레일러 전체로 진행했습니다.
> Day 11~14 정식 평가에서는 씬 단위로 진행할 예정입니다.

### Q2. "원본이 가장 좋다고 점수 주면 시스템이 의미 없는 거 아닌가?"

> MUSHRA에서 원본(hidden reference)은 **항상 100점이 정답**입니다.
> 진짜 비교는 V3.1 vs V3.2 사이에서 어느 게 원본에 더 가깝게 들리는지,
> 또는 어느 게 청취 경험을 향상시키는지를 보는 거예요.

### Q3. "씬 라벨링이 부정확하다고 했는데 평가에 영향 없나?"

> 라벨 부정확성이 EQ 적용을 다른 카테고리로 보내긴 하지만,
> 각 카테고리 EQ 자체는 음향 심리학 근거(명세서 6장)에 기반해서 설계됐습니다.
> '잘못된 카테고리에 대한 의도된 EQ'가 적용되는 거라
> 원본 vs 처리본 비교는 여전히 의미 있습니다.
> 카테고리 정확도는 Day 10 A 모델 학습 완료 후 해결됩니다.

### Q4. "평가 인원 1~2명인데 통계적 의미가 있나?"

> 오늘은 파일럿 데모입니다. 정식 평가는 Day 11~14에 5~10명 평가자로
> 진행하고 paired t-test로 통계 유의성 검증합니다.

### Q5. "Anchor가 뭔가요? 4개 비교군 중 하나가 이상하게 들립니다"

> Anchor는 MUSHRA 표준의 sanity check 자극입니다.
> 의도적으로 품질을 손상시킨 버전(저역 -6dB cut)이고,
> 평가자가 진지하게 듣고 있다면 **가장 낮은 점수**를 받아야 합니다.
> 평가자가 anchor에 100점 같은 점수를 주면 그 평가는
> 통계 분석에서 제외됩니다 (post-screening).

---

## 트러블슈팅

### 페이지 404
- URL의 `?config=` 경로 확인: `?config=mood_eq_demo.yaml` (configs/ 접두사 없음)
- 대안: `?config=configs/mood_eq_demo.yaml`

### `php: command not found`
- PATH 등록 안 됐음 → 본 문서 "사전 준비 1. Case B" 참조

### 오디오 재생 안 됨
- 브라우저 오디오 권한 확인 (Chrome > 사이트 설정 > 소리)
- F12 콘솔에서 `configs/resources/audio/...` 404 에러 있는지

### 결과 저장 실패
- `service/write.php` 200 응답 확인 (F12 Network 탭)
- `results/mood_eq_v1/` 디렉토리 쓰기 권한
- Windows Defender/백신이 php.exe 쓰기 차단하는 경우 있음 → 예외 등록

### Next 버튼 안 눌림
- **모든 stimulus**를 끝까지 청취해야 활성화 (MUSHRA 표준)
- 슬라이더를 한 번 이상 움직여야 활성화

### 포트 8080 점유 중
- `netstat -ano | findstr :8080` 로 PID 확인 후 `taskkill /F /PID <PID>`
- 또는 `php -S localhost:8081 -t .` 로 다른 포트 사용 (URL도 8081로)

---

## 백업 시연 자료 (webMUSHRA 전체 실패 시)

- `outputs/demo/videos/` 영상 6개 (원본/V3.1/V3.2 × 2영상) → VLC로 직접 재생
- `outputs/demo/images/` 시각화 14개 → 슬라이드로 설명
- `outputs/demo/README_demo.md` → 3-act 시연 대본 그대로 활용
