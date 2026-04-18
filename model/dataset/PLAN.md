# CC 영화 앙상블+Gemini pseudo-labeled 데이터셋 + MoodEQ 학습 (확정 plan)

## Context

- **프로젝트**: MoodEQ 학술 프로젝트 — 영화 장면 자동 EQ. Visual + Audio 멀티모달 V/A 회귀 → Mood → EQ 프리셋.
- **영구 제약**: CogniMuse · LIRIS-ACCEDE 영구 불가 가정. LIRIS는 EULA + 기관 이메일 필수(gmail 거부) 확인 (2026-04).
- **VEATIC 검증**: 로컬에 있으나 122/124 영상에 오디오 트랙 없음 → audio branch 독립 학습 불가.
- **2026-04-18 재조사 완료**: "영화+audio+video+continuous V/A+즉시 다운" 완벽 조합은 존재하지 않음. 가장 근접한 후보로 **Emo-FilM** (2025 Nature Scientific Data) 발견 — CC0 license, 14편 CC 필름 리스트 확보됨.
- **결정된 경로**: **CC-licensed 영화 원본을 수집 + 다계층 자동 라벨링(앙상블 + Gemini 2.5 Pro + human adjudication)으로 paired multimodal V/A 데이터셋 직접 구축 + 기존 train_cog의 feature-level fusion + gate 학습**.

- **Phase 1 실제 진행 상황 (2026-04-18)**:
  - 8편 CC 영화 로컬 확보 (Blender Open Movies + Archive.org), 총 1,109 windows 분할 완료 (전부 오디오 포함)
  - Emo-FilM 14편 consensus annotation(ds004872) S3 직접 다운 + 50 컬럼 → V/A 파생 CSV 14개 생성 완료
  - 매칭 영화 3편(Big Buck Bunny/Sintel/Tears of Steel)에 1Hz V/A GT 라벨 있음
  - 미확보: Emo-FilM 비-Blender 11편 원본 URL (Vimeo/LIRIS 계열) — best effort 병행 탐색

- **방법론 원칙 (평가 신뢰성 보장용)**:
  1. **Film-level split**: window-level random split 금지. 영화 단위로 train/val/test 분할 후 test film 내부에서만 gold sample.
  2. **V/A distance 분포 재분석**: Layer 1 앙상블 결과로 25/50/75/90 percentile 재계산. CogniMuse/LIRIS 기존 임계값 재사용 금지.
  3. **CCC 분리 보고**: `CCC_pseudo`(val_films pseudo-label)와 `CCC_human`(test_gold human annotation) 분리. `CCC_human`이 Primary, gap > 0.25면 overfit 경보.

## Phase 0 리서치 결과 (2026-04-18 확정)

### A. 모델 가용성 — 전부 확인

| 컴포넌트 | 상태 | URL / 세부 |
|---|---|---|
| Essentia DEAM V/A (musicnn, vggish) | ✅ | https://essentia.upf.edu/models.html . 2-dim regression [1,9], TF `.pb`, CC BY-NC-SA 4.0 |
| Essentia emoMusic V/A | ✅ | 동일 출처 `/emomusic/` |
| Essentia MuSe V/A | ✅ | 동일 출처 `/muse/` |
| EmoNet (Toisoul 2021) | ✅ | https://github.com/face-analysis/emonet . face-based 한계 |
| VEATIC pretrained (ViT) | ✅ | https://github.com/AlbusPeter/VEATIC . Google Drive weight |
| CLIP zero-shot V/A | ✅ | OpenAI CLIP + prompt-probability 매핑 |
| **Gemini 2.5 Pro** | ✅ | Video 258 tok/sec, context caching 90% 할인. 본 프로젝트 전체 라벨링 예상 **<$50** |

### B. Claude API 대체 평가 — **구조적 부적합**

- Claude API (2026-04) 지원: **이미지·PDF·텍스트**. **비디오·오디오 native 미지원**.
- Claude로 대체 시: per-window 프레임 추출 후 이미지로 전송 (4 frames × 1200 tokens), 오디오는 별도 파이프라인
- 비용: Sonnet 4.6 ≈ $144 (video만), Opus 4.7 ≈ $720+ (video만). Gemini 대비 3-15배.
- **결론**: Gemini 2.5 Pro 유지.

### C. CC 영화 인벤토리 확정

**Emo-FilM 논문(Morgenroth et al. 2025 Nature Scientific Data)이 확인해준 CC-licensed 14편**:
After The Rain, Between Viewings, Big Buck Bunny, Chatter, First Bite, Lesson Learned, Payload, Sintel, Spaceman, Superhero, Tears Of Steel, The Secret Number, To Claire From Sonny, You Again.

추가 가능 (중복 배제):
- Blender Open Movies: Elephants Dream, Cosmos Laundromat, Spring, Agent 327, Hero, Caminandes 1/2/3
- Vimeo CC-BY 단편, Archive.org Prelinger 선별

**추정 최종 코퍼스**: 17-25편, **약 3-4시간**, **~3,000 windows @4s**. 규모는 CogniMuse 7편보다 큼, LIRIS-ACCEDE Continuous(30편)보다 작음. 학술적으로 충분.

### D. Emo-FilM의 부가 가치와 한계

- **한계**: V/A 직접 주석 없음(Component Process Model 기반 49 items). 직접 V/A training set 불가.
- **가치**:
  1. **검증된 CC 필름 리스트 제공** → 수집 단계 가속
  2. 1 Hz 연속 emotion component 주석 → **보조 라벨 / cross-check / regularization** 소스
  3. Emo-FilM annotations는 `OpenNeuro ds004872` (CC0) 에서 직접 다운 가능

## 설계 결정 (Phase 0 반영 확정)

### 1. CC 영화 코퍼스
- **Core 14편**: Emo-FilM 리스트 그대로
- **Extension**: Blender 추가 단편 + 선별 Vimeo/Archive.org
- **Target scale**: ~3K windows (4s stride)

### 2. Layer 1 앙상블 (6 모델 유지)
- **Audio 3**: Essentia DEAM-musicnn, emoMusic-musicnn, MuSe-musicnn (musicnn backbone 통일)
- **Visual 3**: EmoNet (face), VEATIC ViT (context), CLIP zero-shot
- **Calibration**: 각 모델을 [-1, 1] V/A로 매핑 (validation split에서 스케일 학습)
- **Aggregation**: `ensemble_V = mean`, `ensemble_A = mean`, `ensemble_std = std`

### 3. Layer 2 — Gemini 2.5 Pro
- Input: 영화 전체 업로드 (File API + context cache) + window 구간 표시
- Output schema: `{"valence": float, "arousal": float, "confidence": float, "reasoning": str}` (V/A ∈ [-1, 1])
- Prompt engineering: 10개 샘플로 품질 검증 후 확정
- Temperature 0, seed 고정, 3 repeat 평균 (안정성)
- 예상 비용: <$50

### 4. Layer 3 — Human adjudication
- Agreement: `|ensemble_V − gemini_V| < 0.2 AND |ensemble_A − gemini_A| < 0.2` → 자동 채택 (weighted avg)
- Disagreement: ~200 clips → human queue
- Test set: 별도 무작위 200 clips → human gold standard
- High uncertainty: `ensemble_std > 0.3` OR `gemini_confidence < 0.5` → 학습 제외
- 평가 도구: 2D V/A slider 웹 UI
- 평가자 ≥2명, Krippendorff's α ≥ 0.7 목표
- **Emo-FilM component annotations cross-check**: Emo-FilM 14편에 대해서는 기존 49-item 주석과 대조하여 추가 QA

### 5. Fusion 아키텍처 — 기존 AutoEQModelCog 유지
- Paired multimodal 확보 → feature-level fusion + gate 학습 가능
- `AutoEQModelCog`, `combined_loss_cog`, `TrainerCog` 변경 없음

## 변경 파일

### 신규 생성

| 파일 | 역할 |
|---|---|
| `scripts/cc_movies/film_list.py` | Emo-FilM 14편 + Blender 추가 소스 URL 리스트 |
| `scripts/cc_movies/download_films.py` | yt-dlp / wget 기반 CC 영화 일괄 다운 |
| `scripts/cc_movies/extract_windows.py` | ffmpeg로 영화 → 4s window MP4 + WAV 분할 |
| `scripts/cc_movies/download_emo_film_annotations.py` | OpenNeuro ds004872 다운 + BIDS → CSV 변환 |
| `model/autoEQ/pseudo_label/layer1_essentia.py` | Essentia 3(audio) 앙상블 (TF subprocess) |
| `model/autoEQ/pseudo_label/layer1_visual.py` | EmoNet + VEATIC ViT + CLIP 앙상블 |
| `model/autoEQ/pseudo_label/layer1_aggregate.py` | 6 모델 calibration + 집계 |
| `model/autoEQ/pseudo_label/layer2_gemini.py` | Gemini 2.5 Pro 호출 (File API 캐싱, batch) |
| `model/autoEQ/pseudo_label/layer3_adjudicate.py` | Agreement/disagreement 분기 + 가중평균 |
| `model/autoEQ/pseudo_label/emo_film_crosscheck.py` | Emo-FilM 49-item → V/A 파생 + 우리 라벨과 correlation QA |
| `model/autoEQ/pseudo_label/analyze_va_distance.py` | V/A scatter + pairwise distance 25/50/75/90 percentile 재계산, 4분면 class balance, CogniMuse 기존값 대비 리포트 |
| `scripts/cc_movies/make_film_split.py` | 영화 단위 train/val/test split 생성 (4분면 stratified, seed 고정, `film_split.json` 출력) |
| `model/autoEQ/pseudo_label/human_ui/` | 2D V/A slider Streamlit UI |
| `model/autoEQ/pseudo_label/build_dataset.py` | 최종 train/disagreement/test split |
| `model/autoEQ/train_cog/ccmovies_preprocess.py` | window → X-CLIP + PANNs feature + metadata |
| `model/autoEQ/train_cog/run_train_ccmovies.py` | AutoEQModelCog 학습 |
| `model/autoEQ/train_cog/tests/test_pseudo_label_pipeline.py` | Layer unit + integration 테스트 |

### 최소 수정

| 파일 | 수정 |
|---|---|
| `train_cog/config.py` | `ccmovies_dir`, `pseudo_label_confidence_threshold`, `pseudo_label_weight_mode`, `emo_film_annotation_dir` 필드 추가 |
| `train_cog/dataset.py` | `PrecomputedCogDataset`는 `split_name="ccmovies"`로 그대로 재사용 |

### 보존 (무수정)
- CogniMuse 전용 파일 전부 (`cognimuse_preprocess.py`, `run_train.py`, `run_lomo.py`, etc.)
- `AutoEQModelCog`, `combined_loss_cog`, `TrainerCog`

## 실행 순서

### Phase 1: CC 영화 + Emo-FilM 주석 수집 (3-5일)
1. Emo-FilM 14편 원본 URL 탐색 (Blender Studio + LIRIS 원본 확보 가능한 것들)
2. Blender 추가 단편 다운
3. `scripts/cc_movies/download_films.py` 실행 → `dataset/autoEQ/CCMovies/films/*.mp4`
4. 4s window 분할 → `windows/{movie_id}_{win_idx}.mp4, .wav`
5. OpenNeuro ds004872 Emo-FilM 주석 다운 → 49-item 1Hz 파싱 → window별 aggregate

### Phase 2: Layer 1 앙상블 (1주)
1. Essentia 3 모델 `.pb` 다운 + TF inference 래퍼 (subprocess 격리)
2. EmoNet weight + VEATIC ViT weight 다운 + PyTorch inference
3. 각 window × 6 모델 inference → raw V/A 행렬
4. Calibration: validation split에서 각 모델 평균·분산 정규화 → [-1, 1]
5. 집계 테이블 생성
6. **V/A 분포·거리 분석** (`analyze_va_distance.py`):
   - Layer 1 ensemble V/A의 전체 scatter + per-film density
   - pairwise V/A Euclidean distance 히스토그램 + **25/50/75/90 percentile 재계산**
   - 4분면(HVHA/HVLA/LVHA/LVLA) window 수 + class balance (min_mood_class_pct)
   - CogniMuse/LIRIS 기존 percentile 병기 → "재사용 금지" 판정 근거 기록
   - 결과는 Phase 3 Gemini prompt 분포 힌트 + Phase 5 negative sampling 임계값 재설정용

### Phase 3: Layer 2 Gemini (3-5일)
1. Google AI Studio 프로젝트 세팅, API 키 `.env` 관리 (사용자 제공)
2. Prompt + JSON schema 확정, 10개 샘플 품질 검증
3. 영화별 File API 업로드 + context cache 생성
4. Batch query (3 repeat 평균)
5. 실시간 비용 모니터링

### Phase 4: Layer 3 분기 + Human Adjudication + Film-level split (1-2주)
1. **Film-level split 먼저** (data leakage 방지 — window-level random split 금지):
   - 확보된 영화 17-25편을 **70/15/15 (train/val/test) 영화 단위 분할**
   - Stratify 기준: Layer 1 ensemble V/A 기반 **4분면 stratified** — test films가 HVHA/HVLA/LVHA/LVLA 각 사분면 최소 1편 포함
   - 결정 결과를 `film_split.json`에 고정 (seed 포함)
2. Disagreement 필터링 (agreement 자동 채택 / disagreement 큐 / high uncertainty 제외)
3. Streamlit V/A slider UI 구축
4. **Gold test set**: **test films 내부에서만** V/A quadrant stratified 200 clips 추출 (전체 무작위 금지)
5. **Disagreement queue**: train+val films에서 disagreement 약 200 clips
6. 평가자 2-3명 모집/셀프 → 총 ~400 clips 주석
7. Krippendorff's α 계산, 0.7 미만 시 가이드 재정비
8. Emo-FilM 14편 매칭 영화에 대해서는 component→V/A 파생값과 cross-check
9. 최종 데이터셋 구조 확정:
   - `train`: train films × all windows (pseudo-label, confidence≥0.7)
   - `val`: val films × all windows (pseudo-label, monitor용)
   - `test_gold`: test films × 200 clips (human annotation)
   - `test_pseudo`: test films × 나머지 windows (pseudo vs model 비교용)

### Phase 5: MoodEQ 학습 (1주)
1. `ccmovies_preprocess.py` → feature .pt (train/val/test films 분리된 상태로)
2. Phase 0 distribution gate (`analyze_ccmovies_distribution.py`, 기존 CogniMuse 버전 복제)
3. Negative sampling 임계값을 Phase 2 V/A distance 분석 결과로 **재설정** (CogniMuse/LIRIS 기본값 덮어쓰기)
4. `run_train_ccmovies.py` → AutoEQModelCog 30 epoch
5. **두 가지 CCC 분리 평가** (순환 평가 방지):
   - **CCC_pseudo**: val_films의 pseudo-label vs 모델 예측 → 학습 sanity check
   - **CCC_human**: test_gold 200 clips human annotation vs 모델 예측 → **Primary 일반화 지표**
   - gap 진단: `CCC_pseudo ≫ CCC_human` (gap > 0.25) → pseudo-label overfit 경보

## Verification

- **Phase 1**: 14+ CC 영화 다운 완료, windows ≥ 2,500 생성, Emo-FilM 주석 정상 파싱
- **Phase 2**: 6 모델 모두 NaN 없이 inference, calibration 후 각 모델 V/A std ≤ 0.3, V/A distance 분포 리포트 생성 + min_mood_class_pct ≥ 1%
- **Phase 3**: Gemini JSON 파싱 100% 성공, confidence 평균 ≥ 0.6, 총 비용 < $100
- **Phase 4**: Film-level split 확정 (train/val/test films 중복 0), test films가 4분면 stratified coverage, Krippendorff's α ≥ 0.7 on test_gold, disagreement rate ≤ 30%, Emo-FilM cross-check correlation ≥ 0.5
- **Phase 5**: 
  - **CCC_human ≥ 0.25** on test_gold (Primary — 영화 V/A SOTA 0.3-0.5 범위, 작은 gold set 감안)
  - **CCC_pseudo ≥ 0.40** on val_films pseudo-label (Sanity check)
  - `CCC_pseudo − CCC_human ≤ 0.25` (overfit gap 상한)

## 리스크와 완화

| 리스크 | 완화 |
|---|---|
| CC 영화 scale < 2K windows | Vimeo/Archive.org 추가 탐색 + 필요 시 4s stride를 2s overlap로 조정해 window 수 배증 |
| EmoNet face 커버리지 구멍 | face 미검출 시 EmoNet 가중치 0, ensemble std 증가로 자동 disagreement 유도 |
| Essentia TF 통합 복잡 | subprocess + `tensorflow` 별도 conda env |
| Gemini API 비용 폭증 | per-batch cap, context cache hit rate ≥90% 모니터링 |
| Human 품질 편차 | 2+명 + 3rd 중재자. 가이드라인 + pilot 50 clips |
| Pseudo-label noise | confidence ≥ 0.7 샘플만 학습, 나머지는 low weight |
| **Test data leakage** (same-film frames in train+test) | Phase 4에서 **film-level split 먼저** → test films 내부에서만 gold clip 추출. Window-level random split 금지. |
| **순환 평가** (pseudo-label로 모델 평가) | CCC_pseudo(sanity) + CCC_human(test_gold, Primary) 분리 보고. gap > 0.25면 overfit 경보. |
| **V/A 분포 이질성** (CogniMuse 기준 임계값 미스매치) | Phase 2 `analyze_va_distance.py` 결과로 negative sampling percentile 재계산. 기존 값 강제 재사용 금지. |
| Emo-FilM 영상 원본 못 구함 | Blender 3편(Big Buck Bunny, Sintel, Tears of Steel)만 확보해도 ~35분 = ~500 windows. 나머지 11편은 best effort |

## 참고 URL

- Emo-FilM 논문: https://www.nature.com/articles/s41597-025-04803-5
- Emo-FilM annotations: https://openneuro.org/datasets/ds004872 (CC0)
- Emo-FilM fMRI: https://openneuro.org/datasets/ds004892 (stimuli 포함 안 됨)
- Emo-FilM code: https://github.com/MIPLabCH/Emo-FilM
- Essentia models: https://essentia.upf.edu/models.html
- EmoNet: https://github.com/face-analysis/emonet
- VEATIC: https://github.com/AlbusPeter/VEATIC
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini context caching: https://ai.google.dev/gemini-api/docs/caching
- Blender Open Movies: https://studio.blender.org/films/
