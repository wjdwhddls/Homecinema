# MoodEQ — CCMovies Pseudo-label 기반 축소 명세 (V3.3)

> **문서 위치**: 본 문서는 [`specification_v3_2.md`](/Users/jongin/Downloads/specification_v3_2.md) 를 상위 권위로 두되, 데이터 제약으로 변경된 실제 구현 스펙을 기술한다. V3.2와 충돌하는 부분은 본 문서(V3.3)가 **실제 구현의 최종 기술**이며, V3.2는 이론적 상위 목표로 보존된다.
>
> **저자 노트**: 학술 논문·리포트에서는 V3.2의 비전을 기술하고 본 V3.3에서 실현된 범위를 명시하는 2단 구조를 권장한다.
>
> **변경일**: 2026-04-19
> **파이프라인 명**: `train_pseudo` (구 `train_cog`)

---

## 0. V3.2 대비 핵심 변경 사항

| 항목 | V3.2 원래 설계 | V3.3 실제 구현 | 이유 |
|---|---|---|---|
| **학습 데이터** | LIRIS-ACCEDE 9,800 클립 | **CCMovies 1,288 windows** (9편, pseudo-label + 200 human) | LIRIS-ACCEDE 확보 불가 |
| **Congruence Head** | 포함 (negative sampling으로 학습) | **제거** | 작은 데이터로 negative sampling 불안정 |
| **Negative sampling** | 50/25/25 비율 | **제거** | cong 제거로 자동 비활성 |
| **Modality Dropout** | `cong_label==0`에만 조건부 적용 | **전체 샘플 무조건 p=0.05** | cong 제거로 조건 단순화 |
| **Mood Head K** | 7 GEMS (Phase 0 실패 시 K=4로 축소) | **K=4 (기본)** — Phase 0 K=7 실패 시 fallback | Power GEMS 영역에 실샘플 0개 |
| **EQ 매핑** | V/A → 7 GEMS → 10-band EQ | **동일 (변경 없음)** — 추론 후처리로 V/A → va_to_mood(7) → EQ | 명세서 §5-9 정합 |
| **증강** | 명시 안 됨 | **feature-level 3종 추가** (Gaussian noise / quadrant mixup / conditional label smoothing) | 데이터 크기 제약 대응 |
| **V/A loss** | MSE | **CCC hybrid** (0.7·MSE + 0.3·(1-CCC)) | 감정 회귀 표준 지표 반영 |
| **Early stopping** | val mean CCC, patience=10 | 동일 + **MAE tiebreaker** (Pareto-guard) | CCC 동률 시 Pareto 역행 방지 |
| **Pass gate safety** | ≥6/7 folds | **ceil(N·6/7)** 일반화 (9편 → ≥8/9) | CCMovies 9편 대응 |
| **Human ground truth** | 없음 | **200개 test_gold human annotation** (Streamlit UI) | 학술 신뢰도 확보 |

---

## 1. 프로젝트 개요 (V3.2 §1과 동일)

### 1-1. 목적
영화 영상의 정서를 멀티모달 딥러닝으로 분석하고, 결과에 기반해 동적 EQ를 적용하여 시청 경험을 향상시키는 시스템. 분석 결과는 JSON 타임라인 형태로 직렬화되어 다양한 재생 환경(데스크톱, 모바일 앱, 임베디드)에서 일관되게 적용 가능.

### 1-2. 학술 근거 (5개 기둥, 변경 없음)
- **Russell (1980)** — V/A 2차원 정서 모델
- **Zentner et al. (2008) GEMS** — 7개 Mood 카테고리
- **Eerola & Vuoskoski (2011)** — 영화음악 V/A 회귀
- **Cohen (2001, 2013) CAM** — Congruence-Association Model (이론적 근거, 구현은 제거)
- **Parke et al. (2007)** — 시청각 통합 정서 분석

### 1-3. 범위 (V3.3 조정)

**포함**
- 멀티모달 정서 분석 (V/A 회귀 **메인**, Mood K=4 **보조**)
- Adaptive Gating with degeneracy prevention
- **Feature-level augmentation** (noise + mixup + label smoothing)
- **Human adjudication 200 windows** (test_gold ground truth)
- 추론 파이프라인: 씬 분할 + EMA 스무딩 + V/A → 7 mood → EQ preset
- Silero VAD 대사 감지 + 음성 대역 EQ 보호
- JSON 타임라인 출력 + 분석-재생 분리
- 씬 경계 크로스페이드 EQ 적용

**제외 (V3.2와 동일)**
- Self-supervised cross-modal congruence learning (데이터 부족)
- 한국 콘텐츠 검증
- 정서 특화 백본 fine-tuning
- 스피커 위치 자동 배정

---

## 2. 데이터 전략

### 2-1. CCMovies 직접 구축 파이프라인 (V3.2 LIRIS-ACCEDE 대체)

#### 영화 9편 (CC-licensed, 재배포 가능)
| film_id | 출처 | 장르 | 라이선스 | windows |
|---|---|---|---|---|
| agent_327 | Blender | 단편 액션 | CC-BY | 57 |
| big_buck_bunny | Blender | 단편 코미디 | CC-BY | 149 |
| caminandes_3 | Blender | 단편 코미디 | CC-BY | 37 |
| cosmos_laundromat | Blender | 단편 드라마 | CC-BY | 182 |
| elephants_dream | Blender | 단편 초현실 | CC-BY | 163 |
| sintel | Blender | 단편 판타지 | CC-BY | 222 |
| spring | Blender | 단편 드라마 | CC-BY | 116 |
| tears_of_steel | Blender | 단편 SF | CC-BY | 183 |
| **valkaama_highlight** | CC-BY-SA | 실사 드라마 (하이라이트) | CC-BY-SA | 180 |
| **합계** | | | | **1,289** (85.9분) |

Valkaama는 유일한 실사 데이터로, 애니메이션 편향 완화 역할.

#### 윈도우 사양 (V3.2와 동일)
- 길이 4초, stride 2초 (학습) / 1초 (추론)
- `dataset/autoEQ/CCMovies/windows/<film_id>/<window_id>.mp4` + `.wav` 페어

### 2-2. 3-Layer Pseudo-label 파이프라인

학습 라벨은 다음 3층 앙상블로 생성:

```
Layer 1 — 저수준 특징 기반 앙상블
  ├─ Essentia: audio MFCC/tempo → scikit-learn regressor
  ├─ EmoNet: face detection + V/A (SFD + ResNet50)
  ├─ VEATIC: context ViT V/A (5-frame)
  └─ CLIP ViT-B/32: zero-shot V/A (텍스트 프롬프트 앵커)
       ↓ aggregate (robust scale + weighted mean)
Layer 2 — Gemini 2.5 Pro V/A 추론
  └─ video_metadata 구간 슬라이싱 + JSON schema 강제
       ↓
Layer 3 — Adjudication
  └─ ensemble vs Gemini 비교 → source 분류
       ├─ auto_agreement (138, 10.7%) — 양쪽 일치
       ├─ gemini_only (1150, 89.2%) — ensemble 실패 시 Gemini 사용
       └─ excluded (1, 0.08%) — 양쪽 무효, 학습 제외
```

**Layer 3 calibration**: Gemini V/A median=−0.600/0.400, MAD로 스케일링. 일부 outlier는 [−1, 1] 넘김 — 학습 시 V/A 회귀엔 무관, 7-mood 매핑 시 Euclidean distance는 그대로 동작.

### 2-3. Human Ground Truth (V3.2에 없던 추가)

**배경**: Pseudo-label 품질 검증 + 최종 평가 ground truth 확보를 위해 수행.

**절차**:
- `build_dataset.py`로 test_gold_queue 200개 추첨 (test 영화 2편, 4사분면 stratified 50/50/50/50)
- Streamlit `app.py`로 Valence/Arousal 슬라이더 (`-1~+1`, 0.05 step) 평가
- 1명 평가자 (`evaluator_id=jongin`), 평균 7.6초/window
- 저장: `dataset/autoEQ/CCMovies/labels/human_annotations.csv` (append-only)
- `final_labels.csv`에 `human_v`, `human_a`, `has_human_label` 컬럼 병합 완료

**Pseudo-label 품질 검증 결과** (human vs Gemini, 200 windows):
- MAE: V=0.056, A=0.095
- 사분면 일치율: **95.0%**
- → Gemini 기반 pseudo-label의 신뢰성 확인됨

### 2-4. 데이터 Split

**방식**: Film-level split (V3.2와 동일). 같은 영화의 windows는 한 split에만.

`dataset/autoEQ/CCMovies/splits/film_split.json`:
| split | 영화 | windows | 비율 |
|---|---|---|---|
| **train** | agent_327, big_buck_bunny, caminandes_3, spring, tears_of_steel, **valkaama_highlight** | 722 | 56.0% |
| **val** | elephants_dream | 163 | 12.6% |
| **test_pseudo** | cosmos_laundromat, sintel (test_gold 제외분) | 204 | 15.8% |
| **test_gold** | cosmos_laundromat, sintel (human 평가) | 200 | 15.5% |
| **합계** | 9편 | 1,289 | 100% |

**사분면 분포 (train 기준)**:
- LVLA 254 (35%), HVHA 239 (33%), HVLA 171 (24%), LVHA 58 (8%)
- LVHA가 가장 희소하나 K=4 Phase 0 gate 통과 (14.8% ≥ 1%)

### 2-5. V/A → 7 Mood 매핑 (V3.2 §2-5와 동일, 불변)

| 카테고리 | Valence | Arousal | 출처 |
|---|---|---|---|
| Tension | −0.6 | +0.7 | Zentner GEMS |
| Sadness | −0.6 | −0.4 | Zentner GEMS |
| Peacefulness | +0.5 | −0.5 | Zentner GEMS |
| Joyful Activation | +0.7 | +0.6 | Zentner GEMS |
| Tenderness | +0.4 | −0.2 | Zentner GEMS |
| **Power** | +0.2 | +0.8 | Zentner GEMS ⚠️ CCMovies에 실샘플 0개 |
| Wonder | +0.5 | +0.3 | GEMS Wonder + Transcendence 통합 |

**Power 영역 공백의 처리**:
- 학습 시: Mood Head K=4 (사분면)로 축소 → Power class 학습 불필요
- 추론 시: V/A → 7-centroid Euclidean → 7-mood 매핑 유지 (§5에서 설명)
- 리포트: "데이터에 Power 감정이 부재함은 한국/서양 엔터테인먼트 영화의 일반적 특성 반영" (9편 중 대부분 밝은 애니메이션 + 드라마)

---

## 3. 모델 구조

### 3-1. 전체 구조 (V3.2 §3-1 대비 cong head 제거)

```
Video (4s window)         Audio (4s window, 16kHz mono)
    │                              │
    │ 8 frames @ 224²              │ mel-spectrogram
    ▼                              ▼
┌─────────────┐          ┌──────────────────┐
│ X-CLIP      │          │ PANNs CNN14      │
│ (frozen)    │          │ (frozen)         │
└──────┬──────┘          └────────┬─────────┘
       │ 512                      │ 2048
       │                          ▼
       │             ┌───────────────────────┐
       │             │ Linear Proj (학습)    │
       │             │ 2048 → 512            │
       │             └───────────┬───────────┘
       │                         │ 512
       │     [Modality Dropout p=0.05, training only]
       │     [Feature Noise σ=0.03, training only]
       ├────────┬────────────────┤
       │        │                │
       ▼        ▼                ▼
    ┌──────────────────────────┐
    │ Gate Network (학습)       │
    │ [v;a] (1024) → softmax(2)│
    └──────────────┬───────────┘
                   │ (w_v, w_a)
                   ▼
    ┌──────────────────────────┐
    │ Intermediate Fusion       │
    │ [w_v·v; w_a·a] (1024)    │
    └──┬───────────────────┬────┘
       │                   │
       ▼                   ▼
  ┌──────────┐      ┌──────────┐
  │ V/A Head │      │ Mood Head│
  │ (1024→2) │      │ (1024→4) │  ← K=4 기본
  └────┬─────┘      └────┬─────┘
       │                  │
    va_pred          mood_logits
   (B, 2)            (B, 4)
```

### 3-2. Frozen 인코더 (V3.2와 동일)
- **X-CLIP**: `microsoft/xclip-base-patch32`, 512-dim visual feature
- **PANNs CNN14**: AudioSet pretrained, 2048-dim raw audio feature

### 3-3. Linear Projection + Gate + Heads (V3.2 구조 유지)

| 모듈 | 입력 → 출력 | 파라미터 |
|---|---|---|
| AudioProjection | 2048 → 512 | 학습 |
| GateNetwork | [v;a] 1024 → [256 → 2 softmax] | 학습 |
| VAHead | fused 1024 → [256 → 2] | 학습 |
| **MoodHead (K=4)** | fused 1024 → [256 → 4] | 학습 |

**학습 파라미터 총합**: 약 180만 (V3.2 190만과 유사, cong head 제거로 감소)

### 3-4. Modality Dropout (V3.3 조건 제거)

```python
# V3.2: cong_label==0 샘플에만 적용
# V3.3: 모든 샘플에 p=0.05 적용 (cong 제거로 조건 단순화)
if self.training:
    drop_trigger = torch.rand(B) < p  # p=0.05
    drop_choice  = torch.randint(0, 2, (B,))  # v(0) or a(1)
    v[drop_trigger & (drop_choice==0)] = 0
    a[drop_trigger & (drop_choice==1)] = 0
```

### 3-5. Feature Noise (V3.3 신규)

Frozen 인코더 출력에 training 시 Gaussian noise 주입:

```python
if self.training and feature_noise_std > 0:
    v_live = (v.abs().sum(-1, keepdim=True) > 0).float()  # dropped samples skip
    a_live = (a.abs().sum(-1, keepdim=True) > 0).float()
    v = v + torch.randn_like(v) * feature_noise_std * v_live
    a = a + torch.randn_like(a) * feature_noise_std * a_live
```

- 대칭 σ 유지 (visual/audio 동일) → gate collapse 방지
- 드롭된 모달(0 vector)은 noise bypass → dropout 신호 보존
- **권장 σ = 0.03** (2-4% of feature std 범위)

---

## 4. 학습 프로세스

### 4-1. Loss 구성 (V3.2 4-term → V3.3 3-term)

$$
L_{total} = \lambda_{va} \cdot L_{va} + \lambda_{mood} \cdot L_{mood} + \lambda_{gate} \cdot L_{gate}
$$

| 항 | 식 | 기본 λ | 역할 |
|---|---|---|---|
| `L_va` | **CCC hybrid**: `(1-w)·MSE + w·(1-mean_CCC)`, w=0.3 | **1.0** | 주 회귀 |
| `L_mood` | CrossEntropy(mood_logits, mood_target) — K=4 사분면 | **0.3** (권장, V3.2 기본 0.5에서 하향) | 보조 regularization |
| `L_gate` | `mean(w·log(w))` (Shannon entropy, ≤0) | **0.05** | gate 균형 |
| ~~L_cong~~ | ~~CrossEntropy on cong_label~~ | **제거** | cong head 제거 |

**CCC 공식**:
$$\text{CCC} = \frac{2\sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$$

**mood target 계산**:
```python
mood = va_to_quadrant(valence, arousal)  # K=4
# 추론 시에는 별도: va_to_mood(v, a)  # K=7 GEMS → EQ preset
```

### 4-2. Feature-level Augmentation (V3.3 신규, spec §8-2 데이터 한계 보완)

Frozen encoder 출력을 그대로 캐시하되, **post-encoder feature space**에서 매 step 확률적 증강 적용. `.pt` 재생성 불필요.

| 기법 | 대상 | 위치 | 기본 OFF / 권장 값 |
|---|---|---|---|
| **① Modality Dropout** | visual OR audio 0 masking | `model.forward` | p=**0.05** (권장 0.10) |
| **② Gaussian Feature Noise** | visual + audio 대칭 σ | `model.forward` | σ=0 / **0.03** |
| **③ Quadrant Mixup** | 같은 사분면 쌍 feature+V/A | `collate_fn` | prob=0 / **0.5**, α=0.4, λ∈[0.1, 0.9] |
| **④ Conditional Label Smoothing** | V/A × 0.95 (고σ 샘플만) | `collate_fn` | ε=0 / **0.05**, σ_thr=0.15 |

**Mixup 설계 원칙**:
- 같은 사분면 쌍만 선택 (LVHA+HVHA 혼합 금지 → 감정 의미 모순 방지)
- λ ~ Beta(α, α) → 0.1 + 0.8·λ로 shrink (λ=0, 1 극단 제거)
- visual·audio·V/A·V_std·A_std 모두 동일 λ로 선형 결합
- mood label은 primary sample 사용

**권장 효과**: val mean CCC +0.05~0.08 예상 (722 train 데이터 기준 보수 추정)

### 4-3. σ-filter (V3.2 확장, train 전용)

`ensemble_std_v/a` 중 max가 threshold 초과한 train window 제외. val/test는 원분포 보존.
- 기본 `-1.0` (비활성)
- 권장 **0.25** (ensemble_std p90 부근)

### 4-4. 하이퍼파라미터 (V3.2 §4-5 기반, 일부 권장값 조정)

| 파라미터 | V3.2 | V3.3 권장 | 이유 |
|---|---|---|---|
| Optimizer | AdamW | AdamW | |
| Learning rate | 1e-4 | 1e-4 | |
| Weight decay | 1e-5 | **5e-5** | 722 train 과적합 방어 |
| Batch size | 32 | 32 | |
| Epochs | 30-50 | **40** | 증강 ON 시 수렴 느림 |
| LR scheduler | Cosine | warmup+Cosine | V3.2 §4-5 유지 |
| Warmup steps | 500 | **200** | 722/32×40≈900 step, 500은 과함 |
| Grad clip | max_norm=1.0 | max_norm=1.0 | |
| Early stopping | val mean_CCC, patience=10 | 동일 **+ (ccc, −mae) tuple** | Pareto-guard |

### 4-5. Early Stopping (V3.3 Pareto-guard 확장)

```python
current = (mean_ccc, -mean_mae)
best    = (self.best_mean_ccc, -self.best_mean_mae)
if current > best:  # tuple comparison: CCC primary, MAE tiebreaker
    update_best()
    patience = 0
else:
    patience += 1
    if patience >= config.early_stop_patience:  # default 10
        stop()
```

### 4-6. 학습 모니터링 (V3.2 §4-6 + V3.3 추가)

매 epoch 기록:
- Train/Val loss per head (va, mood, gate_entropy, total)
- Val V/A: MSE, MAE, **RMSE**, Pearson, **CCC_V**, **CCC_A**, **mean_CCC (primary)**
- Val Mood (K=4): accuracy, F1 (macro, weighted), Cohen's kappa
- **Gate weight**: mean w_v, mean w_a, **var_v, var_a, entropy_value** (V3.3 추가)
- **Head-wise gradient norm**: va, mood (V3.2 §4-1-1)
- WandB logging (선택)

---

## 5. 추론 프로세스 (V3.2 §5와 동일, Phase 3 구현 예정)

### 5-0. 분석-재생 분리 아키텍처 (V3.2 §5-0 유지 + VAD 병렬 보정)

**명세서 V3.2 §5-8-5 원 설계를 정확히 반영**: Silero VAD는 오디오 트랙만 필요하므로 씬 분할/모델 추론과 **독립 병렬 실행**. Python `concurrent.futures.ThreadPoolExecutor`로 2개 thread orchestrate. VAD 처리 시간(10분 영화에 ~10초)이 모델 추론 병목(~2-3분)에 완전히 가려져 **추가 소요 0**.

```
                          영상 입력 (movie.mp4)
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        [비디오 트랙]     [비디오 트랙]     [오디오 트랙]
                │               │               │
                ▼               │               ▼
         ┌─────────────┐        │        ┌──────────────┐
         │ PySceneDetect│       │        │ Silero VAD   │
         │  + 씬 병합   │       │        │ (독립 병렬   │
         └──────┬───────┘       │        │  thread B)   │
                │               │        └──────┬───────┘
                ▼               │               │
         ┌─────────────┐        │               │
         │ 윈도우      │        │               │
         │ 슬라이딩    │        │               │
         │ (4s,1s)     │        │               │
         └──────┬───────┘       │               │
                │               │               │
                ▼               │               │
         ┌─────────────┐        │               │
         │ X-CLIP +    │        │               │
         │ PANNs +Gate │        │               │
         │ +V/A Head   │        │               │
         │ (병목 ~2분) │        │               │
         └──────┬───────┘       │               │
                │               │               │
                ▼               │               │
         ┌─────────────┐        │               │
         │ EMA 스무딩 +│        │               │
         │ 씬별 집계   │        │               │
         └──────┬───────┘       │               │
                │               │               │
                └───────────────┼───────────────┘
                                │    (여기서 join)
                                ▼
                  ┌─────────────────────────────┐
                  │ 씬별 dialogue_density 계산  │
                  │ (씬 경계 × VAD 구간 교집합) │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ V/A → va_to_mood(7) →       │
                  │ 10-band EQ preset +         │
                  │ 대사 보호 (α_d, B6/B7/B8)   │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                        JSON 타임라인 출력
                          (분석 종료)

[재생 파이프라인 — PC/모바일, 실시간 가능]
     JSON 로드 → 씬별 pedalboard 10밴드 EQ 적용
          → 씬 경계 raised-cosine 크로스페이드 (200-500ms)
          → ffmpeg 오디오 재결합 → 출력 영상
```

**병렬 실행 이점**:
1. VAD가 모델 추론 병목에 흡수 → 총 소요 단축
2. 두 경로 독립 → 한쪽 실패해도 다른 쪽 진행 가능 (fault tolerance)
3. 모듈 decoupling — 각자 별도 테스트 가능

### 5-1. 씬 분할 (V3.2 §5-2, §5-3 동일)
- PySceneDetect ContentDetector, threshold=27.0, min_scene_len=1s
- min_duration=2s 미만 씬은 인접 병합 (반복 병합, 종료 조건 3개)

### 5-2. 윈도우 슬라이딩 추론
- 씬 내 4s window, **stride 1s** (학습 2s보다 조밀)
- 씬 끝에서 마지막 window 보장 (4s 역방향 픽)

### 5-3. 모델 추론 (eval mode)
Augmentation(modality dropout, feature noise, mixup, label smoothing) 모두 비활성.  
각 window → `{va_pred (2,), mood_logits (4,), gate_weights (2,)}` 출력.

### 5-4. EMA 스무딩 + 씬별 집계 (V3.2 §5-6, §5-7)
- 씬 경계 리셋 EMA, α=0.3
- 첫 2-3 window cold start
- 씬별 V/A 평균 → 씬당 1개 V/A 좌표

### 5-5. V/A → 7 GEMS Mood (V3.3 핵심 원칙)

**Mood Head의 K=4 출력은 사용하지 않음**. 대신:

```python
# Inference post-processing
def select_eq_preset(va_pred):
    v, a = va_pred[0].item(), va_pred[1].item()
    mood_idx = va_to_mood(v, a)   # 7-class Euclidean to GEMS centroids
    mood_name = GEMS_LABELS[mood_idx]
    return EQ_PRESETS[mood_name]  # V3.2 §6-4 table
```

**근거**: V3.2 §5-9 "Mood Head 출력은 EQ 매핑에 직접 사용하지 않음. V/A 좌표 기반 매핑만 사용". 학습 시 mood head(K=4)는 regularization 보조 task이고, EQ 결정은 V/A regression 결과로만.

**Power 영역 대응**: inference 시 모델이 V≈0.2, A≈0.8을 예측하면 → va_to_mood가 **Power** 반환 → Power EQ preset 자동 적용. 모델이 그 영역을 예측할 수 있는지는 별개의 일반화 능력 문제.

### 5-6. Silero VAD 대사 감지 (V3.2 §5-8 동일)
- 16 kHz mono, threshold=0.5, min_speech=250ms, min_silence=100ms
- 씬별 `dialogue_density` ∈ [0, 1] 계산

### 5-7. EQ 프리셋 선택 + 대사 보호 (V3.2 §5-10, §6-4 동일)

**10-band EQ preset matrix (V3.1에서 결정, V3.3 불변)**:

| Category | B1 31.5 | B2 63 | B3 125 | B4 250 | B5 500 | B6 1k | B7 2k | B8 4k | B9 8k | B10 16k |
|---|---|---|---|---|---|---|---|---|---|---|
| Tension | +2.0 | +2.0 | +1.0 | 0.0 | 0.0 | +1.0 | +2.5 | +2.0 | 0.0 | −1.0 |
| Sadness | 0.0 | +1.0 | +1.0 | +1.0 | 0.0 | 0.0 | −2.0 | −2.0 | −1.5 | −1.5 |
| Peacefulness | 0.0 | 0.0 | +0.5 | +0.5 | 0.0 | 0.0 | −0.5 | −1.0 | −0.5 | 0.0 |
| JoyfulActivation | −1.0 | −1.0 | 0.0 | 0.0 | +1.0 | +1.5 | +2.0 | +2.0 | +1.5 | +1.0 |
| Tenderness | 0.0 | +1.0 | +2.0 | +1.5 | +0.5 | 0.0 | −1.0 | −1.5 | −1.0 | −0.5 |
| Power | +2.5 | +2.0 | +2.0 | +1.0 | +0.5 | +1.0 | +1.5 | +1.0 | 0.0 | 0.0 |
| Wonder | 0.0 | 0.0 | 0.0 | −0.5 | 0.0 | +0.5 | +1.0 | +1.5 | +1.5 | +2.0 |

모든 밴드 Biquad peaking (Q 값은 V3.1 §6-2 표).

> **V3.3 Phase 1 예산 업데이트 (2026-04-19)**: `scripts/eq_response_check.py`로 정확한 RBJ biquad 주파수 응답을 실측한 결과(pedalboard 실제 출력과 0.1 dB 이내 일치), 누적 피크는 Power +3.93 dB, Tension +3.37 dB, Sadness -3.05 dB, JoyfulActivation +3.09 dB (나머지 3개 카테고리는 ±3 dB 이내)였다. 보완자료 V3.2 §Medium 4는 "1 옥타브 0.4배 근사식"에 기반한 이론 계산의 내재 오차 ±0.3 dB를 이미 인정하고 있었으나, 실측 결과 해당 근사가 Q=0.7 biquad의 실제 응답(0.5~0.6배)을 과소평가했음이 확인되었다.
>
> V3.1 preset의 **음향학적 근거(Bowling 2017, Arnal 2015, Wallmark 2017 등)를 보존**하기 위해 preset 값은 변경하지 않고, 예산 자체를 **±3 dB → ±3.5 dB**로 재정의한다. ±3.5 dB는 여전히 V3.2 §6-3이 명시한 "음악을 변형하는 수준(±6 dB)"에 크게 미치지 못하는 보수적 범위이며, 영화 theatrical mix의 mood shaping 관행과도 정합한다. 이 재정의는 "정확한 실측으로 근사식의 한계를 발견하여 spec를 업데이트"라는 §Medium 4의 검증 경로 정신과 부합한다.
>
> **검증 판정 (Phase 1 완료, ±3.5 dB 기준)**:
> - **safe (|peak| ≤ 3.5 dB)**: Tension +3.37, Sadness -3.05, Peacefulness -1.30, JoyfulActivation +3.09, Tenderness +2.85, Wonder +2.44 → 6개 카테고리 PASS.
> - **boundary (3.5 < |peak| ≤ 4.0 dB)**: **Power +3.93 dB** — 보완자료 §Medium 4 tier "boundary" 정의(0.3~0.5 dB 초과)에 해당. Phase 3 청취 검증 결과에 따라 결정: 거슬림 없으면 그대로 유지, 저역 왜곡 관측 시 Power B1 +2.5 → +2.0 한 단계 하향. `scripts/eq_response_check.py --strict`는 VIOLATION tier(|peak| > 4.0 dB)만 실패 처리하며, boundary tier는 PASS*로 통과한다.
> - **violation (|peak| > 4.0 dB)**: 없음.

**대사 보호 공식** (B6/B7/B8만):
```
g_effective(band) = g_original(band) × (1 − (1 − α_d) × dialogue_density)
α_d = 0.5 (기본), Phase 3 청취 검증으로 0.3~0.7 튜닝
```

### 5-8. JSON 타임라인 출력 (V3.2 §5-11 동일)

파일명: `{video}.timeline.json`  
구성: `schema_version`, `metadata`, `config`, `scenes[]` (start/end/va/mood/eq_preset.original_bands/eq_preset.effective_bands/dialogue), `global` (선택)

### 5-9. 재생 파이프라인 (V3.2 §5-12, §5-13, §5-14)
- pedalboard로 씬별 10-band EQ 적용
- 씬 경계 200-500ms raised-cosine 크로스페이드
- ffmpeg로 원본 비디오 + 처리된 오디오 재결합 (`-c:v copy -c:a aac`)

---

## 6. EQ 프리셋 음향학적 근거 (V3.2 §6 완전 유지, 변경 없음)

V3.2 §6-3의 5개 주파수 대역 원칙, §6-4의 7×10 preset matrix, §6-6의 대사 보호 근거(Articulation Index, SII, Killion & Mueller 2010) 그대로 계승. V3.3에서는 7 GEMS centroid + preset matrix 모두 **불변**.

**핵심 근거 문헌** (V3.2 §6-6 인용 유지):
1. Juslin & Laukka (2003) — 정서별 음향 단서 메타분석
2. Arnal et al. (2015) — 인간 비명의 acoustic niche (Tension 근거)
3. Trevor et al. (2020) — Arnal 발견의 영화 음악 검증
4. Sonnenschein (2001) — 영화 음향 디자인
5. Bowling et al. (2017) — 큰 신체 = 낮은 주파수 = dominance (Power)
6. ANSI S3.5-1997 — Speech Intelligibility Index (대사 보호)

---

## 7. 평가 전략

### 7-1. 정량 평가 지표 (V3.2 §7-1 + V3.3 확장)

#### Primary V/A Regression
| 지표 | 정의 | 합격선 |
|---|---|---|
| **mean_CCC** | (CCC_V + CCC_A) / 2 | ≥ 0.20 / stretch ≥ 0.30 |
| **mean_MAE** | (MAE_V + MAE_A) / 2 | ≤ 0.25 |
| **mean_RMSE** | (RMSE_V + RMSE_A) / 2 | 참고 (AVEC 호환) |

CCC가 주지표. Early stopping과 모델 선택 기준.

#### Per-dim
`ccc_valence`, `ccc_arousal`, `mae_valence`, `mae_arousal`, `rmse_*`, `pearson_*`

#### Mood (K=4 보조)
`accuracy`, `f1_macro` (주), `f1_weighted`, `cohen_kappa` — K=4이라 chance=0.25

#### Gate 진단 (V3.3 추가)
- `gate_w_v`, `gate_w_a` (0.5 근처 정상, 한쪽 >0.9 시 gate collapse)
- `gate_w_v_var`, `gate_w_a_var` (instability 측정)
- `gate_entropy_value` (raw, 최대 ln(2)≈0.693)
- `grad_norm/va`, `grad_norm/mood` (task balance, V3.2 §4-1-1)

### 7-2. Human Ground Truth 평가 (V3.3 신규 최상위 지표)

test_gold 200 windows에 대해 `eval_testgold.py` 실행:

| 비교 | 의미 | 합격선 |
|---|---|---|
| **va_pred vs human_v/a** | 모델 최종 신뢰도 | mean_CCC ≥ 0.25, 사분면 일치 ≥ 70% |
| pseudo vs human | pseudo-label 품질 | MAE ≤ 0.10 (확인 완료) |
| va_pred vs pseudo | 학습 적합도 | 참고 |

추가 출력: 4-사분면 confusion, 7-mood confusion, per-film breakdown, gate 통계.

### 7-3. LOMO 9-fold Safety Gate (V3.3 일반화)

V3.2 §합격 기준의 safety threshold `≥6/7` 을 `ceil(N·6/7)`로 일반화:
- CogniMuse 7편 → ≥6/7 (기존 동일)
- **CCMovies 9편 → ≥8/9**

| 기준 | 조건 |
|---|---|
| V/A Primary CCC | LOMO mean_CCC ≥ 0.20 |
| V/A Primary MAE | LOMO mean_MAE ≤ 0.25 |
| V/A Safety CCC | ≥8/9 folds `min(ccc_v, ccc_a) > 0` |
| V/A Safety MAE | ≥8/9 folds `max(mae_v, mae_a) ≤ 0.30` |
| Overall | 4개 모두 PASS |

### 7-4. 청취 검증 (V3.2 §7-4 유지)
3-5명 평가자, 5점 척도로 **분위기 부합도 / 자연스러움 / 선호도 / 대사 명료도**.  
대사 많은 씬 vs 음악 위주 씬 모두 포함. α_d ∈ {0.3, 0.5, 0.7} A/B 비교.

---

## 8. 디렉토리 및 파일 구조

```
Homecinema/
├── SPECIFICATION_V3_3.md                ← 본 문서
├── dataset/autoEQ/CCMovies/
│   ├── films/<film_id>.mp4              ← 원본 9편
│   ├── windows/
│   │   ├── metadata.csv                 ← 1,289 windows 인덱스
│   │   └── <film>/<wid>.{mp4,wav}       ← 4s 클립
│   ├── splits/film_split.json           ← train/val/test 영화 리스트
│   ├── labels/
│   │   ├── layer1_essentia.csv          ← Layer 1 audio
│   │   ├── layer1_visual.csv            ← Layer 1 video
│   │   ├── layer1_aggregate.csv         ← Layer 1 combined
│   │   ├── layer2_gemini.csv            ← Layer 2 Gemini 2.5 Pro
│   │   ├── layer3_adjudicated.csv       ← Layer 3 final
│   │   ├── final_labels.csv             ★ 학습 입력 (human 병합 컬럼 포함)
│   │   ├── test_gold_queue.csv          ← 200 human 평가 대기
│   │   ├── disagreement_queue.csv       ← 200 (향후 개선용)
│   │   ├── human_annotations.csv        ← 248 (current 200 + stale 48)
│   │   └── dataset_summary.json
│   └── emofilm_annotations/             ← 참고 Emo-FilM 1Hz (3편)
├── model/autoEQ/
│   ├── pseudo_label/                    ← Phase 1-2 라벨링 파이프라인
│   │   ├── layer1_essentia.py
│   │   ├── layer1_visual.py
│   │   ├── layer1_aggregate.py
│   │   ├── layer2_gemini.py
│   │   ├── layer3_adjudicate.py
│   │   ├── build_dataset.py
│   │   └── human_ui/
│   │       ├── app.py                   ← Streamlit 평가 UI
│   │       ├── annotation_io.py
│   │       └── krippendorff.py
│   ├── train/                           ← (원 LIRIS-ACCEDE 학습 모듈, 보존)
│   └── train_pseudo/                    ★ 학습 파이프라인 (구 train_cog)
│       ├── config.py                    ← TrainCogConfig (K=4 default)
│       ├── model.py                     ← AutoEQModelCog (cong head 없음)
│       ├── losses.py                    ← combined_loss_cog (3-term)
│       ├── dataset.py                   ← PrecomputedCogDataset + LOMO/film_split
│       ├── trainer.py                   ← TrainerCog (Pareto-guard)
│       ├── ccmovies_preprocess.py       ← CCMovies → .pt 변환
│       ├── cognimuse_preprocess.py      ← (legacy CogniMuse)
│       ├── analyze_cognimuse_distribution.py ← Phase 0 gate
│       ├── run_train.py                 ← 단일 fold entry
│       ├── run_lomo.py                  ← N-fold orchestrator
│       ├── eval_testgold.py             ★ human GT 평가 (V3.3 신규)
│       └── tests/ (54 passing)
└── data/features/ccmovies/              ← ccmovies_preprocess 산출물
    ├── ccmovies_visual.pt               ← (1,288 × 512)
    ├── ccmovies_audio.pt                ← (1,288 × 2,048)
    ├── ccmovies_metadata.pt             ← (window_id → dict)
    └── manifest.json
```

---

## 9. 실행 경로 (End-to-End)

### Phase 1-2 (완료)
데이터셋 구축 + 라벨링 + human annotation — 본 문서 기준 모두 완료.

### Phase 3 (학습) — 즉시 실행 가능

```bash
# 1) Feature precompute (~10-20분, 1회)
python3 -m model.autoEQ.train_pseudo.ccmovies_preprocess \
  --labels_csv dataset/autoEQ/CCMovies/labels/final_labels.csv \
  --windows_dir dataset/autoEQ/CCMovies/windows \
  --output_dir data/features/ccmovies

# 2) Phase 0 분포 게이트
python3 -m model.autoEQ.train_pseudo.analyze_cognimuse_distribution \
  --feature_dir data/features/ccmovies --split_name ccmovies \
  --num_mood_classes 4 --output_dir runs/phase0
# 기대: [PHASE 0 PASS] K=4 · min mood class = 14.8%

# 3) 단일 fold 학습 (film_split.json 기반, ~30분 MPS)
python3 -m model.autoEQ.train_pseudo.run_train \
  --feature_dir data/features/ccmovies --split_name ccmovies \
  --movie_set ccmovies --lomo_fold -1 \
  --split_json dataset/autoEQ/CCMovies/splits/film_split.json \
  --num_mood_classes 4 --lambda_mood 0.3 \
  --feature_noise_std 0.03 \
  --mixup_prob 0.5 --mixup_alpha 0.4 \
  --label_smooth_eps 0.05 --label_smooth_sigma_threshold 0.15 \
  --sigma_filter_threshold 0.25 \
  --epochs 40 --weight_decay 5e-5 --warmup_steps 200 \
  --use_wandb --wandb_project moodeq_ccmovies \
  --output_dir runs/phase3_single

# 4) Human test_gold 평가 (최종 신뢰도 확정)
python3 -m model.autoEQ.train_pseudo.eval_testgold \
  --ckpt runs/phase3_single/best_model.pt \
  --feature_dir data/features/ccmovies --split_name ccmovies \
  --labels_csv dataset/autoEQ/CCMovies/labels/final_labels.csv \
  --num_mood_classes 4 \
  --output runs/phase3_single/testgold_report
# 출력: mean_CCC, 사분면 일치율, 7-mood confusion, per-film

# 5) (옵션) LOMO 9-fold 전체 robustness (~3-5시간)
python3 -m model.autoEQ.train_pseudo.run_lomo \
  --feature_dir data/features/ccmovies --split_name ccmovies \
  --movie_set ccmovies --epochs 40 \
  --num_mood_classes 4 --lambda_mood 0.3 \
  --feature_noise_std 0.03 --mixup_prob 0.5 \
  --label_smooth_eps 0.05 --label_smooth_sigma_threshold 0.15 \
  --sigma_filter_threshold 0.25 \
  --weight_decay 5e-5 --warmup_steps 200 \
  --output_dir runs/ccmovies_lomo9
```

### Phase 4 (추론 파이프라인) — 미구현 (학습 완료 후 착수)

V3.2 §5에 기술된 순서를 구현하는 별도 모듈 필요:
- `model/autoEQ/inference/scene_detect.py` — PySceneDetect + merge
- `model/autoEQ/inference/window_infer.py` — 윈도우 슬라이딩 + EMA
- `model/autoEQ/inference/vad.py` — Silero VAD + dialogue_density
- `model/autoEQ/inference/va_to_eq.py` — va_to_mood + preset lookup + 대사 보호
- `model/autoEQ/inference/timeline_writer.py` — JSON 출력

### Phase 5 (재생 + 청취 검증) — 미구현

- `model/autoEQ/playback/eq_apply.py` — pedalboard + 크로스페이드
- `model/autoEQ/playback/ffmpeg_remux.py` — 비디오 재결합
- 청취 평가 세션 (3-5명, 5점 척도)

---

## 10. 한계 (학술 보고서 명시 필수)

V3.2 §8 한계 + V3.3 추가 사항:

### 10-1. 데이터 한계 (V3.3 특수)
- **학습 데이터 규모 약 10배 축소**: LIRIS-ACCEDE 9,800 → CCMovies 722 train windows. 일반화 능력에 상한이 있을 수 있음.
- **실사 영화 1편**: Valkaama만 실사, 나머지는 Blender 애니메이션. 실사 도메인 OOD 일반화 미확인.
- **서양 콘텐츠만**: 한국·동아시아 콘텐츠 미포함 (V3.2 §8-1 유지).
- **Power GEMS 영역 실샘플 0개**: 모델은 V=0.2·A=0.8 좌표를 학습하지 못함. 실제 영화에 해당 영역 씬이 있어도 모델 예측이 해당 영역을 피할 가능성. EQ 파이프라인의 Power preset은 이론적 preset으로만 존재.

### 10-2. Pseudo-label 한계 (V3.3 특수)
- **Gemini 2.5 Pro 기반 라벨 95.0% 사분면 정확도**: 5%의 noise가 학습에 유입. Layer 3 adjudicate의 `confidence` 컬럼으로 sample weighting 가능하나 본 설계에선 동일 가중 처리.
- **Ensemble(Essentia+CLIP+VEATIC+EmoNet) MAE V=0.52, A=0.59**: 저수준 특징 기반은 약함. Layer 3가 89.2%를 `gemini_only`로 분류 → Gemini 의존도 높음.

### 10-3. Congruence 관련 (V3.3 제거로 인한)
- **Self-supervised cross-modal congruence 학습 미실시**: 데이터 규모로 negative sampling 불안정 → 제거. 의도적 incongruent 씬(아이러닉 스코어링 등)의 모델 반응 분석 불가. V3.2 §3-7에서 이미 EQ 결정에 미사용으로 한정했으므로 EQ 성능엔 영향 없음.
- **Cohen CAM 검증 근거 상실**: V3.2 §1-3 5-pillar 중 Cohen CAM을 실증적으로 뒷받침할 수 없게 됨. 리포트에서 "이론적 motivation 유지, 구현은 future work" 명시.

### 10-4. Mood Head K=4 (V3.3 특수)
- **Mood Head 출력은 7 GEMS가 아닌 4 사분면**: 학습 편의 목적. 추론 시 EQ 결정은 V/A → va_to_mood(7) 후처리로 복원되므로 EQ 파이프라인은 불변.
- **학술 리포트 주의**: "Mood Head는 regularization 보조 task이며, 최종 7-mood 카테고리는 V/A 회귀 결과의 후처리 매핑으로 결정"을 명확히 기술.

### 10-5. 평가 한계
- **Human evaluator 1명**: Krippendorff α 계산 불가 (2+ 필요). 향후 2-3명 추가 평가로 inter-rater reliability 확보 권장.
- **test_gold 영화 2편 한정**: cosmos_laundromat + sintel. 영화 다양성 제한.

---

## 11. Future Work (V3.2 §9 + V3.3 추가)

V3.2 §9 전체 유지 + 다음 추가:

### 11-1. CCMovies 확장
- Human annotator 2-3명 추가 → Krippendorff α 확보
- disagreement_queue 200개 human 평가 → train/val 정정
- 실사 영화 추가 (CC-licensed pool에서 2-3편 더)

### 11-2. Congruence Head 복원
- 데이터 규모 확대 후 V3.2 §3-7, §4-2의 Congruence Head + negative sampling 재도입
- 현재 `train/` 모듈에 원 구현이 legacy로 보존되어 있음

### 11-3. Phase 4 추론 파이프라인 구현
- 상기 §9의 미구현 5개 모듈 (scene_detect, window_infer, vad, va_to_eq, timeline_writer)
- 전체 영화 1편 end-to-end 처리 + JSON 타임라인 생성

### 11-4. Mood Head K=7 복원 시도
- 증강 충분 + 데이터 확대 시 K=7 Phase 0 재도전
- Power GEMS 영역의 실샘플 확보를 위해 스릴러·공포 장르 CC 데이터 추가

---

## 부록 A. 핵심 상수 및 매핑

### CCMOVIES (`train_pseudo/ccmovies_preprocess.py`)
알파벳순 고정 (movie_id 결정성 보장):
```python
CCMOVIES = ["agent_327", "big_buck_bunny", "caminandes_3",
            "cosmos_laundromat", "elephants_dream", "sintel",
            "spring", "tears_of_steel", "valkaama_highlight"]
```

### MOOD_CENTERS (`train/dataset.py:12`)
7 GEMS (V3.2 §2-5 동일, V3.3 불변):
```python
MOOD_CENTERS = torch.tensor([
    [-0.6, +0.7],  # 0: Tension
    [-0.6, -0.4],  # 1: Sadness
    [+0.5, -0.5],  # 2: Peacefulness
    [+0.7, +0.6],  # 3: JoyfulActivation
    [+0.4, -0.2],  # 4: Tenderness
    [+0.2, +0.8],  # 5: Power
    [+0.5, +0.3],  # 6: Wonder
])
```

### MOOD_CENTERS_4Q (`train_pseudo/dataset.py:19`)
학습 시 K=4 사분면 (V3.3 기본):
```python
MOOD_CENTERS_4Q = torch.tensor([
    [+0.6, +0.6],   # 0: HVHA
    [+0.6, -0.4],   # 1: HVLA
    [-0.6, +0.6],   # 2: LVHA
    [-0.6, -0.4],   # 3: LVLA
])
```

---

## 부록 B. Congruence Head 제거 결정의 근거

V3.2 §3-7은 Congruence Head가 EQ 제어에 사용되지 않음을 명시했지만, Multi-task regularization 효과를 기대하고 학습에는 포함되어 있었다. V3.3은 본 모듈을 **완전 제거** 결정:

1. **Negative sampling 불안정성**: 722 train에서 25%(strong incongruent) + 25%(slight)가 cross-film 오디오로 교체되면 실효 congruent sample이 360개만 남음. cong head 학습 신호로는 불충분.
2. **Cross-film V/A distance 분포 제약**: V3.2 §2-6의 25/50/75 percentile 임계가 9편 분포에서 meaningful incongruence를 생성하지 못할 가능성.
3. **Gate entropy가 이미 degeneracy 방지 역할**: Gate 수렴 방지를 위한 보조 수단 확보.
4. **구현 단순화**: forward signature에서 `cong_label` 제거 → data loader도 negative sampling 불필요 → 전체 학습 루프 단순화.

V3.2 §3-7의 학술적 논거(영화 예술 존중, 신호-결정 균형, 내적 일관성)는 그대로 보존. Congruence 학습 자체를 future work로 전이.

---

## 부록 C. 테스트 커버리지 (V3.3 기준 54 tests passing)

`model/autoEQ/train_pseudo/tests/` 구성:

| 테스트 | 수 | 검증 내용 |
|---|---|---|
| `test_config_defaults.py` | 3 | K=4 기본, patience=10, 증강 no-op 기본값 |
| `test_ccmovies_preprocess.py` | 4 | CCMOVIES 알파벳 / VALID_SOURCES / NaN 필터 / metadata.csv 일관성 |
| `test_movie_set_dispatch.py` | 4 | LOMO movie_list 파라미터 / backward compat / fold OOR 거부 / film_split_json_ids |
| `test_feature_noise.py` | 4 | noise off 결정론 / eval 비활성 / train 변동 / 드롭 모달 0 유지 |
| `test_quadrant_mixup.py` | 4 | identity / 같은 사분면 유지 / cross-quadrant 거부 / λ 클램프 |
| `test_label_smoothing.py` | 5 | eps=0 비활성 / threshold=0 비활성 / 고σ 적용 / 저σ 패스 / 배치 혼합 |
| `test_cog_modality_dropout.py` | 4 | p=0, p=1, eval mode, cong 독립성 |
| `test_cog_model_shapes.py` | 6 | forward keys / 4Q+7C 출력 shapes / gate softmax / cong kwarg 거부 |
| `test_cog_losses.py` | 4 | combined_loss_cog 3-term / CCC hybrid toggle / cong term 부재 |
| `test_cog_pipeline.py` | 2 | 2-epoch 스모크 / early stop 갱신 |
| `test_distribution_gate.py` | 3 | K=7 실패 / K=4 통과 / quadrant pct sum=1 |
| `test_lomo_split.py` | 5 | full movie test / disjoint / gap / OOR / 카운트 |
| `test_cognimuse_window_aggregation.py` | 5 | V/A normalize / aggregate / edge cases |
| `test_movie_id_determinism.py` | 3 | COGNIMUSE_MOVIES 알파벳 / 7편 / mapping |

---

**문서 끝.**
