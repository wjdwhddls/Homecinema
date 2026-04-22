# LIRIS-ACCEDE 전환 플랜 V2 (피드백 반영 최종본)

## 0. 변경 이력

- **V1 → V2** (2026-04-20):
  1. §2-1 LIRIS annotation 실제 구조 반영 (V/A 1~5 scalar 존재, variance 컬럼 확인, 공식 split 재분석)
  2. "Pareto comparison" → "Lexicographic comparison" 전면 용어 교정
  3. "Label Smoothing" → "Target Shrinkage (regression variant)" 용어 교정
  4. 평가 지표 프레이밍 변경: CCC 단독 primary, Pearson·MAE는 diagnostic
  5. PANNs 4초 입력 sanity check를 Phase 1 착수 전 필수 작업으로 승격
  6. Mood K=7 유지 조건 3개 명시 (data sufficiency / learnability / non-interference)
  7. Phase 2 ablation에 AST vs PANNs, Pure MSE vs CCC hybrid vs Pure CCC, λ_gate_entropy 3 level 추가
  8. Bootstrap 95% CI 보고를 평가 섹션에 정식 통합

---

## 1. Context (왜 이 작업인가)

- 원래 V3.2 명세서(`/Users/jongin/Downloads/specification_v3_2.md`)에 따라 LIRIS-ACCEDE로 학습할 계획이었으나, 데이터셋 확보 불가로 V3.3 축소 명세(`SPECIFICATION_V3_3.md`, CCMovies 9편 + pseudo-label)로 전환 → V3.3 최종 모델(X-CLIP + AST + GMU + multi-task K=4) 학습 완료.
- **2026-04-20: LIRIS-ACCEDE 데이터셋이 `/Users/jongin/Downloads/LIRIS-ACCEDE-{data,annotations,features}.zip`으로 도착**. V3.3 진행분 취소 후 LIRIS로 재시작.
- 사용자 요구: **"V3.3의 학습 프로세스 외곽은 깔끔하니 유지, 모델 내부는 V3.2 원래 설계(PANNs + Gate + Multi-task)로 복귀"**.
- 본 플랜은 V1의 구조를 유지하되, 피드백 리뷰에서 지적된 LIRIS 데이터 오해·용어 오용·평가 방법 미흡 사항을 교정한 V2.

---

## 2. 사용자 확정 결정사항

1. **Congruence Head + Negative Sampling**: 미복원 (V3.3 loss 깔끔함 유지)
2. **Mood Head 클래스 수**: K=7 GEMS 복원 (단, §5의 3조건 게이트 통과 시)
3. **코드 구조**: 새 `model/autoEQ/train_liris/` 폴더 생성
4. **베이스 모델 구성**: V3.2 원래 설계 — **X-CLIP + PANNs CNN14 + Gate Network + Intermediate Fusion + Multi-task (VA + Mood K=7)**
5. **평가 지표 프레이밍**:
   - **Primary**: `mean_CCC` (AVEC 표준)
   - **Diagnostic**: `mean_Pearson` (선형 상관), `mean_MAE`·`mean_RMSE` (절댓값 오차), 축별 지표
   - Early stopping은 `(mean_CCC, mean_Pearson, −mean_MAE)` **lexicographic** tie-breaking (CCC 우선, 동률 시 Pearson, 다시 동률 시 MAE)
   - **"Pareto"가 아님** — Python tuple 비교는 lexicographic이므로 정확한 용어 사용

---

## 3. LIRIS-ACCEDE 실제 파일 구조 (V2 신규 — 직접 확인)

### 3-1. `ACCEDEranking.txt` (9,800 rows + header)
```
id | name | valenceRank | arousalRank | valenceValue | arousalValue | valenceVariance | arousalVariance
0  | ACCEDE00000.mp4 | 3755 | 787 | 2.76 | 1.37 | 0.118 | 0.149
```
- **valenceValue / arousalValue**: Baveye 2014 pairwise ranking 기반 scalar 회귀 결과, **1~5 스케일**
- **valenceVariance / arousalVariance**: per-clip annotation 불확실성 (Target Shrinkage 조건 판정에 사용 가능)
- **변환 공식**: `v_norm = (v_raw − 3) / 2`, `a_norm = (a_raw − 3) / 2` → [-1, +1] 범위

### 3-2. `ACCEDEsets.txt` (공식 split, 9,800 rows)
- `set = 0`: **test** (4,900 excerpts, 80 movies)
- `set = 1`: **learning** (2,450 excerpts, 40 movies)
- `set = 2`: **validation** (2,450 excerpts, 40 movies)
- 총 160 movies · 40 + 40 + 80

### 3-3. `ACCEDEdescription.xml` (9,800 clip metadata)
- 각 clip의 `<movie>` 태그 → 원본 영화명. **film_level_split()의 `film_id` 추출 소스**.

### 3-4. 본 프로젝트에서의 선택 (중요 결정)

V3.2 플랜은 120/20/20을 지정했고 V1은 이를 그대로 따랐다. 하지만 LIRIS 공식 split(40/40/80)이 존재하며, 타 LIRIS 논문과의 비교 가능성을 위해 **공식 split을 기본값으로 채택**한다:

| split | 영화 수 | clip 수 | 용도 |
|---|---|---|---|
| learning (set=1) | 40 | 2,450 | train |
| validation (set=2) | 40 | 2,450 | val (early stop 기준) |
| test (set=0) | 80 | 4,900 | test (최종 평가) |

Window 변환 후: **train ~9,800 / val ~9,800 / test ~19,600 windows** (평균 stride 2s, 8~12s clip에서 클립당 3~5 window)

**장점**: 타 LIRIS 벤치마크(Baveye 2015, ACII/TAC)와 직접 비교 가능
**단점**: test set이 train의 2배로 커서 학습 데이터 희소. 필요 시 `--use_full_learning_set` 플래그로 set 1+2를 모두 train으로 쓰고 자체 val 5% hold-out 옵션 제공.

### 3-5. Continuous annotation의 적용 범위
- LIRIS continuous subset은 **30 full movies에만 해당**, discrete 9,800 clip에는 continuous V/A가 없음
- V1에서 "continuous [0, 1] → [-1, +1]" 변환을 언급했으나 **discrete에는 적용 불가** → V2에서 제거

---

## 4. 구성 요약 ("V3.3의 학습 프로세스 외곽" × "V3.2의 모델 내부")

| 계층 | 선택 | 출처 |
|---|---|---|
| Visual encoder | X-CLIP base-patch32 (frozen, 512-d) | V3.2 · V3.3 공통 |
| Audio encoder | **PANNs CNN14 (frozen, 2048-d)** + Linear Projection 2048→512 (학습) | V3.2 원복 |
| Fusion | **Gate Network (1024 → 256 → 2 softmax)** + Intermediate Fusion `concat(w_v·v, w_a·a)` → 1024-d | V3.2 원복 |
| VA Head | 1024 → 256 → 2 | 공통 |
| Mood Head | 1024 → 256 → **7 (K=7 GEMS, 3조건 게이트 통과 시)** | V3.2 + 사용자 결정 |
| Congruence Head | 없음 | 사용자 결정 |
| Modality Dropout | p=0.05 전체 무조건 | V3.3 단순화 유지 |
| Feature Noise | σ=0.03 대칭 Gaussian | V3.3 신규 유지 |
| Quadrant Mixup | prob=0.5, α=0.4 | V3.3 신규 유지 |
| **Target Shrinkage** (구 "Conditional Label Smoothing") | ε=0.05, var_threshold=0.15 (고-uncertainty만) | V3.3 개념 유지, 용어 교정 |
| V/A loss | CCC hybrid (0.7·MSE + 0.3·(1−CCC)), Phase 2에서 w∈{0, 0.3, 1.0} ablation | V3.3 채택 + §2-5 검증 |
| Mood loss | CE(K=7), λ_mood=0.3 | V3.3 구조 + K=7 |
| Gate entropy loss | λ_ent=0.05 (기본), Phase 2에서 {0, 0.05, 0.1} ablation | V3.2 복원 + §2-8 검증 |
| **Early stopping** | **Lexicographic** (mean_CCC, mean_Pearson, −mean_MAE), patience=10 | V3.3 2-tuple → 3-tuple 확장 + §2-2 용어 교정 |
| **Primary metric** | **mean_CCC** (AVEC 표준) — Pearson·MAE는 diagnostic | §2-3 재프레이밍 |
| σ-filter | 제거 | — |

---

## 5. Mood K=7 유지 3조건 게이트 (§2-10 반영)

K=7 GEMS Mood Head를 채택하려면 **아래 3조건을 모두 만족**해야 한다. 하나라도 실패 시 K=4 quadrant fallback (V3.3 구조 복귀).

### 조건 1: Data sufficiency
- LIRIS train split(2,450 clip × 3~5 window ≈ 9,800 windows)에서 각 7-class sample count가 전체의 **≥ 1%** (약 98개 이상)
- V/A → 7-centroid Euclidean 매핑 후 class별 count 집계

### 조건 2: Learnability
- K=7 학습 후 val **mood F1_macro > random baseline (1/7 ≈ 14.3%)**
- 7-class CE loss가 실제로 감소하고 최소한의 분류 능력 획득 확인

### 조건 3: Non-interference
- K=7 vs K=4 변형 대조 학습에서 **val mean_CCC 차이 ≤ 0.01**
- K=7 도입이 primary task(V/A 회귀)를 저해하지 않음 확인

**게이트 로직**:
```
if (min_class_ratio >= 0.01 and
    mood_f1_macro > 0.143 and
    abs(ccc_k7 - ccc_k4) <= 0.01):
    USE K=7
else:
    USE K=4 (fallback to V3.3 structure)
```

---

## 6. Phase 0 (신규) — 플랜 착수 전 30분 Sanity Check

### 6-0-1. PANNs 4초 입력 sanity check (§2-6)
PANNs CNN14는 AudioSet 10초 clip으로 pretrain. 4초 입력은 out-of-distribution. 입력 길이 mismatch가 feature 품질 저하 유발 가능 — 학습 3시간 투입 전 확인 필수.

**절차**:
1. LIRIS clip 10개 랜덤 선택
2. 각 clip에서 (a) 4s crop, (b) 동일 4s를 10s 영역에 zero-padding 두 버전 준비
3. PANNs forward → 2048-d feature 10쌍
4. cosine similarity 계산 후 평균

**판정**:
- mean_cos_sim ≥ **0.90** → 4초 그대로 진행
- 0.80 ≤ mean_cos_sim < 0.90 → zero-padding 채택 (`pad_to_10s=True`)
- mean_cos_sim < 0.80 → PANNs 4초 부적합. 대안:
  - 4초 feature 3개 평균 풀링 (12초를 3 chunk로)
  - AST로 회귀 (가변 길이 robust)
  - 사용자 재협의

산출물: `runs/phase0_panns_sanity/report.json` (클립별 cos_sim + 평균 + 권장 설정)

---

## 7. V3.2 ↔ V3.3 ↔ 본 플랜 정합성 매트릭스 (V2 업데이트)

| 항목 | V3.2 명세 요구 | V3.3 구현(CCMovies) | 본 플랜 V2 | 비고 |
|---|---|---|---|---|
| 학습 데이터 | LIRIS 9,800 / ~160편 | CCMovies 1,289 windows / 9편 | **LIRIS 9,800 / 160편** | V3.2 복귀 |
| OOD 검증 | COGNIMUSE 200 / 7편 | (legacy 코드만) | **COGNIMUSE 200 / 7편 유지** | V3.2 그대로 |
| Split | Train 120 / Val 20 / Test 20 | LOMO 9-fold | **LIRIS 공식 split 40 / 40 / 80 (기본)** 또는 full learning + 5% val hold-out (옵션) | §3-4 재분석, V3.2 flag보다 LIRIS 관행 존중 |
| V/A range | [-1, +1] | [-1, +1] | **(v_raw − 3) / 2 변환, [-1, +1]** | §3-1 실제 스케일 반영 |
| Per-sample std | — | ensemble std | **valenceVariance / arousalVariance 활용** | §3-1에서 확인됨. Target Shrinkage 조건 판정에 사용 |
| Window | 4s / stride 2·1 | 동일 | 동일 | 공통 |
| 라벨 소스 | LIRIS 단일 V/A | 3-Layer Pseudo-label + 200 human | **LIRIS valenceValue/arousalValue 직접 사용** | Pseudo-label 파이프라인 전면 폐기 |
| Video encoder | X-CLIP frozen | 동일 | 동일 | 공통 |
| Audio encoder | PANNs CNN14 (2048-d) | AST (768-d) | **PANNs 원복** (사용자 결정) | Phase 0 sanity + Phase 2 ablation (§2-7)로 검증 |
| Fusion | Gate Network + Intermediate Fusion | GMU | **Gate 원복** (사용자 결정) | LIRIS ~9,800 train windows에서 재검증 |
| Mood Head | K=7 (GEMS) | K=4 | **K=7 (3조건 게이트 통과 시)** | §5 게이트 로직 |
| Congruence Head | 포함 | 제거 | 미복원 (사용자 결정) | 학술 메시지 축소는 "V3.3 실증적 단순화 우위"로 방어 |
| Negative Sampling | 50/25/25 | 제거 | 미사용 | — |
| V/A loss | MSE | CCC hybrid w=0.3 | **CCC hybrid w=0.3 (기본)** + Phase 2 ablation (§2-5, w∈{0, 0.3, 1.0}) | 재검증 |
| Mood loss | CE(K=7), λ=0.5 | CE(K=4), λ=0.3 | **CE(K=7), λ=0.3** | 구조는 V3.3 λ 채택, 클래스는 V3.2 |
| Gate entropy | λ=0.05 | N/A (GMU 부적합) | **λ=0.05 (기본)** + Phase 2 ablation (§2-7, λ∈{0, 0.05, 0.1}) | Gate Network는 적용 가능 |
| Modality Dropout | p=0.1, cong_label==0만 | p=0.05 전체 | p=0.05 전체 | V3.3 단순화 유지 (cong 없음) |
| Feature Augmentation | 명시 없음 | Noise + Mixup + Shrinkage | **동일 유지** + Phase 2 sensitivity ablation | LIRIS 규모에서 과잉 가능성 |
| **Early stopping** | val mean_CCC, patience=10 | (mean_CCC, −mean_MAE) "Pareto" (실제론 lexicographic), patience=10 | **lexicographic (mean_CCC, mean_Pearson, −mean_MAE), patience=10** | §2-2 용어 교정 |
| **Primary metric** | mean_CCC | mean_CCC (Pareto에는 MAE도 동등 취급) | **mean_CCC 단독 primary, Pearson/MAE는 diagnostic** | §2-3 재프레이밍 |
| **결과 보고 형식** | 단일 값 | 단일 값 | **점 추정 + bootstrap 95% CI** | §2-8 |
| EQ 프리셋 | 7 × 10 Biquad peaking | 동일 | 동일 | 불변 |
| VAD / 대사 보호 | Silero VAD + α_d·B6/B7/B8 | 동일 | 동일 | 추론 파이프라인 재사용 |
| JSON 타임라인 | v1.0 스키마 | 구현 완료 | 그대로 재사용 | — |

---

## 8. 최종 학습 프로세스 차근차근 (End-to-End)

### Step 0. 데이터 규모 재계산 (LIRIS 공식 split 기준)
- LIRIS Discrete: 9,800 clip × 평균 10초 ≈ 98,000초
- 4s window · stride 2s → clip당 평균 ~4 window → **약 39,200 windows 전체**
- 공식 split: **train 2,450 × ~4 ≈ 9,800 / val 2,450 × ~4 ≈ 9,800 / test 4,900 × ~4 ≈ 19,600**
- Batch 32 → train epoch당 ~306 step, 40 epoch × 306 = 12,240 step

### Step 1. Feature Precompute (학습 전 1회, ~3시간)
각 window → `(visual: 512-d, audio: 2048-d)` .pt 캐시
- Visual: X-CLIP base-patch32, 8 frames × 224² → 512-d [CLS]
- Audio: PANNs CNN14, 16kHz mono mel-spectrogram → 2048-d GAP (Phase 0 sanity 결과에 따라 4s 또는 10s padded)
- 저장: `data/features/liris_panns/{liris_visual.pt, liris_audio.pt, liris_metadata.pt, manifest.json}`

### Step 2. 한 Training Step의 내부 흐름
```
(a) DataLoader __getitem__ per sample:
    {visual_feat (512,), audio_feat (2048,), v, a, mood_k7, v_var, a_var, film_id}
    ※ v = (valenceValue − 3) / 2, a = (arousalValue − 3) / 2  (§3-1 변환)
    ※ v_var, a_var는 Target Shrinkage 조건 판정에 사용

(b) collate_fn (batch level):
    ├─ [증강 3] Quadrant Mixup (prob=0.5): 같은 사분면 샘플 쌍, λ~Beta(0.4,0.4)→[0.1,0.9] shrink
    └─ [증강 4] Target Shrinkage: max(v_var, a_var) > 0.15 샘플만 (v, a) × 0.95
       ※ "Label Smoothing"이 아닌 "Target Shrinkage (regression variant)" — §2-4

(c) model.forward(visual, audio):
    ├─ [증강 1] Modality Dropout (p=0.05): 시각 or 오디오 중 하나 zero masking
    ├─ [증강 2] Feature Noise (σ=0.03): 대칭 Gaussian, dropped sample은 bypass
    ├─ AudioProjection: 2048 → 512
    ├─ Gate Network: concat(v,a) 1024 → 256 → 2 softmax → (w_v, w_a)
    ├─ Intermediate Fusion: fused = concat(w_v·v, w_a·a) → 1024-d
    └─ Heads:
        ├─ VAHead: 1024 → 256 → 2 → va_pred
        └─ MoodHead: 1024 → 256 → 7 → mood_logits

(d) Loss 계산
    L_total = L_va + λ_mood·L_mood + λ_ent·L_gate_ent
      L_va   = 0.7·MSE(va_pred, va_target) + 0.3·(1 − CCC(va_pred, va_target))
      L_mood = CE(mood_logits, mood_k7_label)
      L_ent  = −mean(−Σ p·log p)  ← Gate의 (w_v, w_a) entropy 최대화

(e) Backward + clip_grad_norm(1.0) + optimizer.step() + lr_scheduler.step()

(f) Metric 로깅: loss per head, grad_norm/{va,mood,gate}, gate_w_v_var, gate_entropy_value
```

### Step 3. Validation (매 epoch 종료 후, 증강·dropout·noise 모두 OFF)
```python
metrics = {
    # Primary
    'mean_ccc': (ccc_v + ccc_a) / 2,      # AVEC 표준, early stopping 주 기준
    # Diagnostics
    'mean_pearson': (pearson_v + pearson_a) / 2,
    'mean_mae': (mae_v + mae_a) / 2,
    'mean_rmse': (rmse_v + rmse_a) / 2,
    # Per-axis
    'ccc_v', 'ccc_a', 'pearson_v', 'pearson_a', 'mae_v', 'mae_a',
    # Mood diagnostic
    'mood_f1_macro', 'mood_cohen_kappa',
}

# Lexicographic early stopping (§2-2 용어 교정)
current = (mean_ccc, mean_pearson, -mean_mae)
best    = (best_mean_ccc, best_mean_pearson, -best_mean_mae)
if current > best:  # Python tuple 비교는 lexicographic
    save_best_model()
    patience = 0
else:
    patience += 1
    if patience >= 10:
        stop()
```

### Step 4. 최종 평가 (Bootstrap CI 포함, §2-8)
```python
from scipy.stats import bootstrap

# Test set 전체에서 (pred, target) 쌍 확보
data = (va_pred_array, va_target_array)

# 1,000회 resampling
res_ccc = bootstrap((data[0], data[1]), ccc_statistic,
                     n_resamples=1000, confidence_level=0.95)

# 보고: "mean_CCC = 0.550 [95% CI: 0.523, 0.578]"
```
세 데이터셋 모두 동일 적용:
- **LIRIS val** (40 films holdout)
- **LIRIS test** (80 films, 공식 test)
- **COGNIMUSE OOD** (7 films, 200 windows)

---

## 9. 가중치 및 설정 검증

### 9.1. Loss weights
| 파라미터 | 기본값 | 근거 | Phase 2 ablation |
|---|---|---|---|
| `ccc_hybrid_w` | 0.3 | V3.3 유지, AVEC 표준 관행 | **{0, 0.3, 1.0}** (§2-5) |
| `λ_mood` | 0.3 | V3.3 실측 근거 | **{0, 0.1, 0.3, 0.5}** |
| `λ_gate_entropy` | 0.05 | V3.2 원래 값 | **{0, 0.05, 0.1}** (§2-8) |

### 9.2. Optimizer
| 파라미터 | 값 | 검증 |
|---|---|---|
| Optimizer | AdamW | ✅ 표준 |
| LR | 1e-4 | ✅ 학습 파라미터 ~190만 규모 적정 |
| Weight decay | 1e-5 | ✅ Precomputed feature에서 과적합 여지 낮음 |
| LR scheduler | warmup 500 → cosine | ✅ 전체 ~12,240 step 중 500 warmup = 4% |
| Grad clip | max_norm=1.0 | ✅ CCC loss gradient spike 방지 |

### 9.3. 학습 스케줄
| 파라미터 | 값 | 검증 |
|---|---|---|
| Batch size | 32 | Phase 2 {32, 64} 비교 권장 |
| Epochs | 40 | V3.3 peak epoch 17~21, 40 + patience 10 충분 |
| Early stop patience | 10 | CCC noise 대응 |
| Seed | 42 | 재현성 |

### 9.4. 모델 차원 (§9.2 V3.2 명세와 일치)
| 모듈 | 차원 |
|---|---|
| AudioProjection | 2048 → 512 |
| Gate Network | 1024 → 256 → 2 |
| Intermediate Fusion | concat(w_v·v, w_a·a) → 1024 |
| VA Head | 1024 → 256 → 2 |
| Mood Head | 1024 → 256 → **7** |
**총 학습 파라미터 추정**: ~183만 (V3.2 명세 "약 190만"과 일치)

### 9.5. 종합 judgement
- ✅ **확정 유지**: optimizer, lr, weight decay, grad clip, lr scheduler, 모델 차원, CCC hybrid w, patience, seed, lexicographic early stop
- ⚠ **Phase 2 ablation 필수**: λ_mood, λ_gate_entropy, ccc_hybrid_w, batch_size, AST vs PANNs, 증강 on/off
- 🔴 **Phase 1에서 확정**: K=7 vs K=4 (3조건 게이트), Target Shrinkage 활성/비활성 (valenceVariance 분포 확인 후)

---

## 10. 데이터 증강 4종 상세

### 증강 1: Modality Dropout
- **위치**: `model.py::forward()` 입력 직후
- **동작**: 배치의 각 샘플에 p=0.05 확률로 trigger, 50/50으로 visual 또는 audio를 zero vector로 교체
- **효과**: Gate Network가 한 모달에만 의존하지 않도록 강제
- **추론 시 OFF**

### 증강 2: Feature Noise (Gaussian)
- **위치**: `model.py::forward()` Modality Dropout 다음
- **동작**: `x' = x + ε, ε ~ N(0, 0.03²)`, 대칭 적용(v·a 동일 σ), dropped sample은 bypass
- **효과**: Frozen encoder 출력의 작은 변동성 주입 → head 과의존 방지
- **추론 시 OFF**

### 증강 3: Quadrant Mixup
- **위치**: `dataset.py::collate_fn` (DataLoader)
- **동작**:
  - prob=0.5로 배치의 각 sample에 적용
  - **같은 사분면 쌍만** 섞음 (HVHA ↔ HVHA, 극단 섞기 금지)
  - `λ ~ Beta(0.4, 0.4) → 0.1 + 0.8·λ` (양 끝 제거)
  - visual·audio·v·a·v_var·a_var 모두 동일 λ로 선형 결합
  - mood label은 primary sample 유지
- **효과**: V/A 회귀 smoothness 유도
- **LIRIS 주의**: 9,800 train windows에서 CCMovies 722 대비 13배 큼 → 효과 약화 가능성, Phase 2 {0, 0.3, 0.5} ablation

### 증강 4: Target Shrinkage (구 "Conditional Label Smoothing", §2-4 용어 교정)
- **위치**: `collate_fn`
- **조건**: LIRIS의 `max(valenceVariance, arousalVariance) > 0.15` 샘플만 (고-uncertainty)
- **동작**: 해당 샘플의 `(v, a) → (v·0.95, a·0.95)` — 극단 target을 중심으로 5% 축소
- **구현**:
  ```python
  if target_shrinkage_eps > 0:
      for sample in batch:
          if max(sample.v_var, sample.a_var) > var_threshold:
              sample.v *= (1 - target_shrinkage_eps)  # 0.95
              sample.a *= (1 - target_shrinkage_eps)
  ```
- **효과**: Annotation 불확실성이 높은 샘플의 극단값 영향력 감쇠
- **LIRIS 적용**:
  - V1에서 "LIRIS는 per-sample std 없으니 비활성"이라 잘못 기술했으나, **실제로 valenceVariance/arousalVariance가 존재**(§3-1) → Phase 1에서 variance 분포 확인 후 threshold 0.15 적정성 재검증
  - 분포가 0.15 기준에 맞지 않으면 quartile 기반 재조정 (예: Q3 값 threshold)

### 증강 적용 순서
```
DataLoader __getitem__
   ↓
collate_fn
  ├─ Quadrant Mixup (prob=0.5)          ← 배치 단위
  └─ Target Shrinkage (var > 0.15 샘플)  ← 샘플 단위
   ↓
model.forward
  ├─ Modality Dropout (p=0.05)          ← 샘플 단위
  └─ Feature Noise (σ=0.03)             ← feature 단위
   ↓
Gate → Fusion → Heads → Loss
```

---

## 11. Critical Files (수정/신규)

### 11-1. 신규 생성 (`model/autoEQ/train_liris/`)
- **`config.py`** — `TrainLirisConfig`:
  - feature_dir=`data/features/liris_panns/`
  - audio_raw_dim=2048, fused_dim=1024
  - num_mood_classes=7 (K=4 fallback 포함)
  - λ_mood=0.3, λ_gate_entropy=0.05, ccc_hybrid_w=0.3
  - modality_dropout_p=0.05, feature_noise_std=0.03
  - mixup_prob=0.5, mixup_alpha=0.4
  - **target_shrinkage_eps=0.05, variance_threshold=0.15** (용어 교정)
  - batch_size=32, lr=1e-4, weight_decay=1e-5
  - epochs=40, early_stop_patience=10
  - grad_clip=1.0, warmup_steps=500, seed=42
  - **use_official_split=True** (ACCEDEsets.txt 사용)
  - **pad_audio_to_10s** — Phase 0 sanity 결과에 따라 설정

- **`liris_preprocess.py`** — LIRIS 데이터 전처리:
  - `ACCEDEranking.txt` 파싱 → `{id, film_id, v_raw, a_raw, v_var, a_var}` 생성
  - `(v-3)/2, (a-3)/2` 변환 (§3-1)
  - `ACCEDEdescription.xml` → 각 clip의 `<movie>` 태그로 `film_id` 추출
  - `ACCEDEsets.txt` → official split dict (set 0/1/2)
  - 4s window 슬라이싱 (stride 2s)
  - X-CLIP (512-d) + PANNs (2048-d) feature 추출
  - V/A 분포 시각화 + class count report
  - 출력: `data/features/liris_panns/{liris_visual.pt, liris_audio.pt, liris_metadata.pt, manifest.json}`

- **`dataset.py`** — `PrecomputedLirisDataset` + `official_split()` + `film_level_split()` (옵션)
  - 재사용: 기존 `train_pseudo/dataset.py::va_to_mood` (7-class GEMS)

- **`model.py`** — `AutoEQModelLiris`:
  - 기반: V3.3 `train_pseudo/model_base/model.py` (Gated Weighted Concat + PANNs + MoodHead)
  - MoodHead 출력: 1024 → 256 → 7
  - Congruence Head 없음

- **`losses.py`** — `combined_loss_liris`:
  - CCC hybrid VA + K=7 CE + gate_entropy
  - `ccc_hybrid_w` 파라미터화 (Phase 2 ablation {0, 0.3, 1.0})
  - `λ_gate_entropy` 파라미터화 (Phase 2 ablation {0, 0.05, 0.1})

- **`trainer.py`** — **Lexicographic early stopping** (§2-2):
  ```python
  current = (mean_ccc, mean_pearson, -mean_mae)
  best    = (self.best_mean_ccc, self.best_mean_pearson, -self.best_mean_mae)
  if current > best:  # tuple comparison is lexicographic
      update_best(); patience = 0
  else:
      patience += 1
  ```
  - Gate 로깅: `gate_w_v_var`, `gate_w_a_var`, `gate_entropy_value`
  - Head-wise grad norm: `va`, `mood`, `gate`

- **`metrics.py`** — 축별 지표 + **Bootstrap CI** (§2-8):
  - `compute_ccc`, `compute_pearson`, `compute_mae`, `compute_rmse` (per-axis)
  - `compute_bootstrap_ci(preds, targets, metric_fn, n_resamples=1000, ci=0.95)`
  - 매 validation: 점 추정값 + (선택적) bootstrap CI

- **`run_train.py`** — CLI entrypoint
- **`eval_test.py`** — 최종 평가 스크립트 (bootstrap CI 자동 계산)
- **`tests/`** — 8종 단위 테스트:
  1. official split 무결성 (같은 film이 여러 split에 없음)
  2. V/A 변환 공식 정확성 (`(v-3)/2` round-trip)
  3. Quadrant stratification
  4. CCC 공식
  5. Pearson 공식
  6. 7-class va_to_mood
  7. Modality dropout shape
  8. Mixup shape

### 11-2. 재사용 (변경 없음)
- `train_pseudo/dataset.py::va_to_mood` — 7-class GEMS centroid 매핑
- `train_pseudo/model_base/*` — V3.2 baseline 구조 참조/일부 복제
- `train/encoders.py` — X-CLIPEncoder / PANNsEncoder (liris_preprocess에서 재사용)
- `infer_pseudo/*` — 추론 파이프라인 전체 (VARIANTS에 `liris_base` 엔트리 추가)
- `playback/*` — pedalboard + crossfade + ffmpeg
- `scripts/eq_response_check.py`

### 11-3. 폐기/미사용
- `model/autoEQ/pseudo_label/*` — Essentia/EmoNet/VEATIC/Gemini 3-layer, Streamlit UI
- `train_pseudo/ccmovies_preprocess.py`, `cognimuse_preprocess.py` (legacy)
- σ-filter 관련 코드 전반
- `dataset/autoEQ/CCMovies/*` (보존만, 학습 미참조)
- `model/autoEQ/train/` V3.2 skeleton (historical 보존)

---

## 12. 실행 로드맵

### Phase 0 — 30분 Sanity Check (§2-6)
- [ ] LIRIS annotation zip 해제, ACCEDEranking.txt / ACCEDEsets.txt / ACCEDEdescription.xml 스키마 확인
- [ ] PANNs 4초 입력 sanity check (10 clip, cos_sim 측정)
- [ ] `pad_audio_to_10s` 확정
- 산출: `runs/phase0_panns_sanity/report.json`

### Phase 1 — LIRIS 데이터 준비 (1주)
1. zip 해제: `unzip LIRIS-ACCEDE-{data,annotations,features}.zip -d dataset/LIRIS_ACCEDE/`
2. **메타데이터 파싱**: ACCEDEranking.txt, ACCEDEsets.txt, ACCEDEdescription.xml
3. **V/A 변환**: `(v_raw − 3) / 2, (a_raw − 3) / 2` → [-1, +1]
4. **Variance 분포 확인**: valenceVariance/arousalVariance의 histogram, Q1/Q2/Q3 계산 → target_shrinkage의 variance_threshold 적정성 검증
5. **Official split 적용** (`use_official_split=True`) + film_id 추출 검증 (같은 film이 여러 split에 없음 assert)
6. **Mood K=7 3조건 게이트** (§5):
   - Data sufficiency: train 각 class count ≥ 1%
   - Learnability: 짧은 K=7 mini-run에서 F1 > 14.3%
   - Non-interference: K=7 vs K=4 짧은 대조 run에서 ΔCCC ≤ 0.01
   - 실패 시 K=4 fallback
7. **Feature precompute**: X-CLIP 512-d + PANNs 2048-d → `data/features/liris_panns/`. 예상 MPS 3시간
8. 산출: `runs/phase1_liris/distribution_report.json`, `k7_gate_result.json`

### Phase 2 — 학습 (1~2주)
1. `train_liris/` 스켈레톤 생성 후 단위 테스트 8종 통과
2. **베이스라인 학습**:
   ```bash
   python3 -m model.autoEQ.train_liris.run_train \
     --feature_dir data/features/liris_panns \
     --num_mood_classes 7 \
     --lambda_mood 0.3 --lambda_gate_entropy 0.05 \
     --ccc_hybrid_w 0.3 \
     --mixup_prob 0.5 --target_shrinkage_eps 0.05 \
     --epochs 40 --use_wandb \
     --output_dir runs/liris_phase2_baseline
   ```
3. **3-tuple lexicographic early stop 모니터링**: grad_norm, gate_entropy, mean_CCC(primary)
4. **Primary 지표 목표**:
   - **mean_CCC** (primary, AVEC 표준): baseline ≥ 0.30, stretch ≥ 0.45
   - **mean_Pearson** (diagnostic, tie-breaker): 보고만
   - **mean_MAE** (diagnostic, tie-breaker): 보고만
   - V3.3 CCMovies mean_CCC 0.5625 → LIRIS에선 더 높을 가능성, 실측 후 합격선 재정의
5. **Ablation 세트 (Phase 2 후반)**:
   - (a) Audio encoder: PANNs vs AST (§2-7)
   - (b) VA loss: `ccc_hybrid_w ∈ {0, 0.3, 1.0}` (§2-5)
   - (c) Gate entropy: `λ_ent ∈ {0, 0.05, 0.1}` (§2-8)
   - (d) Augmentation off-ablation (LIRIS 규모 과잉 여부)
6. **COGNIMUSE OOD 평가**: 별도 precompute + 동일 모델 추론

### Phase 3 — 추론/재생 파이프라인 (3~5일)
1. `infer_pseudo/model_inference.py::VARIANTS`에 `liris_base` entry 추가
2. 샘플 영화 1편 end-to-end 실행
3. 청취 평가: α_d ∈ {0.3, 0.5, 0.7} A/B

---

## 13. Verification (완료 확인)

### 13-1. 코드/데이터 무결성
- `pytest model/autoEQ/train_liris/tests/ -v` (8/8 PASS)
- `liris_preprocess --verify_split` — film_id 중복 0
- `liris_preprocess --verify_va_conversion` — `(v_raw − 3) / 2 ∈ [-1, +1]` round-trip

### 13-2. 학습 sanity
- 2 epoch mini-run에서 L_total 감소, grad_norm 10배 내 균형
- 3축 metric (CCC/Pearson/MAE) 모두 계산·로깅
- Lexicographic early stop 동작 확인

### 13-3. Metric 목표 (Primary CCC + Diagnostics)
**Primary (AVEC convention)**:
- LIRIS val: mean_CCC **≥ 0.30** (baseline) · **≥ 0.45** (stretch)
- LIRIS test: mean_CCC ≥ 0.28 (OOD 허용)
- COGNIMUSE OOD: mean_CCC ≥ 0.20 · min(CCC_V, CCC_A) > 0

**Bootstrap 95% CI** (§2-8): 모든 test 평가에서 점 추정 + CI 보고
```
예: mean_CCC = 0.550 [95% CI: 0.523, 0.578]
```

**Diagnostics (해석 보조)**:
- mean_Pearson 보고 (CCC와 차이 크면 bias/scale mismatch 진단)
- mean_MAE ≤ 0.25 (baseline) / ≤ 0.20 (stretch)
- 축별 지표 테이블

### 13-4. End-to-end
- 샘플 1편 `analyze → timeline.json → playback → remuxed mp4` 무오류 통과
- VAD 기반 대사 보호 적용 확인

---

## 14. 리스크 및 트레이드오프

1. **PANNs 4초 입력 품질** (§2-6): Phase 0 sanity로 사전 차단. 실패 시 zero-padding 또는 AST로 회귀.
2. **LIRIS 공식 split 선택의 비교 기준**: official 40/40/80 대 자체 120/20/20. 전자는 타 논문 비교 가능, 후자는 train 커짐. 기본값은 official, 필요 시 flag.
3. **K=7 Power 클래스 sparsity**: §5 3조건 게이트로 사전 차단, 실패 시 K=4 fallback.
4. **Congruence 미복원의 학술 메시지 축소**: 보고서 "V3.3 실증적 단순화: 표준 데이터(LIRIS)에서도 congruence 없이 충분" 방어 문단.
5. **Gate degenerate 가능성**: Modality dropout(0.05) + gate entropy(0.05)로 방지, Phase 2 ablation 확인.
6. **증강 효과 약화**: LIRIS ~9,800 train windows(CCMovies 722 대비 13배). Phase 2 off-ablation.
7. **용어 오용 전파 방지** (§2-2, §2-4): "Pareto", "Label Smoothing" 용어를 신규 코드에서 사용하지 않도록 리뷰.

---

## 15. 결론

**"V3.3 학습 프로세스 외곽 + V3.2 모델 내부" 하이브리드**로 LIRIS 전환 진행.

**V2 추가 교정사항**:
- LIRIS 실제 annotation 구조 반영 (`(v-3)/2` 변환, valenceVariance 활용, 공식 split 채택)
- "Pareto" → "Lexicographic" 용어 교정
- "Label Smoothing" → "Target Shrinkage" 용어 교정
- CCC 단독 primary, Pearson·MAE는 diagnostic으로 재프레이밍
- Phase 0 PANNs sanity check 30분 작업 추가
- Mood K=7 유지 3조건 게이트 명시 (data / learnability / non-interference)
- Phase 2 ablation 3종 추가 (AST/PANNs, ccc_hybrid_w, λ_gate_entropy)
- Bootstrap 95% CI 보고를 평가 섹션에 정식 통합

**학위 논문 수준 학술 정합성 확보**: 용어 정확성 (lexicographic, target shrinkage, AVEC-compliant CCC framing), 통계적 근거 (bootstrap CI), 데이터 구조 정합성 (LIRIS 공식 split, V/A 변환, variance 활용).

Phase 0 (30분) → Phase 1 (1주, LIRIS 데이터 준비 + K=7 게이트) → Phase 2 (1~2주, 베이스라인 + ablation 4종) → Phase 3 (3~5일, 추론 통합) 순으로 진행.
