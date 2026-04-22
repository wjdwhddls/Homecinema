# LIRIS-ACCEDE 전환 타당성 판정 + V3.3 프로세스 재사용 플랜

## Context (왜 이 작업인가)

- 사용자가 원래 V3.2 명세서(`/Users/jongin/Downloads/specification_v3_2.md`)에 따라 LIRIS-ACCEDE로 학습할 계획이었으나, 데이터셋 확보 불가로 V3.3 축소 명세(`SPECIFICATION_V3_3.md`, CCMovies 9편 + pseudo-label)로 전환 → V3.3 최종 모델(`X-CLIP + AST + GMU + multi-task(K=4)`) 학습 완료 상태.
- **2026-04-20 현재, LIRIS-ACCEDE 데이터셋이 `/Users/jongin/Downloads/LIRIS-ACCEDE-{data,annotations,features}.zip`으로 도착**. 사용자는 기존 V3.3 진행분을 모두 취소하고 LIRIS로 재시작 원함.
- 사용자의 핵심 요구: **"V3.3의 학습 프로세스가 깔끔해서 마음에 들어. 그대로 LIRIS 데이터만 갈아끼우고 싶다."**
- 본 플랜의 목표: V3.2 원래 명세서와 V3.3 실제 구현의 차이를 정밀 비교하여, V3.3 프로세스로 LIRIS 학습을 재개해도 V3.2 명세 관점에서 타당한지 판정 + 구체적 전환 로드맵 제시.

---

## 사용자 확정 결정사항 (AskUserQuestion 회신 반영)

1. **Congruence Head + Negative Sampling**: **미복원** (V3.3의 깔끔한 loss 구성 유지)
2. **Mood Head 클래스 수**: **K=7 GEMS 복원** (CCMovies에만 해당했던 Power 공백 제약 해소)
3. **코드 구조**: **새 `model/autoEQ/train_liris/` 폴더 생성** — V3.3 베스트 프랙티스만 깔끔하게 추출, 기존 `train/`(V3.2 skeleton)과 `train_pseudo/`(V3.3 구현)는 historical로 보존
4. **베이스 모델 구성 (사용자 추가 지시)**: **V3.2 원래 설계 복귀**. `X-CLIP + PANNs CNN14 + Gate Network + Intermediate Fusion + Multi-task (VA + Mood K=7)`. V3.3에서 AST/GMU로 교체된 것은 CCMovies small-data 한계에 대한 대응이었으므로, LIRIS 9800 클립에서는 원래 V3.2 설계가 통계적으로 타당.
5. **평가 지표 (사용자 추가 지시)**: **CCC · Pearson · MAE 세 축 모두 primary**. V3.3의 `(mean_CCC, −mean_MAE)` 2-tuple Pareto early stop을 **`(mean_CCC, mean_Pearson, −mean_MAE)` 3-tuple로 확장** — lexicographic 비교(CCC 최우선, 동률 시 Pearson, 다시 동률 시 MAE). 합격 게이트와 최종 보고서 테이블도 세 지표 모두 제시.

### 구성 요약 ("V3.3의 학습 프로세스 외곽" × "V3.2의 모델 내부")

| 계층 | 선택 | 출처 |
|---|---|---|
| Visual encoder | X-CLIP base-patch32 (frozen, 512-d) | V3.2 · V3.3 공통 |
| **Audio encoder** | **PANNs CNN14 (frozen, 2048-d) + Linear Projection 2048→512 (학습)** | **V3.2 원복** |
| **Fusion** | **Gate Network (1024 → 256 → 2 softmax) + Intermediate Fusion `concat(w_v·v, w_a·a)` → 1024-d** | **V3.2 원복** |
| VA Head | 1024 → 256 → 2 | V3.2 · V3.3 공통 |
| **Mood Head** | **1024 → 256 → 7 (K=7 GEMS)** | V3.2 + 사용자 결정 |
| Congruence Head | **없음** | 사용자 결정 |
| Modality Dropout | p=0.05 전체 무조건 (cong 없음) | V3.3 단순화 유지 |
| Feature Noise | σ=0.03 대칭 Gaussian | V3.3 신규 유지 |
| Quadrant Mixup | prob=0.5, α=0.4 | V3.3 신규 유지 |
| Label Smoothing | ε=0.05, σ_thr=0.15 (고-uncertainty만) | V3.3 신규 유지 (LIRIS에선 annotation std가 다르므로 Phase 2 ablation 필요) |
| **V/A loss** | **CCC hybrid (0.7·MSE + 0.3·(1−CCC))** | V3.3 채택 유지 |
| **Mood loss** | **CE(K=7), λ_mood=0.3** | V3.3 구조 + K=7 |
| **Gate entropy loss** | **선택적 복원** (V3.2 λ_ent=0.05). Gate Network가 scalar 2-way softmax이므로 적용 가능. Phase 2에서 ON/OFF ablation 1회 후 결정 | V3.2 복원 후보 |
| Early stopping | **(mean_CCC, mean_Pearson, −mean_MAE) 3-tuple Pareto, patience=10** | V3.3 2-tuple에서 Pearson 추가 확장 |
| 평가 지표(primary) | **mean_CCC · mean_Pearson · mean_MAE** 세 축 모두 primary | 사용자 지시 |
| σ-filter | **제거** (pseudo-label 불확실성 지표였음) | — |

---

## V3.2 ↔ V3.3 ↔ 본 플랜(V3.3+LIRIS) 정합성 매트릭스

| 항목 | V3.2 명세 요구 | V3.3 구현(CCMovies) | **본 플랜 (V3.3+LIRIS)** | 비고 |
|---|---|---|---|---|
| 학습 데이터 | LIRIS-ACCEDE 9,800 클립 / ~160편 | CCMovies 1,289 windows / 9편 | **LIRIS-ACCEDE 9,800 클립** | V3.2 복귀 |
| OOD 검증 | COGNIMUSE 200 / 7편 | (legacy 코드만) | **COGNIMUSE 200 / 7편 유지** | V3.2 그대로 |
| Film-level split | Train 120 / Val 20 / Test 20편 + COGNIMUSE OOD | LOMO 9-fold (영화 9편) | **Train 120 / Val 20 / Test 20 + COGNIMUSE OOD** | V3.2 복귀, LOMO 포기 (160편에 LOMO는 연산 비현실) |
| Stratification | V/A 4분면(HVHA/HVLA/LVHA/LVLA) 비례 분할 | 동일 | **동일** | V3.2 그대로 |
| Window | 4s / stride 2(학습) · 1(추론) | 동일 | **동일** | 공통 |
| 라벨 소스 | LIRIS 단일 V/A (정답) | 3-Layer Pseudo-label + Gemini + 200 human | **LIRIS ranking + continuous V/A annotations 직접 사용** | pseudo-label 파이프라인 전면 폐기 |
| V/A 범위 | [-1, +1] 정규화 | [-1, +1] (pseudo) | **LIRIS ranking [0, 9800]·continuous [0, 1] → [-1, +1] 변환** | Phase 1 전처리에서 수행 |
| Video encoder | X-CLIP frozen (512-d) | 동일 | **동일** | 공통 |
| Audio encoder | **PANNs CNN14 frozen (2048-d)** | AST frozen (768-d) (§4-3 결합 효과 근거) | **PANNs 원복** (사용자 결정) | LIRIS의 큰 데이터에서 V3.2 원래 설계가 ablation상 우위(σ=OFF 조건 Δ_CCC=+0.067) |
| Fusion | **Gate Network + Intermediate Fusion** | GMU (Arevalo 2017) | **Gate 원복** (사용자 결정) | V3.3의 GMU 채택 근거는 "1032 train/fold small-data에서 Gate가 over-engineered"였음. LIRIS ~40000 windows에서는 Gate가 정상 동작 예상 |
| Mood Head | K=7 (GEMS), Phase 0 실패 시 K=4 fallback | **K=4 (fallback 적용됨)** | **K=7 복원** (사용자 확정) | Phase 1 분포 확인 후 min class <1%면 재검토 |
| Congruence Head | 포함 (CE, λ=0.5) | **제거** | **미복원** (사용자 확정) | V3.2 학술 메시지 일부 축소. 보고서는 "V3.3 실증적 단순화의 우위"로 기술 |
| Negative Sampling | 50/25/25 비율, 같은 영화 제외 | 제거 | **미사용** | Congruence 미복원과 동일 이유 |
| V/A loss | MSE | **CCC hybrid** (0.7·MSE + 0.3·(1−CCC)) | **CCC hybrid 유지** | 감정 회귀 표준 지표. V3.3 개선을 외곽 프로세스로 흡수 |
| Loss 총항 | 4-term (va + mood + cong + gate_ent) | **2-term** (va + 0.3·mood) | **2-term 기본** (va + 0.3·mood) + **gate_entropy 선택적 복원** (λ_ent=0.05, Phase 2 ablation) | Gate Network는 scalar 2-way softmax이므로 entropy term 적용 가능 — V3.3의 "GMU에는 부적합" 문제 없음 |
| Modality Dropout | p=0.1, cong_label==0에만 | p=0.05, 전체 무조건 | **p=0.05 전체 무조건** | V3.3 단순화 유지 (cong 없으므로) |
| Feature Augmentation | 명시 없음 | Gaussian noise σ=0.03 + Quadrant Mixup prob=0.5, α=0.4 + Label Smoothing ε=0.05 | **동일 유지** | V3.3의 강점. 데이터 9800으로 커지면 효과 약해질 수 있어 Phase 2에서 sensitivity ablation 1회 권장 |
| σ-filter | — | −1.0 (OFF, V3.3 공식) | **제거 (파라미터 자체 삭제)** | pseudo-label 불확실성 지표였음. LIRIS 정답 기반이라 무의미 |
| Early stopping | val mean_CCC, patience=10 | (mean_CCC, −mean_MAE) Pareto, patience=10 | **Pareto 유지** | V3.3 개선 유지 |
| EQ 프리셋 | 7 × 10 Biquad peaking | 동일 | **동일** | §6-4 행렬 불변 |
| VAD / 대사 보호 | Silero VAD + α_d=0.5 · B6/B7/B8 | 동일 (infer_pseudo 구현 완료) | **동일** | 추론 파이프라인 그대로 재사용 |
| JSON 타임라인 | v1.0 스키마 | 구현 완료 | **그대로 재사용** | infer_pseudo/playback 모듈 재활용 |

**판정 총평**: 본 플랜은 **"V3.3의 학습 프로세스 외곽" × "V3.2의 모델 내부"** 하이브리드다. V3.3에서 CCMovies small-data 대응으로 교체됐던 **AST/GMU는 원래 V3.2 구성(PANNs + Gate Network + Intermediate Fusion)으로 되돌리고**, V3.3에서 정비된 학습 루프(**CCC hybrid loss · Pareto early stop · Feature Augmentation 3종 · 깔끔한 config 구조**)는 그대로 재사용. Mood Head는 K=7로 복원하여 V3.2 원래 설계와 일치. Congruence Head만 사용자 결정에 따라 미복원 — V3.2 학술 메시지는 일부 축소되지만 "V3.3의 실증적 단순성 우위"로 보고서에서 방어 가능.

---

---

## 최종 학습 프로세스 차근차근 (End-to-End 세부 흐름)

### Step 0. 데이터 규모 계산
- LIRIS-ACCEDE: 9,800 클립 × 평균 10초 = 98,000초
- 4s window · stride 2s → 클립당 평균 ~4 window → **약 39,000 학습 windows**
- Film-level split 120/20/20 → train ~29,000 / val ~5,000 / test ~5,000
- Batch 32 → epoch당 약 **900 training step**, 40 epoch × 900 = 36,000 step

### Step 1. Feature Precompute (학습 전 1회, ~3시간)
각 window → `(visual: 512-d, audio: 2048-d)` .pt 캐시
- Visual: X-CLIP base-patch32. 8 frames × 224×224 → 512-d [CLS]
- Audio: PANNs CNN14. 16kHz mono mel-spectrogram → 2048-d GAP
- 저장: `data/features/liris_panns/{liris_visual.pt, liris_audio.pt, liris_metadata.pt}`
- 학습 중에는 .pt에서 O(1) 로드 — encoder forward 재실행 없음 → 학습 epoch당 ~5분

### Step 2. 한 Training Step의 내부 흐름
```
(a) DataLoader batch 생성 (batch_size=32)
    ├─ __getitem__: {visual_feat (512,), audio_feat (2048,), v, a, mood_k7, v_std, a_std, film_id}
    └─ collate_fn:
        ├─ [증강 3] Quadrant Mixup (prob=0.5)
        │    └─ 같은 사분면 쌍 뽑아 λ~Beta(0.4,0.4)→[0.1,0.9] shrink, visual/audio/v/a 모두 동일 λ 선형결합
        └─ [증강 4] Conditional Label Smoothing (max(v_std,a_std)>0.15 샘플만 V/A × 0.95, ε=0.05)

(b) model.forward(visual, audio)
    ├─ [증강 1] Modality Dropout (p=0.05): 시각 or 오디오 중 하나를 zero masking
    ├─ [증강 2] Feature Noise (σ=0.03): 대칭 Gaussian noise, dropped sample은 bypass
    ├─ Linear Projection: audio 2048 → 512
    ├─ Gate Network: concat(v,a) 1024 → 256 → 2 softmax → (w_v, w_a)
    ├─ Intermediate Fusion: fused = concat(w_v·v, w_a·a) → 1024-d
    └─ Heads:
        ├─ VA Head: 1024 → 256 → 2  → va_pred
        └─ Mood Head: 1024 → 256 → 7 → mood_logits

(c) Loss 계산
    L_total = L_va + λ_mood·L_mood + λ_ent·L_gate_ent (optional)
      L_va   = 0.7·MSE(va_pred, va_target) + 0.3·(1 − mean_CCC(va_pred, va_target))
      L_mood = CE(mood_logits, mood_k7_label)
      L_ent  = -mean( -Σ p·log p )  ← Gate Network의 w_v,w_a 분포 entropy 최대화

(d) Backward + clip_grad_norm(1.0) + optimizer.step() + lr_scheduler.step()

(e) Metric 로깅: loss per head, grad_norm/{va,mood,gate}, gate_w_v_var, gate_entropy_value
```

### Step 3. Validation (매 epoch 종료 후)
```
with torch.no_grad(): model.eval()
for batch in val_loader:  # 증강/dropout/noise 모두 OFF
    va_pred, mood_logits, gate_w = model(...)
    accumulate predictions

metrics = {
    ccc_v, ccc_a, mean_ccc,
    pearson_v, pearson_a, mean_pearson,
    mae_v, mae_a, mean_mae,
    rmse_v, rmse_a, mood_acc_k7, mood_f1_macro,
}

3-tuple Pareto compare: (mean_ccc, mean_pearson, -mean_mae) lexicographic
  if better → save best_model.pt, reset patience
  else → patience += 1; if patience ≥ 10 → stop
```

### Step 4. 최종 평가 (학습 종료 후)
- Val holdout (20편) : 최고 체크포인트 3축 지표
- Test holdout (20편) : 동일 체크포인트 3축 지표 — in-distribution 일반화
- COGNIMUSE OOD (7편 · 200 windows) : Out-of-distribution 일반화

---

## 가중치 및 설정 검증 (각 값이 타당한지 점검)

### 4.1. Loss weights
| 파라미터 | 값 | 근거 | 검증 |
|---|---|---|---|
| `ccc_hybrid_w` | 0.3 | V3.3 유지. 순수 CCC loss는 학습 초기 불안정 → MSE와 혼합 | ✅ 표준 관행(AVEC 챌린지 benchmark) |
| `λ_mood` | 0.3 | V3.3 후속 조합 실측 근거. `shared backbone regularization` | ⚠️ **LIRIS에서 재검증 필요** — multi-task 효과는 데이터 스케일에 민감. Phase 2에서 {0, 0.1, 0.3, 0.5} ablation 1회 권장 |
| `λ_gate_entropy` | 0.05 | V3.2 원래 값. Gate degenerate(한쪽으로 쏠림) 방지 | ⚠️ **Phase 2 ablation 필수** — V3.3 CCMovies에선 과한 제약으로 작동했으나 LIRIS 규모에선 적정 예상 |

### 4.2. Optimizer
| 파라미터 | 값 | 검증 |
|---|---|---|
| Optimizer | AdamW | ✅ V3.2/V3.3 공통, 표준 |
| Learning rate | 1e-4 | ✅ 학습 파라미터 ~190만 규모에서 적정. 너무 높으면 CCC loss 발산, 너무 낮으면 40 epoch 내 미수렴 |
| Weight decay | 1e-5 | ✅ Precomputed feature 기반이라 과적합 여지 낮음. 1e-4는 과도 |
| LR scheduler | warmup 500 → cosine | ✅ 전체 ~36000 step 중 500 warmup = 1.4% → 표준 |
| Grad clip | max_norm=1.0 | ✅ CCC loss gradient 스파이크 방지 |

### 4.3. 학습 스케줄
| 파라미터 | 값 | 검증 |
|---|---|---|
| Batch size | 32 | ⚠️ **LIRIS 규모에서 64 상향 고려**. Precomputed feature라 OOM 거의 없음. 단 Mixup 효과가 배치 크기에 민감 — Phase 2에서 {32, 64} 비교 권장 |
| Epochs | 40 | ✅ V3.3 실측 peak epoch 17~21. 40 + patience 10이면 충분 |
| Early stop patience | 10 | ✅ CCC noise 대응, V3 이후 표준 |
| Seed | 42 | ✅ 재현성 |

### 4.4. 모델 구조 파라미터
| 모듈 | 차원 | 검증 |
|---|---|---|
| AudioProjection | 2048 → 512 | ✅ PANNs CNN14 표준 |
| Gate Network | 1024 → 256 → 2 | ✅ V3.2 §3-5 |
| Intermediate Fusion | concat(w_v·v, w_a·a) → 1024 | ✅ V3.2 §3-6. "Late Fusion" 아님 |
| VA Head | 1024 → 256 → 2 | ✅ |
| Mood Head | 1024 → 256 → **7** | ✅ V3.2 원복. V3.3 K=4 대비 head 파라미터만 1.5K 증가, 미미 |

**총 학습 파라미터 추정**: Linear Projection 105만 + Gate Network 26만 + VA Head 26만 + Mood Head 26만 ≈ **183만** (V3.2 명세 "약 190만"과 일치)

### 4.5. 전체 설정 judgement
**☑ 제대로 설정됨**: optimizer, lr, weight decay, grad clip, lr scheduler, 모델 차원, CCC hybrid w, 3-tuple Pareto, patience, seed
**⚠ Phase 2에서 ablation으로 검증 필요**: λ_mood (0/0.1/0.3/0.5), λ_gate_entropy (0/0.05), batch_size (32/64), 증강 on/off (LIRIS 규모에서 과잉 가능성)
**🔴 재계산 대상**: LIRIS 분포 확정 후 `mood K=7 class별 비율` 확인 — 최소 클래스 <1%면 `class_weight` 적용 또는 K=4 fallback

---

## 데이터 증강 4종 상세

### 증강 1: Modality Dropout
- **위치**: `model.py` 내 `forward()` 입력 직후 (encoder 출력을 받자마자)
- **시점**: `self.training == True`일 때만
- **동작**: 배치 내 각 샘플에 대해 p=0.05 확률로 trigger → trigger되면 50/50으로 visual 또는 audio 중 하나를 zero vector로 교체
- **구현 (pseudocode)**:
  ```python
  if self.training and p > 0:
      trig = torch.rand(B) < p  # p=0.05 → 기대 1.6 sample/batch
      choice = torch.randint(0, 2, (B,))  # 0=drop visual, 1=drop audio
      v[trig & (choice==0)] = 0
      a[trig & (choice==1)] = 0
  ```
- **효과**: Gate Network가 한 모달에만 의존(degenerate)하지 않도록 강제. Modality collapse 방지.
- **LIRIS 주의**: V3.2 원래 명세는 `cong_label==0`에만 적용이었으나, cong 미복원이므로 V3.3처럼 **무조건 전체 적용**.

### 증강 2: Feature Noise (Gaussian)
- **위치**: `model.py` 내 `forward()`에서 Modality Dropout 다음
- **시점**: `self.training == True`이고 `feature_noise_std > 0`
- **동작**: `x' = x + ε, ε ~ N(0, σ²)`, σ=0.03. 대칭 적용(visual/audio 동일 σ) → gate collapse 방지
- **구현 (pseudocode)**:
  ```python
  if self.training and feature_noise_std > 0:
      v_live = (v.abs().sum(-1, keepdim=True) > 0).float()  # dropped sample = 0
      a_live = (a.abs().sum(-1, keepdim=True) > 0).float()
      v = v + torch.randn_like(v) * feature_noise_std * v_live
      a = a + torch.randn_like(a) * feature_noise_std * a_live
  ```
- **효과**: Frozen encoder 출력의 작은 변동성 주입 → head가 특정 feature dim에 과의존 방지
- **주의**: dropped sample은 0 벡터 유지(bypass) — dropout 신호를 noise가 덮지 않게

### 증강 3: Quadrant Mixup
- **위치**: `dataset.py`의 `collate_fn` (DataLoader 단)
- **시점**: 배치별로 `prob=0.5` 확률로 적용
- **핵심 설계 원칙**: **같은 사분면 쌍만 섞음**. HVHA(+V,+A) ↔ HVHA만 섞기. LVLA ↔ HVHA 같이 극단 섞으면 감정 의미 모순 → 금지
- **λ 분포**: `λ ~ Beta(0.4, 0.4)` → `λ_shrunk = 0.1 + 0.8·λ` (양 끝 제거)
- **혼합 대상**: visual, audio, v, a, v_std, a_std **모두 동일 λ로 선형 결합**. `mood label`은 primary sample 사용 (one-hot 보존 목적)
- **구현 (pseudocode)**:
  ```python
  if random() < mixup_prob:
      for each sample i in batch:
          find j: same_quadrant(i, j), j != i
          λ_raw = Beta(α, α).sample()
          λ = 0.1 + 0.8 * λ_raw
          sample[i].visual = λ*sample[i].visual + (1-λ)*sample[j].visual
          # audio, v, a, v_std, a_std 동일 적용
          sample[i].mood 는 그대로 유지
  ```
- **효과**: 샘플 간 interpolation으로 V/A 회귀의 smoothness 유도. Small-data regularization.
- **LIRIS 주의**: ~29,000 train windows면 CCMovies 722 대비 40배 많음 → **Mixup 효과 감소 예상**. Phase 2에서 `mixup_prob ∈ {0, 0.3, 0.5}` ablation 1회.

### 증강 4: Conditional Label Smoothing
- **위치**: `collate_fn` (Mixup과 함께)
- **조건**: `max(v_std, a_std) > 0.15`인 샘플만 (고-uncertainty 샘플)
- **동작**: 해당 샘플의 `(v, a) → (v*0.95, a*0.95)` — 극단값을 0 방향으로 5% 축소
- **구현 (pseudocode)**:
  ```python
  if label_smooth_eps > 0:
      for each sample:
          if max(v_std, a_std) > 0.15:
              v_smoothed = v * (1 - label_smooth_eps)  # v * 0.95
              a_smoothed = a * (1 - label_smooth_eps)
  ```
- **효과**: Pseudo-label의 고-uncertainty 샘플 영향력 감쇠. 분류가 아닌 회귀에서의 label smoothing은 "극단값 완화" 형태로 변형.
- **LIRIS 주의**: **LIRIS는 ranking-based라 per-sample std가 없음**. → **Phase 1에서 두 가지 선택**:
  - (A) Label Smoothing 비활성화 (ε=0) — 가장 안전
  - (B) LIRIS continuous annotation의 annotator 간 std 유도 후 활용 — 복잡
  - 권장: (A) 비활성으로 시작, Phase 2 ablation에서 효과 확인

### 증강 적용 순서 요약
```
DataLoader __getitem__
   ↓
collate_fn
  ├─ Quadrant Mixup (prob=0.5)           ← 배치 단위
  └─ Conditional Label Smoothing          ← 샘플 단위, LIRIS에서 기본 OFF
   ↓
model.forward
  ├─ Modality Dropout (p=0.05)           ← 샘플 단위
  └─ Feature Noise (σ=0.03)              ← feature 단위
   ↓
Gate → Fusion → Heads → Loss
```

---

## Critical Files (수정/신규)

### 신규 생성 (`model/autoEQ/train_liris/`)
- `config.py` — `TrainLirisConfig` (feature_dir=`data/features/liris_panns/`, audio_raw_dim=**2048** (PANNs), fused_dim=**1024** (Gate+Concat), num_mood_classes=7, λ_mood=0.3, λ_gate_entropy=0.05, CCC hybrid w=0.3, modality_dropout_p=0.05, feature_noise_std=0.03, mixup_prob=0.5, mixup_alpha=0.4, label_smooth_eps=0.05, label_smooth_sigma_threshold=0.15, batch_size=32, lr=1e-4, epochs=40, patience=10, grad_clip=1.0, **σ-filter 파라미터 제거**)
- `liris_preprocess.py` — LIRIS-ACCEDE `ACCEDEranking.txt` + `ACCEDEsets.txt` + `ACCEDEannotations` 파싱 → 4s window 슬라이싱(stride 2s) → **X-CLIP (512-d) + PANNs CNN14 (2048-d)** feature 추출 → `data/features/liris_panns/{liris_visual.pt, liris_audio.pt, liris_metadata.pt, manifest.json}` 출력. LIRIS의 ranking score를 [-1, +1]로 정규화(min-max 또는 V3.2 §2-5 매핑 테이블 기반).
- `dataset.py` — `PrecomputedLirisDataset` + `film_level_split()`. **재사용**: 기존 `model/autoEQ/train_pseudo/dataset.py`의 `va_to_mood()`(7-class) import (공유 유틸로 이동 권장).
- `model.py` — `AutoEQModelLiris`. **기반: V3.3 `train_pseudo/model_base/model.py`** (Gated Weighted Concat + PANNs + MoodHead 이미 구현됨). MoodHead 출력만 `1024 → [256 → 7]`로 변경, cong head 부재 확인.
- `losses.py` — `combined_loss_liris` (CCC hybrid VA + K=7 CE + **선택적 gate_entropy**). V3.3 `train_pseudo/losses.py::combined_loss_cog`에서 cong term 제거, mood K=7로 수정, gate entropy term을 `λ_gate_entropy > 0`일 때만 활성화.
- `trainer.py` — **3-tuple Pareto-guard early stopping**. V3.3 `train_pseudo/trainer.py::TrainerCog` 복제 후 비교 tuple을 확장:
  ```python
  current = (mean_ccc, mean_pearson, -mean_mae)
  best    = (self.best_mean_ccc, self.best_mean_pearson, -self.best_mean_mae)
  if current > best:  # lexicographic: CCC first, Pearson tiebreaker, MAE second tiebreaker
      update_best(); patience = 0
  else:
      patience += 1
  ```
  GMU scalar 요약 로직 대신 **실제 Gate Network의 scalar 2-way softmax w_v, w_a** 로깅 (기존 `model_base/` 계보의 `gate_w_v_var`, `gate_w_a_var`, `gate_entropy_value` 메트릭 활용).
- `metrics.py` — 축별 지표 계산 유틸. V3.3 `train_pseudo/trainer.py` 내 `compute_ccc`, `compute_pearson`, `compute_mae` 추출. 출력 dict: `{ccc_v, ccc_a, mean_ccc, pearson_v, pearson_a, mean_pearson, mae_v, mae_a, mean_mae, rmse_v, rmse_a, mean_rmse}` — 전 지표를 매 validation마다 계산하여 wandb/json 로깅.
- `run_train.py` — CLI entrypoint.
- `tests/` — 단위 테스트 7종 (V3.2 §4-6 요구): film-level split 무결성, 4분면 stratification, CCC 공식, 7-class va_to_mood, modality dropout, feature noise, mixup shape.

### 재사용 (변경 없음)
- `model/autoEQ/train_pseudo/dataset.py::va_to_mood` — 7-class GEMS centroid 매핑 함수
- `model/autoEQ/train_pseudo/model_base/*` — V3.2 baseline 구조(Gated Weighted Concat + PANNs + MoodHead)가 이미 구현되어 있어 참조/일부 복제 가능
- `model/autoEQ/train/encoders.py` — X-CLIPEncoder / **PANNsEncoder** 유틸이 이미 존재. liris_preprocess에서 재사용
- `model/autoEQ/infer_pseudo/*` — 추론 파이프라인 전체 (scene detect, VAD, EMA smoother, mood_mapper, eq_preset, timeline_writer). `model_inference.py`의 `VARIANTS` dispatcher에 `liris_base` (PANNs+Gate+MoodK=7) 항목 추가하면 checkpoint 교체만으로 재사용 가능.
- `model/autoEQ/playback/*` — pedalboard EQ, crossfade, ffmpeg remux 전체 유지
- `scripts/eq_response_check.py`, `scripts/evaluate_lomo_testsets.py`

### 삭제/폐기 대상 (신규 플랜에서 사용 안 함)
- `model/autoEQ/pseudo_label/*` (Essentia/EmoNet/VEATIC/Gemini 3-layer, Streamlit UI) → LIRIS 정답 사용으로 불필요
- `train_pseudo/ccmovies_preprocess.py`, `cognimuse_preprocess.py` (legacy)
- σ-filter 관련 로직 전반 (`sigma_filter_threshold` 파라미터, `ensemble_std_v/a` 참조)
- `dataset/autoEQ/CCMovies/*` 전체 (보존은 하되 학습에서 미참조)

### 원본 V3.2 skeleton (`model/autoEQ/train/`) 
- **사용 안 함, historical 보존**. Congruence/Gate Network/PANNs 경로라 본 플랜(GMU+AST+CongOff)과 충돌. 향후 ablation 재실험 필요 시 복기용.

---

## 실행 로드맵 (Phase 1 → 3)

### Phase 1 — LIRIS 데이터 준비 (1주)
1. zip 3종 해제: `unzip /Users/jongin/Downloads/LIRIS-ACCEDE-{data,annotations,features}.zip -d dataset/LIRIS_ACCEDE/`
2. 메타데이터 파싱: `ACCEDEranking.txt`(pairwise ranking → V/A scalar), `ACCEDEsets.txt`(공식 split), continuous V/A annotations, dialogue VAD annotations 구조 확인.
3. **V/A 정규화 검증**: LIRIS ranking을 [-1, +1]로 min-max 변환 후 4분면 분포 계산. K=7 카테고리별 count 확인 (V3.2 §2-5 매핑 기준). **min class ≥ 1%**면 K=7 확정, 미달 시 사용자와 재협의.
4. Film-level split 작성 (`dataset/LIRIS_ACCEDE/splits/film_split.json`): 120 / 20 / 20, 4분면 stratified, seed=42.
5. 4s window 슬라이싱 + **X-CLIP visual (512-d) + PANNs CNN14 audio (2048-d)** precompute → `data/features/liris_panns/`. 예상 시간: MPS 기준 9800 클립 × ~1초/클립 ≈ 3시간.
6. V/A 분포 시각화 · 산출물 검증: `runs/phase1_liris/distribution_report.json`.

### Phase 2 — 학습 (1~2주)
1. `train_liris/` 스켈레톤 생성 후 단위 테스트 7종 통과.
2. 단일 fold 학습: `python3 -m model.autoEQ.train_liris.run_train --feature_dir data/features/liris_panns --num_mood_classes 7 --lambda_mood 0.3 --lambda_gate_entropy 0.05 --epochs 40 --use_wandb --output_dir runs/liris_phase2`.
3. 모니터링(V3.2 §4-6 + V3.3 §4-6 공통): head별 grad norm(`grad_norm/va`, `grad_norm/mood`), **Gate 진단 (`gate_w_v_var`, `gate_w_a_var`, `gate_entropy_value`)**, **3-tuple Pareto early stop (mean_CCC, mean_Pearson, −mean_MAE)**.
4. **Primary 평가 지표 (3축 모두)**:
   - **mean_CCC** (concordance, early stopping 1순위): V3.2 baseline `≥ 0.30`, stretch `≥ 0.45`
   - **mean_Pearson** (선형 상관, early stopping 2순위 tiebreaker): baseline `≥ 0.40`, stretch `≥ 0.55`
   - **mean_MAE** (절댓값 오차, early stopping 3순위 tiebreaker): baseline `≤ 0.25`, stretch `≤ 0.20`
   - 각 축(V, A) 개별 지표(`ccc_v`, `ccc_a`, `pearson_v`, `pearson_a`, `mae_v`, `mae_a`)도 함께 reporting
   - V3.3 실측(CCMovies에서 mean_CCC 0.5625)이 데이터 ~30배 많은 LIRIS에서 더 높을 가능성 — 실측 후 합격선 재정의.
5. Ablation 1회 권장: **λ_gate_entropy ∈ {0, 0.05}** (Gate degenerate 여부), **증강 off-ablation** (Mixup/Label Smoothing이 LIRIS 규모에서도 도움되는지).
6. COGNIMUSE OOD 평가: 별도 precompute(`data/features/cognimuse_panns/`) 후 동일 모델로 추론 · `scripts/evaluate_lomo_testsets.py` 재사용.

### Phase 3 — 추론/재생 파이프라인 검증 (3~5일)
1. `infer_pseudo/model_inference.py::VARIANTS`에 `liris_base` entry 추가 (PANNs+Gate+MoodHead K=7 반영).
2. 샘플 영화 1편으로 end-to-end 실행 → `timeline.json` 스키마 검증 + pedalboard EQ 재생 + dialogue 보호 확인.
3. Phase 3 청취 평가(V3.2 §7-4): α_d ∈ {0.3, 0.5, 0.7} A/B 비교.

---

## Verification (완료 확인 방법)

- **단위 테스트**: `pytest model/autoEQ/train_liris/tests/ -v` (7개 모두 PASS, CCC/Pearson/MAE 공식 테스트 포함)
- **데이터 무결성**: `python3 -m model.autoEQ.train_liris.liris_preprocess --verify_split`로 같은 film_id가 여러 split에 등장하지 않음을 assert
- **학습 sanity**: 2 epoch mini run에서 loss가 감소하고 grad norm이 10배 이내 균형, 3축 metric(CCC/Pearson/MAE) 모두 계산·로깅 확인
- **Metric 목표 (3축 모두 동시 만족)**:
  - mean_CCC ≥ 0.30 (baseline) · ≥ 0.45 (stretch)
  - mean_Pearson ≥ 0.40 (baseline) · ≥ 0.55 (stretch)
  - mean_MAE ≤ 0.25 (baseline) · ≤ 0.20 (stretch)
- **COGNIMUSE OOD**: test `min(CCC_V, CCC_A) > 0` · `min(Pearson_V, Pearson_A) > 0` in ≥ 6/7 folds
- **End-to-end**: 샘플 1편 `analyze → timeline.json → playback → remuxed mp4` 전 파이프라인 무오류 통과

---

## 리스크 및 트레이드오프

1. **PANNs precompute 연산량**: 9800 클립 × ~1초/클립 ≈ 3시간 (MPS). 중간에 끊기면 `manifest.json`의 resumable 로직 필요 — V3.3 `ccmovies_preprocess.py` 참고.
2. **LIRIS ranking 정규화 방식**: ranking score를 linear하게 [-1, +1]로 매핑할지, V3.2 §2-5의 4분면 stratify 기반으로 정규화할지 결정 필요. Phase 1에서 두 방식 분포를 비교해 선택 (COGNIMUSE 분포와 정합도가 높은 방식 채택).
3. **Mood K=7 Power 클래스 sparsity**: V3.3에서 K=4로 축소한 이유 재발생 가능성. Phase 1 분포 분석 후 min class < 1%이면 K=4로 fallback하거나 weighted CE 사용.
4. **Congruence 미복원의 학술 메시지 약화**: V3.2 5-pillar 중 Cohen CAM이 빠지는 효과. 보고서에서 "V3.3 실증적 단순화: 작은 데이터뿐 아니라 표준 데이터(LIRIS)에서도 congruence 축 없이 충분한 성능 달성"으로 방어 문단 작성 필요.
5. **Gate Network degenerate 가능성**: V3.2 설계의 잠재적 취약점 — Modality dropout(p=0.05) + gate entropy(λ=0.05)로 방지. V3.3 CCMovies에서는 "over-engineered"로 판단됐지만 LIRIS 규모에서는 적정. Phase 2 ablation으로 확인.
6. **데이터 증강 효과 약화 가능성**: Mixup/Label Smoothing이 CCMovies(1289 windows)에서 중요했으나 LIRIS(~40000 windows)에서는 regularization 과잉일 수 있음. Phase 2에서 off-ablation 1회 권장.

---

## 결론

**"V3.3 학습 프로세스 외곽 + V3.2 모델 내부"** 하이브리드로 LIRIS 전환이 타당하다.

- **V3.2 모델 내부 복귀**: X-CLIP + **PANNs CNN14** + **Gate Network + Intermediate Fusion** + **Multi-task (VA + Mood K=7)**. V3.3의 AST/GMU/K=4는 CCMovies 1289-window small-data 제약에 대한 대응이었으므로, LIRIS ~40000 windows에서는 원래 V3.2 설계가 ablation 근거(Δ_CCC=+0.067)상 타당.
- **V3.3 학습 프로세스 외곽 재사용**: CCC hybrid loss, Pareto early stopping(mean_CCC, −mean_MAE), Feature Augmentation 3종(Noise/Mixup/Label Smoothing), 깔끔한 config 구조, Phase별 검증 파이프라인.
- **제거**: Pseudo-label 3-layer ensemble, human annotation UI, σ-filter, Congruence Head, Negative Sampling.
- **복원**: Mood Head K=7 (V3.2 원래), Gate entropy loss 선택적 복원(λ_ent=0.05).

사용자가 선호한 "깔끔한 학습 프로세스"는 그대로 유지되고, 모델 아키텍처는 V3.2 원래 설계로 복귀하여 논문상 "LIRIS-ACCEDE 표준 데이터에 V3.2 명세를 구현"한 형태로 기술 가능. V3.3는 학술 보고서에서 "small-data 대응 실증 연구"로 별도 섹션 취급.
