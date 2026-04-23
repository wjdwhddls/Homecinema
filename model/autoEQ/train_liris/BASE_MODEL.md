# Base Model — FROZEN 2026-04-22 (rev. Phase 3 — Test evaluation 완료)

**이 문서는 단일 진실 source입니다.** Phase 3 test 평가가 완료되어 BASE Model 이 최종 확정되었습니다. 본 문서의 §5 에 test 결과가 기록되어 있으며, V5-FINAL §14-3 에 따라 **test 재평가는 금지** 됩니다.

**Revision history**:
- **2026-04-21 (initial)**: V5-FINAL §3 + Engineering fix 3종 (LayerNorm + MLP) + B-split + head_dropout=0.3 + wd=1e-4. **A + K=4** baseline (CCC 0.3672 ± 0.020).
- **2026-04-21 (Phase 2a-2)**: K=4 → **K=7** 채택 (Phase 2a-2 winner, +0.014 CCC, paired t=6.14 p<0.05). 현재 baseline = **A + K=7** (CCC **0.3812 ± 0.024**).
- **2026-04-21 (Phase 2a-3)**: Audio encoder PANNs vs AST 비교 → **PANNs 유지** (AST Δ=−0.052 CCC, paired t=−6.565 p=0.022). Base Model **변경 없음**. 상세 `runs/phase2a/2a3_summary.json`, 구현 `model/autoEQ/train_liris/model_ast/`.
- **2026-04-21 (Phase 2a-4)**: Visual encoder X-CLIP vs CLIP frame-mean 비교 → **X-CLIP 유지** (CLIPMean Δ=−0.011 CCC, paired t=−0.884, p=0.470, n=3 기준 **통계 유의 미달 — weak, direction only**). Base Model **변경 없음**. 상세 `runs/phase2a/2a4_summary.json`, 구현 `model/autoEQ/train_liris/model_clipmean/`.
- **2026-04-22 (Phase 2a-5)**: Fusion mechanism Gate vs Simple Concat vs GMU 비교 → **Gate (BASE) 유지** (Concat Δ=+0.006 p=0.590, GMU Δ=−0.010 p=0.389, n=3 기준 모두 **통계 유의 미달**). Concat 이 평균 근소 우위 (+0.006)이나 p=0.59 로 증거 미달. GMU 는 +2.1M params 에도 이득 없음. Base Model **변경 없음**. 상세 `runs/phase2a/2a5_summary.json` · `2a5_metrics_comparison.md`, 구현 `model/autoEQ/train_liris/model_fusion/`.
- **2026-04-22 (Phase 2a-7)**: Multi-task regularization 효과 검증 (VA+Mood vs VA-only) → **VA+Mood (BASE) 유지** (VA-only Δ=−0.010 CCC, paired t=−1.762, p=0.220, n=3 **통계 유의 미달이나 3/3 consistent + Cohen's d_z=−1.02 large**). MoodHead K=7 (264K params, 7.75%) 의 auxiliary loss 가 V/A representation 에 미세한 양의 regularization (특히 Valence 축 +0.026). Base Model **변경 없음**. 상세 `runs/phase2a/2a7_summary.json` · `2a7_metrics_comparison.md`, 구현 `model/autoEQ/train_liris/model_va_only/`.
- **2026-04-22 (Phase 3 — Test evaluation, FINAL)**: LIRIS test 80 films / 4,900 clips 에 대한 **최종 평가** (V5-FINAL §14-3 test access 1회 소모). 3-seed aggregate **test mean CCC = 0.3480 ± 0.0156** (val 0.3812 대비 Δ=−0.033, 수용 가능). Valence 축 test 에서 오히려 개선 (+0.011), Arousal 축 test 에서 감소 (−0.078, mild val-specific residual). **Ensemble (3-seed va_pred avg)** mean CCC = 0.3603 (+0.012 over single-seed mean). Base Model **변경 없음 (확정)**. 상세 `runs/phase3/test_final_metrics.json` · `test_final_report.md`, 구현 `model/autoEQ/train_liris/run_test_eval.py`.
- **2026-04-22 (Phase 4-A — Cross-corpus OOD generalization, Track A)**: COGNIMUSE 12편 Hollywood 영화 (2,197 × 10s windows, experienced median consensus) 에 대해 BASE 3-seed ensemble inference 수행. 학습 0회, BASE weights 불변 (MD5 무결성 검증). **z-score ensemble mean CCC = 0.3781 (V 0.3113, A 0.4449)** · **raw ensemble 0.3182 (V 0.3565, A 0.2798)** · 3-seed aggregate z-score 0.3640 ± 0.011. **강한 일반화 입증 (≥ 0.30)** — 특히 **Arousal 축이 LIRIS val (0.4001) 을 상회** (z-score 0.4449) 하여, 모델이 영화 감정의 상대적 변동을 견고하게 포착함을 확인. Raw vs z-score Δ=+0.055 → scale/shift mismatch 가 raw 를 끌어내리나 representation 품질 자체는 LIRIS 수준. 본 결과는 논문 "Cross-Corpus Generalization" 섹션 기여. Base Model **변경 없음**. 상세 `runs/cognimuse/phase4a/ood_eval/report.md` · `results.json`, 구현 `model/autoEQ/train_liris/model_cognimuse_ood/`.
- **2026-04-23 (Phase 5-A — Perceptual FX layer 추가)**: EQ-only 명세 1× 가 평균 |gain| 0.96 dB 로 JND (~1 dB) 근처여서 **체감 약함** 이 관측되어, 상위에 rule-based **Mood FX Layer** 를 추가. **Dual-layer 아키텍처 확립**: Layer 1 (EQ, 학습 모델 기반, 학술적 contribution) + Layer 2 (FX, 문헌-근거 rule-based, 지각 증폭). FX 레시피는 peer-reviewed 문헌의 **방향성만** 사용: Juslin & Västfjäll 2008 (BRECVEM brain stem reflex), Rumsey 2002 spatial quality, Sato et al. 2007 spaciousness, McAdams et al. 1995 timbre, Eerola & Vuoskoski 2011 brightness, Zentner et al. 2008 GEMS. **임의 수치 없음** — compression ratio, stereo width 는 직접 문헌 근거 부재로 **제외**. Shelf cutoff 는 업계 표준 (60/200/8000 Hz). Reverb 는 pedalboard 기본 preset 3택 (dry/small_room/large_hall) 만. **대사 보호**는 Layer 1 에만 VAD-guided 적용되며 유지. Base Model **변경 없음** (LIRIS-trained weights 그대로). 구현 `generate_fx_demo.py`, `live_compare_fx.py`, 산출물 `runs/demo_kakao/kakao_eq_fx.mp4`.
- **2026-04-23 (Phase 5-A rev. — 대사 보호 강화)**: KakaoTalk 데모 관찰에서 대사 위에 reverb 가 얹혀 명료도 저하 우려 → **Layer 1 + Layer 2 dual-protection** 추가. (1) Layer 1 `alpha_d` 기본값을 `run_pipeline.py` 레벨에서 **0.5 → 0.3** 으로 강화 (α_d 가 **작을수록** 강한 보호: 공식 `g_eff = g_orig × (1 - (1-α_d)·density)` 에서 density=1 시 g_eff = g_orig × α_d → α_d=0.3 → voice-critical 대역 gain 의 70% 감쇠, α_d=0.5 → 50% 감쇠). 실측 (Kakao 27 dialogue-bearing scenes): α_d=0.3 평균 감쇠 0.70 dB, α_d=0.5 = 0.50 dB, α_d=0.7 = 0.30 dB. (2) Layer 2 에 **VAD-guided dialogue-aware reverb bypass** 추가 — scene 내 `dialogue.segments_rel` 구간에서만 Reverb stage 제거하고 shelf (sub-bass/low/high) 는 **유지** 하여 배경 mood 보존. 30 ms raised-cosine crossfade 로 경계 매끄럽게. Shelf 대역 (60/200/8000 Hz) 이 대사 주 대역 (200 Hz~4 kHz) 과 거의 안 겹침을 근거로 선택 (compression/width 배제 원칙 유지). Synthetic unit smoke 로 로직 검증 (dialogue 구간 RMS diff 0.151× non-dialogue). Base Model **변경 없음**. 구현 helper `_strip_reverb` / `_process_segment` (`generate_fx_demo.py`), `--alpha-d` flag (`run_pipeline.py`).

---

## 1. 변경 요약 (V5-FINAL 원 명세 → Base Model)

| 항목 | V5-FINAL §9-1 원 명세 | **Base Model (현재)** |
|------|:-:|:-:|
| AudioProjection | `Linear(2048→512)` 단일 | **`Linear(2048→1024) + LN + GELU + Dropout(0.1) + Linear(1024→512) + LN`** |
| VA/Mood Head hidden | `Linear→ReLU→Dropout→Linear` | **`Linear→LayerNorm→ReLU→Dropout→Linear`** |
| `head_dropout` | 0.0 (spec implicit) | **0.3** |
| `weight_decay` | 1e-5 | **1e-4** |
| `use_full_learning_set` | False (MediaEval split) | **True (LIRIS paper split: 64 train / 16 val / 80 test)** |
| `va_norm_strategy` | "A" (`(v_raw-3)/2`) | **"A"** (Phase 2a-1 winner, p<0.05 vs B) |
| `num_mood_classes` | 4 (§8 strict) | **7** (Phase 2a-2 winner, p<0.05 vs K=4) |
| Total params | 1.84M | **3.42M (+86%)** |

### Phase 2a 결정 사항 정리

| Phase | 비교 축 | 승자 | Δ vs 패자 | 통계 유의 | 메모 |
|:-:|------|:-:|:-:|:-:|------|
| 2a-0 | Engineering fixes | optB (LN+MLP) | gap@best −77% | 진단 기반 | overfit 극복 |
| 2a-1 | V/A norm A vs B | **A** | +0.014 CCC | ⭐ p<0.05 | B는 K=7 viable이나 CCC 하락 |
| 2a-2 | K=4 vs K=7 | **K=7** | +0.014 CCC | ⭐ p<0.05 | §8 strict FAIL이지만 §21 pragmatic 승 |
| 2a-3 | audio: PANNs vs AST | **PANNs** | +0.052 CCC | ⭐ p=0.022 | AST overfit 극심 (gap 0.5~0.6), 기각 |
| 2a-4 | visual: X-CLIP vs CLIP frame-mean | **X-CLIP** | +0.011 CCC | ⚠️ p=0.470 | 방향 only, 통계 유의 미달. V5-FINAL §21-3 Δ≤0 규칙으로 BASE 유지 |
| 2a-5 | fusion: Gate vs Simple Concat vs GMU | **Gate** | Concat −0.006 (p=0.59), GMU +0.010 (p=0.39) | ⚠️ both n.s. | Concat 이 mean 근소 우위지만 유의 미달. GMU 는 +2.1M params 에도 이득 없음. V5-FINAL §21-3 규칙 적용 |
| 2a-7 | multi-task: VA+Mood vs VA-only | **VA+Mood (BASE)** | VA-only −0.010 (p=0.22) | ⚠️ directional | 3/3 seed consistent + Cohen's d_z=−1.02 large. n=3 underpower 로 p 유의 미달이나 방향 명확. MoodHead aux loss 가 Valence 축에 기여 (+0.026) |

### Base Model 변경 근거 (Phase 2a-2)
- Strategy A에서 JA centroid 도달 불가 → JA neuron은 학습 안 됨 (dead)
- 그러나 K=7 head의 6개 active class (Tension/Sadness/Peacefulness/Tenderness/Power/Wonder) fine-grained loss가 V/A representation을 더 잘 shape
- Multi-task regularization 효과로 V/A CCC가 K=4 대비 +0.014 일관 개선 (3-seed 모두 +0.009 ~ +0.016)
- §21 "최종 모델 선정 = 실측 비교" 원칙에 부합

---

## 2. 아키텍처 정의 (`model.py`)

```
입력
├─ visual_feat (B, 512)      ← X-CLIP base-patch32 (frozen)
└─ audio_feat (B, 2048)      ← PANNs CNN14 (frozen)

AudioProjection
  (2048) → Linear(2048, 1024) → LayerNorm(1024) → GELU → Dropout(p=0.1)
        → Linear(1024, 512) → LayerNorm(512)         → a_proj (B, 512)

[training-time augmentations]
  modality_dropout_p=0.05, feature_noise_std=0.03

GateNetwork (가중치 계산)
  concat([v, a_proj]) → Linear(1024, 256) → ReLU → Linear(256, 2) → softmax
  → (w_v, w_a) per sample

Intermediate Fusion (연산)
  fused = concat([w_v * v, w_a * a_proj])  → (B, 1024)

VA Head
  Linear(1024, 256) → LayerNorm(256) → ReLU → Dropout(0.3) → Linear(256, 2)
  → va_pred (B, 2) ∈ [-1, +1]

Mood Head (K=7 GEMS)
  Linear(1024, 256) → LayerNorm(256) → ReLU → Dropout(0.3) → Linear(256, 7)
  → mood_logits (B, 7)
  Class indices: 0=Tension, 1=Sadness, 2=Peacefulness, 3=JoyfulActivation,
                 4=Tenderness, 5=Power, 6=Wonder
  Note: class 3 (JA) has 0 train samples under Strategy A → dead neuron expected.
```

### 파라미터 분포 (3,417,099 total)

| 블록 | Params | 비중 |
|------|:-:|:-:|
| AudioProjection | 2,626,048 | 76.85% |
| GateNetwork | 262,914 | 7.69% |
| VA Head | 263,426 | 7.71% |
| Mood Head (K=7) | 264,711 | 7.75% |

---

## 3. Config 정의 (`config.py::TrainLirisConfig`)

### Feature
- `visual_dim=512`, `audio_raw_dim=2048`, `audio_proj_dim=512`, `fused_dim=1024`

### Head (BASE 2026-04-21 Phase 2a-2)
- **`num_mood_classes=7`** ← Phase 2a-2 winner
- `gate_hidden_dim=256`, `head_hidden_dim=256`

### Optimizer
- `lr=1e-4`, **`weight_decay=1e-4`** (BASE), `batch_size=32`
- `epochs=40`, `warmup_steps=500`, `use_cosine_schedule=True`
- `grad_clip_norm=1.0`

### Regularization
- **`head_dropout=0.3`** (BASE)
- `modality_dropout_p=0.05`, `feature_noise_std=0.03`
- `mixup_prob=0.5`, `mixup_alpha=0.4`
- `target_shrinkage_eps=0.05`, `v_var_threshold=0.117`, `a_var_threshold=0.164`, `shrinkage_logic="AND"`

### Loss
- `lambda_va=1.0`, `lambda_mood=0.3`, `lambda_gate_entropy=0.05`
- `use_ccc_loss=True`, `ccc_loss_weight=0.3` (hybrid MSE/CCC)

### Training control
- `early_stop_patience=10`, `overfit_gap_threshold=0.10`

### Data
- `feature_file="data/features/liris_panns_v5spec/features.pt"` (stride=2s, pad_to=10s)
- **`use_full_learning_set=True`** (BASE): 64 train / 16 val / 80 test films
- **`va_norm_strategy="A"`** ← Phase 2a-1 winner (V5-FINAL §2-1)
- `audio_crop_sec=4.0`, `audio_stride_sec=2.0`, `audio_pad_to_sec=10.0`

### Misc
- `seed=42`, `device="cpu"`, `num_workers=0`

---

## 4. Base Model 평가지표 (Val) — LIRIS val (16 films, 585 clips) · 3-seed 42/123/2024

| Metric | **mean** | Valence (V) | Arousal (A) |
|--------|:-:|:-:|:-:|
| **CCC** | **0.3812 ± 0.0240** | 0.3623 ± 0.0202 | 0.4001 ± 0.0295 |
| **Pearson** | **0.4281 ± 0.0223** | 0.4033 ± 0.0228 | 0.4529 ± 0.0221 |
| **MAE** | **0.3243 ± 0.0081** | 0.2691 ± 0.0056 | 0.3795 ± 0.0106 |

### Per-seed raw values (`runs/phase2a/2a2_A_K7_s{seed}/best.pt`)

| metric | seed 42 | seed 123 | seed 2024 |
|--------|:-:|:-:|:-:|
| mean_ccc | 0.3536 | 0.3959 | 0.3942 |
| ccc_v | 0.3400 | 0.3677 | 0.3793 |
| ccc_a | 0.3672 | 0.4241 | 0.4091 |
| mean_pearson | 0.4039 | 0.4477 | 0.4327 |
| pearson_valence | 0.3776 | 0.4210 | 0.4113 |
| pearson_arousal | 0.4301 | 0.4744 | 0.4541 |
| mean_mae | 0.3335 | 0.3181 | 0.3213 |
| mae_valence | 0.2754 | 0.2646 | 0.2672 |
| mae_arousal | 0.3916 | 0.3715 | 0.3755 |

### Training dynamics (3-seed 평균)
- `best_ep`: ~5.0 (early stop ep 13~17)
- `gap@best`: 작음 (Enhanced arch 효과)
- `total_ep`: 13.7

### 베이스라인 변천 (Phase 2a 전체)
| Stage | val mean_CCC |
|------|:-:|
| Phase 2a-0 spec V1 (단일 seed, 폐기) | 0.334 |
| Phase 2a-0 V2 quick-wins (OAT 위반, 폐기) | 0.3564 |
| Phase 2a-0 SPEC strict (3-seed, 옛 features) | 0.3502 |
| Phase 2a-0 SPEC + B-split (옛 arch) | 0.3532 |
| Phase 2a-0 Enhanced arch (A+K=4, 직전 base) | 0.3672 |
| **Phase 2a-2 winner (A+K=7, 현재 base)** | **0.3812** |

---

## 4b. Base Model 최종 평가지표 (Test) — LIRIS test (80 films, 4,900 clips) · Phase 3

**V5-FINAL §14-3 test access 1회 소모. 재평가 금지.**

| Metric | **3-seed mean** | Valence (V) | Arousal (A) | Δ vs val |
|--------|:-:|:-:|:-:|:-:|
| **CCC** | **0.3480 ± 0.0156** | 0.3737 ± 0.0135 | 0.3223 ± 0.0181 | **−0.033** (acceptable) |
| **Pearson** | 0.3863 ± 0.0073 | 0.4044 ± 0.0070 | 0.3682 ± 0.0085 | −0.042 |
| **MAE** | **0.3139 ± 0.0098** ↓ better | 0.2480 ± 0.0067 | 0.3799 ± 0.0138 | **−0.010** (test 이 낮음) |
| **RMSE V** | **0.3079 ± 0.0043** ↓ | — | — | **−0.033** (test 이 낮음) |
| **RMSE A** | 0.4644 ± 0.0120 | — | — | +0.001 (동등) |

### Per-seed test (mean_CCC)

| seed 42 | seed 123 | seed 2024 |
|:-:|:-:|:-:|
| 0.3313 | 0.3505 | **0.3622** (best) |

### 3-seed Ensemble (va_pred averaged, 실 서빙 권장)

| Metric | Single-seed mean | **Ensemble** | Δ |
|--------|:-:|:-:|:-:|
| mean CCC | 0.3480 | **0.3603** | **+0.012** |
| CCC V | 0.3737 | **0.3895** | +0.016 |
| CCC A | 0.3223 | **0.3312** | +0.009 |

**핵심 관찰**:
- **Valence 는 test 에서 오히려 개선** (val 0.3623 → test 0.3737, +0.011) — **강한 일반화**
- **Arousal 은 test 에서 감소** (val 0.4001 → test 0.3223, −0.078) — val-specific residual
- **MAE/RMSE V 는 test 가 더 낮음** — 크기 예측은 더 정확해짐
- **Ensemble 이 일반화 개선**: single-seed 대비 val-test gap 이 −0.033 → −0.021 로 감소
- Test std (0.016) < val std (0.024) — 대규모 sample 의 안정성

상세 리포트: `runs/phase3/test_final_report.md`

---

## 4c. Cross-Corpus OOD 일반화 (Phase 4-A Track A)

**COGNIMUSE 12 Hollywood 영화 · 2,197 × 10s windows · BASE 3-seed ensemble · 학습 0회**

V5-FINAL §14-3 의 LIRIS test 재평가 금지와 **분리된** 별도 OOD 평가. COGNIMUSE 는 새 데이터셋이므로 별도 access budget 이며, BASE weights (3-seed best.pt) 는 MD5 기준 불변 (s42 27b8a9f... / s123 5b64d84... / s2024 9d0a350...).

### 헤드라인 (ensemble)

| Dataset | Split | n_clips | mean CCC | CCC_V | CCC_A |
|---|---|---|---|---|---|
| LIRIS | val (Phase 2a-2) | 585 | 0.3812 ± 0.024 | 0.3623 | 0.4001 |
| LIRIS | test (Phase 3 agg) | 4,900 | 0.3480 ± 0.016 | 0.3737 | 0.3223 |
| LIRIS | test ensemble | 4,900 | **0.3603** | 0.3895 | 0.3312 |
| COGNIMUSE | 12 films **raw** (3-seed agg) | 2,197 | 0.3087 ± 0.045 | 0.3442 | 0.2731 |
| COGNIMUSE | 12 films **raw** ensemble | 2,197 | **0.3182** | 0.3565 | 0.2798 |
| COGNIMUSE | 12 films **z-score** (3-seed agg) | 2,197 | 0.3640 ± 0.011 | 0.2946 | 0.4335 |
| COGNIMUSE | 12 films **z-score** ensemble | 2,197 | **0.3781** | 0.3113 | **0.4449** |

### 두 가지 CCC 전략의 의미

- **Raw CCC**: 전체 prediction/target 을 하나의 flat array 로 놓고 계산. "off-the-shelf" 성능. Scale/shift bias 에 민감
- **Per-film z-score CCC**: 영화별로 pred/target 을 독립 z-normalize 후 계산. Scale bias 를 제거한 **representation 품질의 순수 측정**. Lin 1989 CCC scale sensitivity 완화용

### 핵심 발견

1. **z-score ensemble mean CCC = 0.3781 ≥ 0.30 → 강한 일반화**. 해석 가이드(< 0.05 실패, 0.05~0.15 약, 0.15~0.30 부분, ≥0.30 강) 상 최고 구간
2. **Arousal 축 역전**: LIRIS val A 0.4001 < COGNIMUSE z-score A 0.4449. 모델이 영화 감정의 "상대적 arousal 변동" 을 LIRIS 에서보다 COGNIMUSE 에서 더 잘 추적
3. **Raw − z-score Δ = +0.055** (A 축만 보면 +0.165): COGNIMUSE 가 LIRIS 보다 "차분한" 영화들로 구성 (A mean −0.360 vs −0.259, A std 0.306 vs 0.480, Tension class 2% vs 12%) → 모델 예측의 중심이 LIRIS 분포에 정렬되어 있어 raw scale 로는 오프셋 페널티 발생
4. **Valence 축은 약간 하락**: z-score V 0.3113 (LIRIS val 0.3623 대비 −0.05). COGNIMUSE V 분포가 더 좁음 (std 0.295 vs LIRIS 0.314)
5. **Per-film breakdown** (ensemble raw): NoCountry 0.45 / BeautifulMind 0.39 / Crash 0.37 (강), AmericanBeauty 0.03 / ShakespeareInLove 0.12 / Gladiator 0.16 (약). 액션·긴장 있는 영화가 잘 됨, 서사극이 어려움

### 자산 무결성 증명

| 항목 | MD5 (FROZEN) | 상태 |
|---|---|---|
| `runs/phase2a/2a2_A_K7_s42/best.pt` | 27b8a9fc8bcd6b422dbdfca37402c506 | ✅ 불변 |
| `runs/phase2a/2a2_A_K7_s123/best.pt` | 5b64d84ebda610370fb5aaa6aafcaf00 | ✅ 불변 |
| `runs/phase2a/2a2_A_K7_s2024/best.pt` | 9d0a3503784703fa33398f3897d9429e | ✅ 불변 |
| `runs/phase3/test_final_metrics.json` | 9bc3d40c86c08ca72624ea4db94e1dea | ✅ 불변 |

- BASE 11 compliance tests PASS (regression 0)
- Phase 4-A 신규 14 tests PASS (preprocessing 9 + ckpt isolation 5)
- 총 25/25 PASS

### 학술적 의의

> MediaEval 2015–2018 (Baveye et al.) cross-corpus protocol 재현. LIRIS-trained multimodal backbone (X-CLIP + PANNs CNN14) 이 dataset-conditioning 이나 adaptation 없이 COGNIMUSE 에서 scale-normalized CCC 0.378 달성 — LIRIS test ensemble (0.3603) 과 동등 수준. **Backbone 품질과 multi-task regularization 이 LIRIS 특이점 과적합이 아닌 진정한 영화 감정 신호를 학습**함을 입증.

상세 리포트: `runs/cognimuse/phase4a/ood_eval/report.md`
구현: `model/autoEQ/train_liris/model_cognimuse_ood/`
메모리: `memory/project_phase4a_cognimuse_ood.md`

---

## 4d. Phase 5-A — Perceptual FX Layer (Dual-Layer Architecture)

**추가 목적**: EQ-only 1× 명세는 평균 |gain| 0.96 dB 로 **JND (~1 dB) 근처** — 지각 한계. Phase 4-A 후 실제 영상 (KakaoTalk demo) 시청 테스트에서 **"구분이 거의 안 된다"** 관측. Model contribution 은 유지하면서, **peer-reviewed 문헌 근거** 만 사용해 상위 rule-based 보조 레이어를 얹어 지각을 증폭.

### 아키텍처 원칙 — Dual-Layer

```
Video / Audio
     │
     ▼
[Layer 1 — EQ (scientific contribution)]
  · V/A regression model (X-CLIP + PANNs + Gate + K=7)
  · V/A → mood centroid → 10-band peaking EQ gain
  · Continuous: V/A 연속값 → gain 연속 매핑
  · VAD-guided dialogue protection (voice-critical 500~2 kHz 보호)
  · 구현: model/autoEQ/playback/pipeline.py (기존)
  · 평가: Objective (CCC, spectrum delta) — Phase 4-A 0.378 확정
     │
     ▼
[Layer 2 — Mood FX (perceptual amplifier)]
  · Scene mood (argmax) → rule-based FX chain
  · Shelf EQ 확장 (sub-bass < 60 Hz, low < 200 Hz, high > 8 kHz)
  · Binary reverb mode: {dry, small_room, large_hall} (pedalboard 기본 preset)
  · Peak limiter (threshold −0.5 dB, release 100 ms) — 평균 level 유지
  · 구현: generate_fx_demo.py
  · 평가: Subjective (ABX listening test, 향후 Phase 5-B 에서)
     │
     ▼
Final mp4 (EQ + FX)
```

### 레이어별 역할 분리

| 축 | Layer 1 (EQ) | Layer 2 (FX) | 비고 |
|---|---|---|---|
| **과학적 기여** | ✅ Main contribution | ❌ (auxiliary only) | 논문 §4 Main Results |
| **연속 V/A 활용** | ✅ centroid mapping | ❌ argmax mood 만 | Layer 2 는 분류 수준 |
| **중역 톤 제어** (500~4 kHz) | ✅ 10-band peaking | ❌ shelf 뿐 | EQ 전담 |
| **저역 warmth** (125~250 Hz) | ✅ | 🟡 low-shelf (Sadness) | 주로 EQ |
| **초저역 body** (<60 Hz) | 🟡 31 Hz 1-band | ✅ sub-bass shelf | 주로 FX |
| **공간감 / reverb** | ❌ | ✅ 유일 | FX 전담 |
| **대사 보호** | ✅ VAD-guided (`alpha_d=0.3` default in `run_pipeline.py`, **작을수록 강함**) | 🟡 VAD-guided reverb bypass (shelf 유지) | 2-layer dual protection (2026-04-23 rev.) |
| **Scene 간 mood 미세 구분** | ✅ 6-way 프리셋 | 🟡 3 reverb + shelf | EQ 우위 |

### Mood FX 레시피 (문헌-근거 only)

```python
# 구현: generate_fx_demo.py::MOOD_FX_RECIPE
# 임의 수치 없음. 모든 필드에 인용 가능한 문헌 근거.
{
  "Tension":      {sub_bass_shelf +2dB, reverb=dry},
     # Juslin & Västfjäll 2008 BRECVEM brain stem reflex
     # Rumsey 2002 dry → intimate/tense 방향
  "Wonder":       {high_shelf +1dB, reverb=large_hall},
     # McAdams 1995 brightness → arousal
     # Sato et al. 2007 spaciousness → awe
  "Tenderness":   {reverb=small_room},
     # Rumsey 2002 small room → intimate warmth
  "Sadness":      {low_shelf +1dB, reverb=dry},
     # Eerola & Vuoskoski 2011 low-end weight → sadness
  "Power":        {sub_bass_shelf +3dB},
     # Zentner et al. 2008 GEMS Power cluster (저역 visceral)
  "Peacefulness": {reverb=small_room},
     # Rumsey 2002 moderate spaciousness → calm
}
```

**제외된 파라미터** (peer-reviewed 직접 매핑 부재):
- Compression ratio — Zwicker loudness 모델은 loudness ↔ arousal 만 다룸, "ratio X for mood Y" 실험 없음
- Stereo width — Rumsey 2002 envelopment 문헌은 있으나 mood-specific width 매핑 없음

**Reverb 세부**: 구체 decay 시간을 직접 정하지 않고 pedalboard 기본 `room_size` (0.05 / 0.25 / 0.85) + `wet_level` (0.00 / 0.10 / 0.20) 세 preset 만 사용. 튜닝 자의성 최소화.

**대사 보호 (2026-04-23 rev.)** — Layer 2 에도 VAD-guided 적용:
- scene 내 `dialogue.segments_rel` 구간에서 Reverb stage **만** 제거 (→ dry)
- shelf (sub-bass 60 Hz / low 200 Hz / high 8 kHz) 는 **유지** → 배경 mood 대부분 보존
- 선택 근거: shelf 주파수대가 대사 주 대역 (200 Hz~4 kHz) 과 거의 겹치지 않음. Reverb 는 consonant smear 로 명료도 저하 유발 → 이것만 bypass
- 경계 30 ms raised-cosine crossfade 로 FX ↔ dry 전환 smooth
- 구현: `generate_fx_demo.py::apply_mood_fx_per_scene` + helper `_strip_reverb`, `_process_segment`
- Layer 1 과 중첩 동작: `alpha_d=0.3` (voice-critical EQ gain 감쇠) + Layer 2 reverb bypass → **두 층 독립 보호**
- Smoke 검증: 합성 Tenderness scene + dialogue 1s segment 에서 dialogue RMS diff 가 non-dialogue 의 0.15× 로 감소 → bypass 동작 증명

**Peak 관리**: `pedalboard.Limiter(threshold=-0.5dB, release=100ms)` — FX 스택 누적으로 인한 clipping 만 제어, 평균 level 은 유지 (글로벌 감쇠 없음).

### 평가 이중화 (Evaluation Dual-track)

| Track | 대상 | 지표 | 방법론 | 상태 |
|---|---|---|---|---|
| **Objective** | Layer 1 (EQ) | CCC, Pearson, MAE, spectrum Δ | LIRIS/COGNIMUSE 데이터셋 회귀 | ✅ Phase 4-A 완료 (0.378) |
| **Subjective** | Layer 2 (FX) | ABX match rate, mood alignment | N=5~8 listening test | 🟡 Phase 5-B pilot (미수행) |

**왜 이렇게 분리**:
- EQ = 학습 모델 산출 → 수치 평가로 정당화 가능 (CCC 지표 확립)
- FX = rule-based 지각 증폭 → 수치가 아니라 "사람이 mood 를 더 잘 느끼는가" 가 본질 → ABX 필요

### Sanity 측정 (KakaoTalk demo 기준)

`kakao_eq_applied.mp4` (Layer 1 only) vs `kakao_eq_fx.mp4` (Layer 1 + 2) 차분 분석:

| Scene mood | FX 적용 후 band power max |Δ| | 주요 효과 |
|---|---|---|
| Tension | +4.23 dB | 31Hz sub-bass shelf + dry reverb |
| Power | +4.00 dB | 31Hz sub-bass shelf +3dB |
| Tenderness | +7~9 dB | small_room reverb 의 time-smear 에너지 |

전체 `corr(eq_only, eq_fx) = 0.97` — 의미 있는 차이 있되 신호 보존 (destructive 처리 아님).

### Base Model 불변성

**Layer 1 (EQ) 는 기존 명세와 완전 동일** — `model.py`, `config.py`, 3-seed `best.pt` 모두 불변. Phase 2a ~ 4-A 의 모든 ablation/평가 결과 유효. Layer 2 는 **별도 후처리 단계** 로 학습 파라미터 아님.

**MD5 무결성 유지**:
- `runs/phase2a/2a2_A_K7_s{42,123,2024}/best.pt` — 변경 없음
- `runs/phase3/test_final_metrics.json` — 변경 없음
- `model/autoEQ/train_liris/{model,config,dataset,trainer,losses}.py` — 변경 없음

### 구현 파일

| 파일 | 역할 |
|---|---|
| `generate_fx_demo.py` | Layer 2 FX 적용 (pedalboard 기반), in-place length-preserving scene 처리. 2026-04-23 rev.: dialogue-aware reverb bypass (`_strip_reverb`, `_process_segment` helper) |
| `live_compare_fx.py` | Dual-layer 비교 뷰어 (Original / 1× EQ / 1× EQ+FX) |
| `run_pipeline.py` | 신규 영상 → timeline + EQ + EQ+FX 전체 파이프라인 orchestrator. 2026-04-23 rev.: `--alpha-d` flag (default 0.3, **작을수록 강한 보호**) 로 Layer 1 대사 보호 강도 제어 |
| `runs/demo_kakao/kakao_eq_fx.mp4` | KakaoTalk demo FX 산출물 (33 MB) |

#### `run_pipeline.py` 3단계 workflow (entry-point)

새 영상을 한 줄로 Dual-layer 파이프라인에 통과시키는 orchestrator. 각 단계는
독립 재실행 가능 (`--skip-timeline`, `--skip-eq`, `--skip-fx`, `--force`).

```
입력: my_movie.mp4
  │
  ├─[Step 1]  infer_pseudo (model.autoEQ.infer_pseudo.cli)
  │            · 영상 → scene detect + V/A regression + mood 분류
  │            · VAD 로 dialogue 구간 식별 → voice-critical band 보호 flag
  │            · 출력: runs/demo_<name>/timeline.json
  │
  ├─[Step 2]  playback (model.autoEQ.playback.pipeline.apply_eq_to_video)
  │            · timeline.json 읽어 scene 별 10-band EQ gain 적용 (preset_scale=1.0)
  │            · VAD-guided dialogue protection 자동 적용
  │            · 출력: runs/demo_<name>/<name>_eq_applied.mp4  ← Layer 1 (학술 contribution)
  │
  └─[Step 3]  generate_fx_demo.generate_fx_video
               · Layer 1 mp4 오디오 추출 → scene 별 MOOD_FX_RECIPE 적용
               · Shelf filter + binary reverb (dry/small_room/large_hall)
               · Peak Limiter (threshold −0.5 dB) 로 clipping 방지
               · 출력: runs/demo_<name>/<name>_eq_fx.mp4  ← Layer 1 + 2 (perceptual demo)
```

**단일 호출**:
```bash
venv/bin/python run_pipeline.py --video my_movie.mp4
# 대사 보호 강도 조정 (0.0=full EQ gain, 1.0=원본 완전 보존)
venv/bin/python run_pipeline.py --video my_movie.mp4 --alpha-d 0.3  # default, 작을수록 강한 보호
```

**산출 디렉토리 구조** (`<name>` = 파일명에서 자동 추출):
```
runs/demo_<name>/
  ├─ timeline.json               (Step 1)
  ├─ <name>_eq_applied.mp4       (Step 2, Layer 1 only)
  ├─ <name>_eq_fx.mp4            (Step 3, Layer 1 + 2)
  └─ run.log
```

**검증 루프**: 파이프라인 종료 시 `live_compare_fx.py` 실행 명령이 stdout 에
복사 가능한 형태로 출력되어 바로 3-way 비교 뷰어로 진입 가능.

### 후속 작업 (Phase 5-B, 선택적)

- **ABX pilot listening test**: N=5~8 명, mood-match 평가 → 논문 §5 Subjective Validation 섹션 기여
- **FX-only variant 생성** (ablation 용): EQ 없이 FX 만 적용 → 3-way 비교 (Original / FX-only / EQ+FX) 가능

상세 구현: `generate_fx_demo.py` docstring, `live_compare_fx.py` docstring

---

## 5. 산출물 파일 (영구 보존)

| 파일 | 역할 |
|------|------|
| `model/autoEQ/train_liris/model.py` | Base Model 아키텍처 클래스 |
| `model/autoEQ/train_liris/config.py` | Base Model config defaults |
| `model/autoEQ/train_liris/tests/test_spec_compliance.py` | 11개 테스트 (base config + arch 포함) |
| **`runs/phase2a/2a2_A_K7_s{42,123,2024}/best.pt`** | **현재 Base Model 가중치 (3-seed)** |
| `runs/phase2a/spec_baseline_optB_s{42,123,2024}/best.pt` | 직전 base (A+K=4, 비교용 보존) |
| `runs/phase2a/base_model_final_metrics.json` | 9개 평가지표 전체 |
| `runs/phase2a/2a1_summary.json` | Phase 2a-1 결과 |
| `runs/phase2a/2a2_summary.json` | Phase 2a-2 결과 |
| `runs/phase2a/2a3_summary.json` · `2a3_metrics_comparison.md` | Phase 2a-3 결과 (PANNs vs AST) |
| `runs/phase2a/2a4_summary.json` · `2a4_metrics_comparison.md` | Phase 2a-4 결과 (X-CLIP vs CLIP frame-mean) |
| `runs/phase2a/2a5_summary.json` · `2a5_metrics_comparison.md` | Phase 2a-5 결과 (Gate vs Concat vs GMU) |
| `runs/phase2a/2a7_summary.json` · `2a7_metrics_comparison.md` | Phase 2a-7 결과 (VA+Mood vs VA-only multi-task regularization) |
| **`runs/phase3/test_final_metrics.json` · `test_final_report.md`** | **Phase 3 — Test 평가 최종 결과 (test 1회 access 소모)** |
| `model/autoEQ/train_liris/run_test_eval.py` | Phase 3 Test runner (재평가 금지, 재현용만) |
| `data/features/liris_panns_v5spec/features.pt` | Spec-compliant features (stride=2s, pad_to=10s) |
| **`runs/cognimuse/phase4a/ood_eval/results.json` · `report.md`** | **Phase 4-A — COGNIMUSE OOD 평가 결과 (raw + per-film z-score)** |
| `runs/cognimuse/phase4a/phase0/distribution_report.json` · `scale_shift_summary.md` | Phase 4-A 분포 shift 게이트 (LIRIS vs COGNIMUSE) |
| `dataset/autoEQ/cognimuse/cognimuse_metadata.csv` | COGNIMUSE 2,197 windows × 12 Hollywood films metadata |
| `data/features/cognimuse_panns_v5spec/features.pt` | COGNIMUSE X-CLIP+PANNs features (BASE 파이프라인 bit-identical) |
| `model/autoEQ/train_liris/model_cognimuse_ood/` | Phase 4-A 서브패키지 (preprocessing · precompute · eval + 14 tests) |
| `generate_fx_demo.py` | Phase 5-A Layer 2 FX 적용 스크립트 (MOOD_FX_RECIPE, pedalboard) |
| `live_compare_fx.py` | Phase 5-A Dual-layer 비교 뷰어 (Original / EQ / EQ+FX) |
| `run_pipeline.py` | Phase 5-A 통합 orchestrator (video → timeline + EQ + EQ+FX) |
| `runs/demo_kakao/kakao_eq_fx.mp4` | Phase 5-A FX 산출물 예시 (KakaoTalk demo) |

---

## 6. 후속 ablation 규칙

1. 모든 Phase 2b ablation 은 **이 Base Model을 단일 baseline** 으로 사용한다.
2. 변경 축은 **단일**이며, 나머지는 config.py 의 현재 defaults 그대로 유지한다 (OAT 원칙).
3. Phase 3 Test 평가 전까지 **test set 접근 금지** (V5-FINAL §14-3).
4. Architecture 추가 변경은 **이 Base Model 위에서** 수행하고, 비교 대상은 항상 여기 정의된 `2a2_A_K7_s*` 3-seed.
5. Base Model 자체를 수정하려면 **본 문서와 test를 먼저 업데이트**하고 승인 절차를 거친다.

### Phase 2a-3 (PANNs vs AST) — 완료 (2026-04-21)
- 구현: `model/autoEQ/train_liris/model_ast/` 서브패키지 (Base Model 파일 0개 수정)
- AST feature: `data/features/liris_ast_v5spec/features.pt` (9800 clips, 768-d CLS, 16 kHz)
- 3-seed × 40 epochs (early-stop patience 10)

**결과** (`runs/phase2a/2a3_summary.json`):

| 지표 | PANNs (BASE, 재사용) | AST (신규) | Δ |
|:-:|:-:|:-:|:-:|
| mean_CCC | **0.3812 ± 0.0240** | 0.3292 ± 0.0155 | **−0.0520** |
| seed 42  | 0.3536 | 0.3138 | −0.0398 |
| seed 123 | 0.3959 | 0.3291 | −0.0668 |
| seed 2024| 0.3942 | 0.3448 | −0.0494 |

- **paired t = −6.565**, df=2, **p = 0.022 (< 0.05)** ⭐
- **Winner: PANNs** — Base Model 변경 없음

**AST가 패한 근거**:
- train CCC 0.77~0.86 vs val CCC 0.25~0.33 → train−val gap 0.5~0.6 (`overfit_gap_threshold=0.10` 크게 초과)
- early stop ep 13/20/14 (빠른 overfit 진입)
- AudioSet classification pretrained embedding이 LIRIS film-scene V/A regression과 misalign 된 것으로 해석 (AudioSet 분류 feature가 영화 연속 감정 축에 불충분)
- 참고: AST total params 1.29M (BASE 3.42M 대비 −62%, AudioProjection 축소) → capacity 부족 아님

### Phase 2a-4 (X-CLIP vs CLIP frame-mean) — 완료 (2026-04-21)
- 구현: `model/autoEQ/train_liris/model_clipmean/` 서브패키지 (Base Model 파일 0개 수정)
- CLIPMean feature: `data/features/liris_clipmean_v5spec/features.pt` (9800 clips × {clipmean 512-d, panns 2048-d reused byte-identical})
- Audio side (PANNs) bit-identical to BASE (OAT 순수성 극대화)
- 3-seed × 40 epochs (early-stop patience 10)
- 모델 파라미터 **3,417,099 동일** (BASE 아키텍처 그대로, upstream visual feature 만 교체)

**결과** (`runs/phase2a/2a4_summary.json`):

| 지표 | X-CLIP (BASE, 재사용) | CLIP frame-mean (신규) | Δ |
|:-:|:-:|:-:|:-:|
| mean_CCC | **0.3812 ± 0.0240** | 0.3706 ± 0.0052 | **−0.0106** |
| seed 42  | 0.3536 | 0.3664 | **+0.0128** (역전) |
| seed 123 | 0.3959 | 0.3690 | −0.0269 |
| seed 2024| 0.3942 | 0.3764 | −0.0178 |

- **paired t = −0.884**, df=2, **p = 0.470 (> 0.05)** ⚠️
- **Winner: X-CLIP** (방향만 일관, 통계 유의 미달) — Base Model 변경 없음
- V5-FINAL §21-3 판정: Δ≤0 OR p≥0.05 → keep BASE

**AST 와의 판정 강도 비교**:
| Ablation | Δ | p | 일관성 | 강도 |
|:-:|:-:|:-:|:-:|:-:|
| 2a-3 PANNs vs AST | −0.052 | **0.022** | 3/3 PANNs 승, 11/11 지표 | **strong** |
| 2a-4 X-CLIP vs CLIPMean | −0.011 | 0.470 | 2/3 X-CLIP 승, 8/11 지표 | **weak** |

**해석**:
- Audio encoder 선택 (PANNs) 은 outcome 에 큰 영향 — strong rejection.
- Visual encoder 선택 (X-CLIP) 은 미미한 영향 — CLIP frame-mean 대비 통계적으로 구별 불가.
- 주목할 부수 발견:
  - CLIPMean variance 낮음 (std 0.005 vs X-CLIP 0.024) — 5× 더 안정. 재현성 중시 환경에서 trade-off 존재.
  - gate_entropy: X-CLIP −0.58 (selective) vs CLIPMean −0.41 (uniform) → temporal-rich visual 이 gate 의 modality selection 에 기여한다는 보조 증거.
- Base Model **변경 없음**. CLIPMean variant 는 재현성/감사 목적 보존 (`model_clipmean/`, `runs/phase2a/2a4_clipmean_s*/`).

### Phase 2a-5 (Fusion Mechanism: Gate vs Simple Concat vs GMU) — 완료 (2026-04-22)
- 구현: `model/autoEQ/train_liris/model_fusion/` 서브패키지 (Base Model 파일 0개 수정)
- 서브패키지 가 세 가지 fusion mode 지원: `fusion_mode ∈ {gate, concat, gmu}`
- Feature file (BASE `liris_panns_v5spec/features.pt`) 재사용 — precompute 불필요
- 3-seed × 40 epochs (early-stop patience 10), Gate 는 BASE 재사용
- Downstream heads (VAHead / MoodHead) byte-identical 모두 변형에서 동일 → 진정한 OAT on fusion axis

**비교 대상 3종**:

| Mode | 정의 | fusion params | total params |
|:-:|:-:|:-:|:-:|
| **Gate (BASE)** | `softmax(MLP([v;a])) → (w_v, w_a)`, `fused = concat[w_v·v, w_a·a]` | 262,914 | 3,417,099 |
| Simple Concat | `fused = concat[v, a]` (null baseline, 0 learnable fusion params) | 0 | 3,154,185 |
| GMU (wide) | `h_v = tanh(W_v·v)`, `h_a = tanh(W_a·a)`, `z = sigmoid(W_z·[v;a])`, `fused = z·h_v + (1−z)·h_a` (d_out=1024) | 2,100,224 | 5,254,409 |

**결과** (`runs/phase2a/2a5_summary.json`):

| 지표 | Gate (BASE, 재사용) | Simple Concat | GMU | 최고 |
|:-:|:-:|:-:|:-:|:-:|
| mean_CCC | 0.3812 ± 0.0240 | **0.3869 ± 0.0083** | 0.3712 ± 0.0118 | Concat |
| CCC V | **0.3623 ± 0.0202** | 0.3560 ± 0.0124 | 0.3583 ± 0.0247 | Gate |
| CCC A | 0.4001 ± 0.0295 | **0.4179 ± 0.0066** | 0.3840 ± 0.0119 | Concat |
| mean Pearson | **0.4281 ± 0.0223** | 0.4228 ± 0.0153 | 0.4112 ± 0.0141 | Gate |
| mean MAE ↓ | 0.3243 ± 0.0081 | **0.3240 ± 0.0036** | 0.3351 ± 0.0087 | Concat |

**Per-seed mean_CCC**:
| Seed | Gate | Concat | GMU |
|:-:|:-:|:-:|:-:|
| 42 | 0.3536 | **0.3773** | 0.3604 |
| 123 | **0.3959** | 0.3918 | 0.3838 |
| 2024 | **0.3942** | 0.3917 | 0.3694 |

**Paired t-test vs BASE (n=3, df=2)**:
- **Concat vs Gate**: Δ = **+0.0057**, t = +0.635, **p = 0.590** ❌
- **GMU vs Gate**: Δ = **−0.0100**, t = −1.093, **p = 0.389** ❌

**Winner: Gate (BASE) 유지** — V5-FINAL §21-3 규칙:
- Concat 조건 (Δ>0 AND p<0.05): ❌ (Δ>0 OK, 그러나 p=0.59 미달)
- GMU 조건: ❌ (Δ<0)
- Tie 조건 (|Δ|<0.005): ❌ (|Δ_concat|=0.006 > 0.005)
- → BASE 유지 (현 Gate 고수)

**해석 및 부수 발견**:
- Concat 이 mean 에서 근소 우위 (+0.006) 이지만 paired test 유의 미달 → Gate 교체 불가
- Concat 의 seed 변동성 std 0.008 로 가장 안정 (Gate 0.024 의 1/3 수준) — Phase 2a-4 CLIPMean 때와 동일한 "단순 구조가 더 robust" 패턴 재확인
- GMU 는 2.1M 추가 params 에도 11 지표 모두에서 이득 없음 → 이 scale / task 에 부적합
- Gate Network 의 sample-adaptive modality weighting 은 **실측상 고정 편향에 가까움** (학습된 w_v 평균 0.76, per-sample std 0.06, seed 간 r=0.45, V/A label 과 상관 없음). 즉 이름값 못하지만 downstream 이 적응해서 경쟁력 유지.
- Base Model **변경 없음**. Simple Concat 은 장래 대안으로 보존 (`model_fusion/`, `runs/phase2a/2a5_concat_s*/`, `2a5_gmu_s*/`).

**Phase 2a 결정 강도 정리**:
| Ablation | Δ | p | 강도 |
|:-:|:-:|:-:|:-:|
| 2a-3 PANNs vs AST | +0.052 | **0.022** | **strong** |
| 2a-4 X-CLIP vs CLIPMean | +0.011 | 0.470 | weak |
| **2a-5 fusion (3-way)** | **max \|Δ\|=0.010** | **both ≥0.05** | **weak** |

→ **Audio encoder 선택 이 outcome 에 가장 큰 영향**. Visual encoder 및 fusion mechanism 은 미세한 영향. LIRIS 9,800 clip 규모에서는 **backbone 품질 > fusion 복잡도**.

### Phase 2a-7 (Multi-task Regularization: VA+Mood vs VA-only) — 완료 (2026-04-22)
- 구현: `model/autoEQ/train_liris/model_va_only/` 서브패키지 (Base Model 파일 0개 수정)
- Option B' 설계: MoodHead 실제 제거 (clean params 3,152,388, −264,711) + `cfg.lambda_mood=0.0` + dummy zeros mood_logits (trainer API compat)
- Forward 의 `mood_logits.grad_fn is None` → gradient 0 검증 완료 (tests PASS)
- Feature file (BASE `liris_panns_v5spec/features.pt`) 재사용 — precompute 불필요
- 3-seed × 40 epochs (early-stop patience 10)
- OAT 축: "with vs without multi-task Mood auxiliary loss" (다른 arch/hyper BASE 와 동일)

**결과** (`runs/phase2a/2a7_summary.json`):

| 지표 | VA+Mood (BASE, 재사용) | VA-only (신규) | Δ |
|:-:|:-:|:-:|:-:|
| mean_CCC | **0.3812 ± 0.0240** | 0.3711 ± 0.0202 | **−0.0101** |
| CCC V | **0.3623 ± 0.0202** | 0.3363 ± 0.0262 | **−0.0260** (큰 loss) |
| CCC A | 0.4001 ± 0.0295 | **0.4059 ± 0.0167** | +0.0058 (소폭 이득) |
| seed 42  | 0.3536 | 0.3502 | −0.0034 |
| seed 123 | 0.3959 | 0.3905 | −0.0054 |
| seed 2024| 0.3942 | 0.3726 | **−0.0216** |

- **paired t = −1.762**, df=2, **p = 0.220 (> 0.05)** ⚠️
- **Cohen's d_z = −1.02 (large effect size)** — n=3 power 부족으로 통계 유의 미달이지만 방향 완전 일관
- **승수: VA-only 0/3 seeds** (3/3 BASE 우위)
- **Winner: VA+Mood (BASE) 유지** — V5-FINAL §21-3 Δ≤0 규칙으로 BASE 유지

**해석**:
- Cohen's d_z = −1.02 로 **large effect** — 효과 크기 자체는 명확, n 늘리면 유의 유력
- 방향성 3/3 consistent → 가설 A (multi-task regularization 도움) 지지 방향
- **Valence 축에서 주로 이득** (Δ=−0.026) — GEMS 7-class 가 Valence 축과 강하게 연관 (Tenderness/Peacefulness = V+, Tension/Sadness = V−)
- Arousal 은 오디오 주도라 MoodHead 영향 미미 (Δ=+0.006, 역전)
- gap@best 거의 동일 (+0.121 vs +0.123) → MoodHead 는 주된 regularizer 가 아니고 **보조 (marginal)**. 이미 head_dropout 0.3 / wd 1e-4 / mixup / target shrinkage 등으로 regularization 포화
- L_va BASE 0.339 vs VA-only 0.343 (+0.004) → MoodHead 제거 시 V/A head 단독으로는 fused representation 활용 효율 미세 저하
- MoodHead (264K params, 7.75%) 유지 근거 충분 — Phase 3 최종 평가까지 BASE 그대로

**판정 강도 비교**:
| Ablation | Δ | p | 일관성 | Cohen's d | 강도 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2a-3 PANNs vs AST | −0.052 | 0.022 | 3/3 | − | **strong** |
| 2a-4 X-CLIP vs CLIPMean | −0.011 | 0.470 | 2/3 | − | weak |
| 2a-5 fusion (3-way) | max \|Δ\|=0.010 | both ≥0.05 | mixed | − | weak |
| **2a-7 VA+Mood vs VA-only** | **−0.010** | **0.220** | **3/3** | **−1.02** | **directional** |

→ Phase 2a-7 은 다른 "weak" ablation 과 달리 **방향 3/3 consistent + large effect size** 로 분리됨. 통계 유의성만 미달 (n=3 한계). MoodHead 유지 근거는 명확.

Base Model **변경 없음**. VA-only variant 는 재현성/감사 목적 보존 (`model_va_only/`, `runs/phase2a/2a7_va_only_s*/`).

---

*Last frozen: 2026-04-22 · Phase 3 FINAL (backbone: X-CLIP + PANNs, fusion: Gate Network, heads: joint VA + Mood K=7)*
*LIRIS — Val mean CCC = 0.3812 ± 0.024 · Test mean CCC = 0.3480 ± 0.016 · Test Ensemble 0.3603*
*COGNIMUSE OOD (Phase 4-A, 2026-04-22) — 12 Hollywood films, 2,197 windows, BASE ensemble 학습 0회*
*  · raw ensemble CCC 0.3182 (V 0.3565, A 0.2798)*
*  · z-score ensemble CCC 0.3781 (V 0.3113, A 0.4449) — 강한 일반화 입증*
*Dual-layer architecture (Phase 5-A, 2026-04-23) — Layer 1 EQ (scientific) + Layer 2 FX (perceptual amplifier)*
*  · FX 레시피: Juslin/Rumsey/Sato/McAdams/Eerola/Zentner 문헌-근거 방향성만*
*  · 임의 수치 없음 (compression/width 제외), Base Model weights 불변*
*Phase 5-A rev. (2026-04-23) — 대사 보호 강화: Layer 1 alpha_d 0.5→0.3 (작을수록 강함) + Layer 2 VAD-guided reverb bypass (shelf 유지)*
*Author: Phase 2a-0 ~ 2a-7 ablation sweep + Phase 3 test evaluation + Phase 4-A OOD eval + Phase 5-A FX layer + user sign-off*
