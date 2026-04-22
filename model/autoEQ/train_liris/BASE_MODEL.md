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
*Test mean CCC = 0.3480 ± 0.016 · Ensemble 0.3603 · Val mean CCC = 0.3812 ± 0.024*
*Author: Phase 2a-0 ~ 2a-7 ablation sweep + Phase 3 test evaluation + user sign-off*
