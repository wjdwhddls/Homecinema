# LIRIS-ACCEDE 전환 플랜 V5-FINAL (실측 GT 분포 반영 + 최종 목적 명시)

## Context

- V3.2 명세로 LIRIS 학습 계획 → 데이터 확보 불가로 V3.3 축소(CCMovies 9편+pseudo-label) 전환 완료.
- **2026-04-20 LIRIS-ACCEDE 패키지 도착/진행 상황**:
  - ✅ `LIRIS-ACCEDE-data.zip` (9,800 clip, 필수) — 완료
  - ✅ `LIRIS-ACCEDE-annotations.zip` — 완료 (ACCEDEranking·ACCEDEsets·Readme)
  - ✅ `LIRIS-ACCEDE-features.zip` — 완료 (선택적, 본 플랜 미사용)
  - ⏳ `LIRIS-ACCEDE-movies.zip` (30.4 GB, 175 원본 영화) — **다운로드 중 21.0/30.4 GB, 약 2일 남음**. Phase 3 full movie 재생 검증 시에만 필요. **Phase 0/1/2는 data.zip + annotations.zip만으로 진행 가능**.
- **사용자 최종 목적 (V5-final 신규 명시, §21 참조)**: **학습 단계에서 여러 모델 변형을 성능 비교하여 최종 모델을 결정**. 먼저 **베이스라인(V3.2 원래 설계)을 실제로 돌려본 결과를 보고 후속 ablation 방향을 결정**. OAT(One-At-a-Time) 순차 진행 + 각 ablation 단계 사용자 sign-off.

## 0. 변경 이력

### V5 → V5-FINAL (2026-04-20)
1. **§2-5 & §19 실측 GT 분포 반영** — 9,800 clip 전체에 7-centroid Euclidean 매핑 집계 완료. V5에서 예상한 "JA/TE/PE/WO 4개 class under-represented"는 **오류**로 교정. 실측: **JoyfulActivation 1개만 0% 도달**, 나머지 6/7 class는 충분 (Peacefulness 30.77%, Sadness 27.18%, Tenderness 18.47%, Tension 11.98%, Power 5.81%, Wonder 5.80%).
2. **§19-4 XX% placeholder 교체** — 실측 숫자로 완전히 채움. Phase 3 예측 분포 vs GT 분포 비교는 **KL → JS-divergence**로 전환 (symmetric + bounded [0, log₂]).
3. **§8 게이트 판정 재검토** — 엄격 해석(1개 실패 → K=4) 유지하되, 실측으로 "6/7 class는 학습 가능"이 확인되어 **K=7 옵션의 실용 가능성 유지**. Phase 2a-2에서 K=4 vs K=7 실제 비교 의미 회복.
4. **§21 신설 — 최종 모델 선정 전략** (사용자 목적 직접 명시). 베이스라인 우선 실행 + 사용자 sign-off 게이트 + 모델 선정 기준.
5. **§22 신설 — 다운로드 대기 중 병행 가능 작업** (movies.zip 대기 2일 동안 Phase 0 작업 + 코드 scaffolding 병행).
6. **§8 전략 B 재실행 주석** — Phase 2a-1에서 V/A 정규화 전략 B(min-max stretch)가 승자가 될 경우 K=7 게이트 재실행 필요 명시.
7. **Context 업데이트** — movies.zip 다운로드 상태 + Phase 착수 조건 + 최종 목적 반영.

### V4 → V5 (참고)
- "in-domain OOD" → "in-distribution (ID) hold-out" 교정
- Baveye 2014 → 2015 IEEE TAC 통일
- Bootstrap Phase 3 전용, 21 run (2b-6 중복 제거)
- §8 순서 근거 이원화

### V1~V4 (참고 유지)
- V4: COGNIMUSE 배제, §19 EQ 커버리지 학술 프레이밍, L_va_mse/L_va_ccc 독립 로깅
- V3: 9,800 clip 실측, K=4 default 전환, Phase 2 → 2a/2b, film-level bootstrap, overfit monitor
- V2: Lexicographic/Target Shrinkage 용어, CCC primary, V3.2 모델 내부 복귀

---

## 1. 사용자 확정 결정사항

1. Congruence Head + Negative Sampling: 미복원
2. Mood Head: **K=4 quadrant default** (§8 게이트, 실측상 JA만 실패 → strict K=4), §21 모델 비교에서 K=7 재실측
3. 코드 구조: 새 `model/autoEQ/train_liris/` 폴더
4. 베이스 모델: X-CLIP + PANNs CNN14 + Gate Network + Intermediate Fusion + Multi-task
5. 평가 프레이밍: CCC 단독 primary (AVEC), Pearson/MAE diagnostic
6. Early stopping: Lexicographic `(mean_CCC, mean_Pearson, −mean_MAE)`, patience=10
7. 평가 범위: LIRIS test 80 films = **ID hold-out**, 외부 OOD는 Future Work
8. **최종 목적 (V5-final 신규)**: 모델 성능 비교로 최종 모델 결정. **베이스라인 우선 실행 후 사용자 sign-off**로 후속 ablation 방향 확정 (§21)

---

## 2. LIRIS-ACCEDE 실측 요약

### 2-1. V/A scalar 실측 범위
| 축 | 이론 (README) | 실제 | (raw−3)/2 변환 후 |
|---|---|---|---|
| valenceValue | 1~5 | **1.33 ~ 3.59** | **[−0.84, +0.30]** |
| arousalValue | 1~5 | 1.32 ~ 4.54 | [−0.84, +0.77] |

### 2-2. Variance 분포
- valenceVariance: p50=0.108, p75=0.117, max=0.164
- arousalVariance: p50=0.150, p75=0.164, max=0.218
- 확정 threshold: `v_thr=0.117, a_thr=0.164, AND` → 약 6~10% 발동

### 2-3. 사분면 분포
HVHA 11.0% · HVLA 40.2% · LVHA 18.6% · LVLA 30.2%

### 2-4. LIRIS 공식 split
| split | 영화 | clip | 역할 |
|---|---|---|---|
| learning (set=1) | 40 | 2,450 | train |
| validation (set=2) | 40 | 2,450 | val (early stop) |
| test (set=0) | 80 | 4,900 | **test (ID hold-out, unseen films)** |

### 2-5. 🆕 K=7 GEMS centroid 실측 GT 분포 (V5-final 신설, 9,800 clip 전체)

| Mood | Centroid (V, A) | Count | Ratio | ≥ 1%? |
|---|---|---|---|---|
| Peacefulness | (+0.5, −0.5) | 3,015 | **30.77%** | ✅ |
| Sadness | (−0.6, −0.4) | 2,664 | **27.18%** | ✅ |
| Tenderness | (+0.4, −0.2) | 1,810 | **18.47%** | ✅ |
| Tension | (−0.6, +0.7) | 1,174 | **11.98%** | ✅ |
| Power | (+0.2, +0.8) | 569 | **5.81%** | ✅ |
| Wonder | (+0.5, +0.3) | 568 | **5.80%** | ✅ |
| **Joyful Activation** | (+0.7, +0.6) | **0** | **0.00%** | **❌** |

**핵심 함의** (V5 예상 교정):
- V5는 "JA/TE/PE/WO 4개 under-represented"로 추정했으나, 실측상 **Peacefulness/Tenderness가 오히려 풍부**. 이는 LIRIS의 HVLA 사분면 40.2% 비중이 PE(30.77%) + TE(18.47%)로 분할되어 흡수된 결과.
- **JoyfulActivation 1개만 0%**: v_norm max +0.30 < JA centroid V=+0.7로 구조적 도달 불가.
- Data sufficiency 엄격 판정(min ≥ 1%) → K=4 fallback. 다만 실질적으로 **6/7 class는 학습 가능**하므로 §21 모델 비교에서 K=7 옵션의 정보 가치 보존.

---

## 3. 구성 요약

| 계층 | 선택 |
|---|---|
| Visual encoder | X-CLIP base-patch32 (frozen, 512-d) |
| Audio encoder | PANNs CNN14 (frozen, 2048-d) + Linear Projection 2048→512 |
| Fusion | Gate Network (1024→256→2 softmax) + Intermediate Fusion → 1024-d |
| VA Head | 1024 → 256 → 2 |
| **Mood Head** | **K=4 (default, §8 strict gate)** · Phase 2a-2에서 K=7 실측 비교 |
| Congruence Head | 없음 |
| Modality Dropout | p=0.05 default (Phase 2b-6 ablation {0.05, 0.10, 0.15}) |
| Feature Noise | σ=0.03 대칭 |
| Quadrant Mixup | prob=0.5, α=0.4 |
| Target Shrinkage | per-axis p75 (v=0.117, a=0.164), AND, ε=0.05 |
| V/A loss | CCC hybrid w=0.3 (2b-3 ablation {0, 0.3, 1.0}) |
| Mood loss | CE, λ_mood=0.3 (2b-1 ablation) |
| Gate entropy | λ_ent=0.05 (2b-2 ablation {0, 0.05, 0.1}) |
| Early stopping | Lexicographic (mean_CCC, mean_Pearson, −mean_MAE), patience=10 |
| Primary metric | mean_CCC (AVEC) |
| Bootstrap CI | Phase 3 최종 평가 전용 (film-level cluster resampling) |
| Overfit 감지 | train−val mean_CCC gap > 0.10 |
| Loss 로깅 | total, L_va, L_va_mse, L_va_ccc, L_mood, L_gate_entropy |

---

## 4. V3.2 ↔ V3.3 ↔ V5-FINAL 매트릭스

| 항목 | V3.2 | V3.3 | V5-final | 비고 |
|---|---|---|---|---|
| 학습 데이터 | LIRIS 9,800 | CCMovies 1,289 | LIRIS 9,800 | V3.2 복귀 |
| Split | 120/20/20 | LOMO 9-fold | LIRIS 공식 40/40/80 | — |
| 평가 범위 | COGNIMUSE OOD | — | LIRIS test 80 films (ID hold-out) | V4 교정 |
| V/A range | [-1,+1] | [-1,+1] | (v-3)/2, v∈[-0.84,+0.30] | §2-1 |
| **GT 7-centroid 분포** | — | — | **6/7 class ≥ 5.8%, JA 0%** | **§2-5 (V5-final 실측)** |
| Per-sample std | — | ensemble std | valenceVariance/arousalVariance p75 AND | — |
| Audio encoder | PANNs | AST | **PANNs** | 사용자 |
| Fusion | Gate + Intermediate | GMU | **Gate + Intermediate** | 사용자 |
| Mood Head | K=7 | K=4 | **K=4 default** · Phase 2a-2 K=7 비교 | §8 + §21 |
| V/A loss | MSE | CCC hybrid | CCC hybrid w=0.3 + 2b-3 ablation | — |
| Gate entropy | λ=0.05 | N/A | λ=0.05 + 2b-2 ablation | — |
| Augmentation | 명시 없음 | 3종 | 4종 + 2b-5 off-ablation | — |
| Modality Dropout | p=0.1 cong만 | p=0.05 전체 | p=0.05 + 2b-6 {0.05,0.10,0.15} | — |
| Early stopping | val CCC | "Pareto" | Lexicographic | V2 |
| Primary metric | CCC | CCC | CCC 단독, Pearson/MAE diagnostic | V2 |
| Bootstrap | — | — | Phase 3 전용, film-level | V5 |
| Overfit monitor | — | — | gap > 0.10 | V3 |
| Loss 로깅 | total | total | total, L_va, L_va_mse, L_va_ccc, L_mood, L_gate_ent | V4 |
| Distribution 비교 | — | — | **JS-divergence (pred vs GT)** | V5-final |
| EQ 프리셋 | 7×10 Biquad | 동일 | 동일 + §19 커버리지 보고 | V4 |

---

## 5. Phase 구조

| Phase | Compute | 문서/분석 | 총 소요 | 착수 조건 |
|---|---|---|---|---|
| **0** | ~40분 | 10분 | ~50분 | data.zip + annotations.zip ✅ (다운로드 완료) |
| **1** | ~4시간 | ~4시간 | ~1일 실작업 | data.zip 해제 완료 |
| **2a** | ~3시간 (4 ablation × 40min) | ~4시간 | ~1일 | Phase 1 완료 + 사용자 sign-off |
| **2b** | ~14시간 (21 run × 40min) | ~8시간 | ~3일 | Phase 2a 완료 + 사용자 sign-off |
| **3** | ~1일 (추론 + §19 JS-div + 청취 A/B) | ~1일 | ~3일 | Phase 2 완료, **movies.zip 완료(원본 영화 full playback 검증)** |

**총 compute: ~22시간 / 총 소요: ~3주 (문서 포함)**

**movies.zip 의존성**: Phase 3 샘플 영화 end-to-end 검증에만 필요. Phase 0/1/2는 data.zip 9,800 clip만으로 진행 가능.

---

## 6. Phase 0 — Sanity Check (~50분)

### 6-1. PANNs 4초 입력 품질 (full clip vs 4s crop)
```python
for clip in sampled_10_from_liris_data:
    full_audio = load(clip, sr=16000)
    crop_4s = center_crop(full_audio, 4.0)
    sims.append(cosine_sim(panns(full_audio), panns(crop_4s)))
```
판정: mean ≥ 0.90 OK / 0.80~0.90 주의 / <0.80 대안

### 6-2. V/A 2D scatter + K=7 centroid GT 분포 (✅ §2-5에 실측 기록 완료)

### 6-3. Variance 분포 histogram + p75 threshold 시각화

산출: `runs/phase0_sanity/{panns_sanity.json, va_scatter.png, gt_centroid_dist.json, variance_dist.json}`

**§2-5 실측이 완료되어 §6-2는 시각화만 남음**. Phase 0 실 compute는 PANNs sanity(§6-1)로 축소.

---

## 7. Phase 1 — LIRIS 데이터 준비 (~1일 실작업)

1. `unzip LIRIS-ACCEDE-{data,annotations}.zip -d dataset/LIRIS_ACCEDE/`
2. 메타데이터 파싱 → `liris_metadata.csv`
3. film_id 추출 (`ACCEDEdescription.xml`의 `<movie>`)
4. 공식 split 적용 + film 중복 assert
5. §8 K=7 게이트 (fail-fast, 실측상 FAIL 예상 → K=4)
6. Feature precompute: X-CLIP + PANNs → `data/features/liris_panns/`
7. 산출: metadata, split, gate result, precompute manifest, distribution_report

---

## 8. K=7 Mood Head 게이트 (V5-final 실측 반영)

### 조건 1 (1분): Data sufficiency — **실측 결과 반영**
- 각 7-class count ≥ 1%
- **V5-final 실측 (§2-5)**:
  - 6/7 PASS (Peacefulness 30.77%, Sadness 27.18%, Tenderness 18.47%, Tension 11.98%, Power 5.81%, Wonder 5.80%)
  - **Joyful Activation 0.00% → FAIL** (v_norm max +0.30 < JA centroid V=+0.7)
- **strict 판정**: 1개라도 실패 → K=4 fallback **확정**
- 비용: 1분 (이미 §2-5에서 완료)

### 조건 2 (30분): Non-interference
- K=7 vs K=4 10-epoch mini-run 대조, `|ΔCCC| ≤ 0.01`
- **V5-final 참고**: strict 판정으로 조건 1에서 이미 FAIL이므로 조건 2 skip. 단 Phase 2a-2에서 **모델 비교 목적**으로 재실행 (§21)
- FAIL 시 K=4 확정

### 조건 3 (30분, 조건 2 run 재활용): Learnability
- val `mood_F1_macro > 14.3%`

### 🆕 전략 B 재실행 주석 (V5-final)
Phase 2a-1에서 V/A 정규화 **전략 B (per-axis min-max stretch)가 승자가 되면 v_norm 범위가 [-1, +1] 전체로 확장**되어 JA centroid가 도달 가능해질 수 있음. 이 경우 §8 게이트를 **재실행**하여 K=7 Data sufficiency 재판정 필수.

### 순서 근거 (이원화)
- **결정력 기준**: Non-interference 1순위 (primary task 저해 치명성)
- **실행 효율 기준**: Data sufficiency 1순위 (1분 확인, 실측상 fail 유력)
- **최종 결정**: 실행 효율 채택 (§2-5 실측으로 확정)

### 게이트 의사코드
```python
# 1. Data sufficiency (이미 §2-5 실측 완료)
if min_class_ratio < 0.01:
    gate_result = "K=4 strict (data_sufficiency FAIL: JA=0%)"
    # 단, §21 모델 비교를 위해 K=7 ablation run은 별도 수행

# 2. Non-interference (Phase 2a-2에서 비교 목적 수행)
# 3. Learnability (조건 2 run 재활용)
```

---

## 9. 구현 파일 구조

### 9-1. 신규 `model/autoEQ/train_liris/`
| 파일 | 내용 |
|---|---|
| `config.py` | TrainLirisConfig (num_mood_classes={4,7}, λ_mood=0.3, λ_gate_entropy=0.05, ccc_hybrid_w=0.3, modality_dropout_p=0.05, feature_noise_std=0.03, mixup_prob=0.5, mixup_alpha=0.4, target_shrinkage_eps=0.05, v_var_threshold=0.117, a_var_threshold=0.164, shrinkage_logic="AND", batch_size=32, lr=1e-4, wd=1e-5, epochs=40, patience=10, grad_clip=1.0, warmup=500, seed=42, use_official_split=True, pad_audio_to_10s=auto) |
| `liris_preprocess.py` | 메타데이터 파싱, V/A 변환, film_id 추출, feature precompute, variance threshold 산출 |
| `dataset.py` | `PrecomputedLirisDataset` + `official_split()` |
| `model.py` | `AutoEQModelLiris` (V3.3 `model_base/model.py` 기반) |
| `losses.py` | `combined_loss_liris` — L_va_mse / L_va_ccc 독립 반환 |
| `trainer.py` | Lexicographic early stopping + Overfit auto-monitor + per-term loss 로깅 |
| `metrics.py` | CCC/Pearson/MAE/RMSE + `film_level_bootstrap_ci` (Phase 3 전용) + `js_divergence` (§19) |
| `run_train.py` | CLI |
| `eval_test.py` | 최종 평가 (bootstrap CI + 7-centroid 예측 분포 집계 + JS-div) |
| `tests/` | 9종 단위 테스트 |

### 9-2. 재사용
- `train_pseudo/dataset.py::va_to_mood` (K=7 GEMS)
- `train_pseudo/model_base/*` (V3.2 baseline 참조)
- `train/encoders.py` (X-CLIP / PANNs utils)
- `infer_pseudo/*` — `VARIANTS` dict에 `liris_base` entry 추가 (1줄 확장)
- `playback/*`, `scripts/eq_response_check.py`

### 9-3. 폐기
- `pseudo_label/*`, CCMovies preprocess, COGNIMUSE preprocess, σ-filter 코드, CCMovies data

### 9-4. infer_pseudo VARIANTS 확장 예시
```python
# model/autoEQ/infer_pseudo/model_inference.py
VARIANTS["liris_base"] = {
    "cfg_module":   "model.autoEQ.train_liris.config",
    "cfg_cls":      "TrainLirisConfig",
    "model_module": "model.autoEQ.train_liris.model",
    "model_cls":    "AutoEQModelLiris",
    "audio_encoder": "panns",
}
```

---

## 10. 평가·모니터링 구현

### 10-1. Film-level cluster bootstrap (Phase 3 전용)
```python
def film_level_bootstrap_ci(preds, targets, film_ids, metric_fn,
                            n_resamples=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    uniq = np.unique(film_ids)
    idx_map = {f: np.where(film_ids == f)[0] for f in uniq}
    results = []
    for _ in range(n_resamples):
        sampled = rng.choice(uniq, len(uniq), replace=True)
        indices = np.concatenate([idx_map[f] for f in sampled])
        results.append(metric_fn(preds[indices], targets[indices]))
    lo, hi = np.percentile(results, [(1-ci)/2*100, (1+ci)/2*100])
    return {'point': metric_fn(preds, targets), 'ci_lo': lo, 'ci_hi': hi}
```
Phase 2는 point estimate + paired t-test.

### 10-2. Overfit auto-monitor
```python
gap = train_mean_ccc - val_mean_ccc
if gap > 0.10:
    log.warning(f"Overfit @ epoch {epoch}: gap={gap:.3f}")
```

### 10-3. Loss term 독립 로깅
```python
return L_total, {
    'loss_total': L_total, 'loss_va': L_va,
    'loss_va_mse': mse_term, 'loss_va_ccc': ccc_term,
    'loss_mood': L_mood, 'loss_gate_entropy': L_ent,
}
```

### 10-4. 🆕 JS-divergence (V5-final §19-4)
```python
from scipy.spatial.distance import jensenshannon

def js_divergence_mood_dist(pred_dist, gt_dist, base=2):
    """Symmetric, bounded [0, 1] when base=2. Prefer over KL."""
    return jensenshannon(pred_dist, gt_dist, base=base)
```
KL 대신 JS 선택 이유: symmetric (어느 쪽이 reference인지 무관), bounded [0, 1] (해석 용이), 0 division free.

---

## 11. Phase 2a — 근본 설계 ablation (OAT, ~1일)

**Baseline 2a-0** (사용자 목적에 따라 먼저 실행): **V3.2 원래 설계**
- V/A 정규화: 전략 A `(v-3)/2`
- Mood Head: K=4 quadrant (§8 strict)
- Audio encoder: PANNs CNN14
- Fusion: Gate Network + Intermediate Fusion
- 기본 하이퍼 (§3)

| 단계 | 변경 축 | 조건 | 사용자 sign-off |
|---|---|---|---|
| **2a-0** | **Baseline (V3.2 원래 설계)** | — | **실측 후 결정** |
| 2a-1 | V/A 정규화: 전략 A vs B (min-max stretch) | 2a-0 대비 | 승자 확정 (전략 B 승 시 §8 게이트 재실행) |
| 2a-2 | Mood Head: K=4 vs K=7 | 2a-1 승자 baseline | 승자 확정 |
| 2a-3 | Audio encoder: PANNs vs AST | 2a-2 승자 baseline | 승자 확정 |
| 2a-4 | Fusion: Gate vs GMU (선택적) | 2a-3 승자 baseline | 승자 확정 |

각 단계 **paired t-test + point estimate + 사용자 sign-off**.

---

## 12. Phase 2b — 하이퍼 튜닝 ablation (OAT, ~3일, 21 run)

Baseline 2b-0 = Phase 2a 최종 승자.

| 단계 | 변경 축 | 값 | Run 수 |
|---|---|---|---|
| 2b-1 | λ_mood | {0, 0.1, 0.3, 0.5} | 4 |
| 2b-2 | λ_gate_entropy | {0, 0.05, 0.1} | 3 |
| 2b-3 | ccc_hybrid_w | {0, 0.3, 1.0} | 3 |
| 2b-4 | Batch size | {32, 64} | 2 |
| 2b-5 | 증강 off-ablation | 6 run | 6 |
| 2b-6 | Modality Dropout p | {0.05, 0.10, 0.15} (p=0 = 2b-5-a 공유) | 3 |

### 2b-5 증강 off-ablation 테이블
| Run | Modality Dropout | Feature Noise | Quadrant Mixup | Target Shrinkage |
|---|---|---|---|---|
| 2b-5-a | OFF | ON | ON | ON |
| 2b-5-b | ON | OFF | ON | ON |
| 2b-5-c | ON | ON | OFF | ON |
| 2b-5-d | ON | ON | ON | OFF |
| 2b-5-e (ALL_OFF) | OFF | OFF | OFF | OFF |
| 2b-5-f (ALL_ON, reference) | ON | ON | ON | ON |

**총 21 run ≈ 14시간 compute**

---

## 13. Phase 3 — 추론/재생 + 최종 평가 (~3일)

1. Phase 2 최종 모델 확정 → `infer_pseudo::VARIANTS["liris_base"]` 추가
2. **LIRIS test set (80 unseen films, 4,900 clips) 최종 평가** — film-level bootstrap 95% CI
3. **§19 커버리지 분석** — 예측 7-centroid 분포 vs GT 분포 **JS-divergence** 계산
4. 샘플 영화 1편 end-to-end (analyze → timeline.json → playback → remux). **movies.zip 필요**
5. 청취 A/B: α_d ∈ {0.3, 0.5, 0.7}
6. 최종 보고서 작성

---

## 14. Verification

### 14-1. 코드/데이터 무결성
- `pytest model/autoEQ/train_liris/tests/ -v` (9/9 PASS)
- `liris_preprocess --verify` (split 무결성, V/A round-trip, variance threshold 발동률 assert)

### 14-2. 학습 sanity (2 epoch mini-run)
- L_total + 4개 loss term 독립 감소
- grad_norm head 간 10배 이내
- train/val CCC gap monitor 작동

### 14-3. Primary metric 목표
**Phase 2 (point estimate + paired t-test)**:
- LIRIS val: mean_CCC ≥ 0.30 (baseline) / ≥ 0.45 (stretch)

**Phase 3 (film-level bootstrap 95% CI)**:
- LIRIS test: mean_CCC ≥ 0.28 + CI 보고
- §19: JS-divergence (예측 vs GT 7-centroid 분포)

### 14-4. End-to-end
- 샘플 영화 1편 `analyze → timeline.json → playback → remuxed.mp4` 무오류

---

## 15. 리스크 및 트레이드오프

| # | 리스크 | 대응 |
|---|---|---|
| 1 | Valence positive 편향 → JA만 도달 불가 (§2-5 실측) | K=4 default + §19 학술 보고 + §21 K=7 비교 |
| 2 | Variance threshold | per-axis p75 AND (~6-10% 발동) |
| 3 | PANNs 4초 입력 | Phase 0 sanity |
| 4 | Overfit (183만 param / 9,800 windows) | Overfit auto-monitor + 증강 4종 + patience 10 |
| 5 | 외부 OOD 평가 부재 (ID hold-out만) | LIRIS test 80 films로 within-distribution generalization. Cross-dataset OOD는 Future Work |
| 6 | Congruence 미복원 | 보고서 "실증적 단순화" 방어 |
| 7 | LIRIS split 1:1:2 skew | `--use_full_learning_set` 옵션 |
| 8 | 증강 효과 약화 | Phase 2b-5 off-ablation |
| 9 | EQ 프리셋 커버리지 (JA만) | §19 학술 보고 + JS-div 정량 |
| 10 | **movies.zip 2일 다운로드** | **Phase 0~2는 data.zip만으로 진행, Phase 3 진입 전 완료 확인** (§22) |

---

## 16. 결론

**"V3.3 학습 프로세스 외곽 × V3.2 모델 내부 × LIRIS 실측 × V5 학술 정확성 × V5-final GT 실측 반영 + 최종 목적 명시"** 하이브리드.

V5-final 핵심 교정:
1. **§2-5 실측 GT 분포 반영** — V5 예상(4개 under-represented) 교정, **JA 1개만 0%**
2. **§19-4 XX% 실측 값으로 교체**
3. **§8 게이트 판정 실측 반영** (strict K=4 확정, §21 K=7 비교로 정보 가치 보존)
4. **KL → JS-divergence** (§19-4)
5. **§21 최종 모델 선정 전략 신설** (사용자 목적 직접 명시)
6. **§22 다운로드 대기 병행 작업 신설**
7. **§8 전략 B 재실행 주석**

진행 순서: Phase 0 PANNs sanity (30분) → Phase 1 (~1일) → **Phase 2a-0 Baseline (V3.2 원설계) 실행 + 사용자 sign-off** → Phase 2a-1~4 OAT → Phase 2b 하이퍼 → Phase 3 (movies.zip 완료 시점).

---

## 17. (V3부터 유지) Mood centroid 좌표 + LIRIS 도달성 (V5-final 실측 반영)

| Mood | V | A | GT 분포 | 도달성 |
|---|---|---|---|---|
| Peacefulness | +0.5 | −0.5 | **30.77%** | ✅ (HVLA 흡수) |
| Sadness | −0.6 | −0.4 | 27.18% | ✅ |
| Tenderness | +0.4 | −0.2 | **18.47%** | ✅ (HVLA 흡수) |
| Tension | −0.6 | +0.7 | 11.98% | ✅ |
| Power | +0.2 | +0.8 | 5.81% | ✅ (경계) |
| Wonder | +0.5 | +0.3 | **5.80%** | ✅ (HVLA 경계) |
| **Joyful Activation** | +0.7 | +0.6 | **0.00%** | ❌ **(유일한 under-represented)** |

---

## 18. (V3부터 유지) 데이터 증강 4종 상세

### 증강 1: Modality Dropout (model.forward) — p=0.05 default
### 증강 2: Feature Noise (model.forward) — σ=0.03 대칭
### 증강 3: Quadrant Mixup (collate_fn, prob=0.5) — 같은 사분면 쌍, λ~Beta(0.4,0.4) shrink
### 증강 4: Target Shrinkage (collate_fn) — `(v_var>0.117) & (a_var>0.164)` AND 조건, ε=0.05

### 순서
DataLoader → collate_fn [Mixup → Target Shrinkage] → model.forward [Modality Dropout → Feature Noise] → Gate → Fusion → Heads → Loss

---

## 19. 🆕 EQ 프리셋 커버리지 한계 — 학술적 정직 보고 (V5-final 실측 반영)

### 19-1. 문제 진술 (실측 기반 재작성)

LIRIS-ACCEDE Discrete 9,800 clip 실측(§2-1)에서 valenceValue는 이론 [1, 5]와 달리 [1.33, 3.59]에 분포. 이는 Baveye et al. (2015, IEEE TAC) "LIRIS-ACCEDE: A Video Database for Affective Content Analysis"의 crowdsourced pairwise ranking → affective rating 회귀 과정의 수치적 특성에 기인.

**7-centroid Euclidean 실측 결과** (§2-5):
- 6/7 class는 도달 가능 (Peacefulness 30.77%, Sadness 27.18%, Tenderness 18.47%, Tension 11.98%, Power 5.81%, Wonder 5.80%)
- **Joyful Activation 1개 centroid만 0.00% 도달** (v_norm max +0.30 < JA V=+0.7)

**V5에서 추정한 "JA/TE/PE/WO 4개 class under-represented"는 교정**. HVLA 사분면 40.2%가 PE+TE centroid로 대거 흡수되어 6개 class의 실질 학습 가능성 확보됨.

### 19-2. 연쇄 영향 (학습 → 추론 → EQ)

```
[학습] LIRIS에서 JA 영역 샘플 0개
  → 모델이 JA centroid를 명시적으로 학습 못 함

[추론] 다양한 영화 씬 입력
  → Joyful Activation 씬에도 V_pred ≤ +0.30
  → 7-centroid Euclidean 시 JA 대신 Peacefulness/Tenderness 할당

[EQ] JA 씬에 Peacefulness EQ 적용
  → 감독 의도(Joyful) 완화 효과
```

**단, 다른 6개 mood는 정상 동작 예상** — §21 Phase 2a-0 baseline 실측으로 확인.

### 19-3. 대응 — 은폐 대신 학술적 발견으로 보고

본 연구는 JA centroid 커버리지 부재를 **LIRIS-ACCEDE 데이터셋의 단일 구조적 특성**으로 정직 보고. V3.2 명세 7 mood EQ 프리셋 자체는 완전 보존 (JA EQ 프리셋도 V3.2 §6-4 그대로).

### 19-4. 정량 측정 — Phase 0 실측 GT + Phase 3 예측 JS-divergence (V5-final 완성)

**Step 1 (§2-5 완료)** — Ground truth K=7 분포:

| Mood | Count | GT Ratio |
|---|---|---|
| Peacefulness | 3,015 | 30.77% |
| Sadness | 2,664 | 27.18% |
| Tenderness | 1,810 | 18.47% |
| Tension | 1,174 | 11.98% |
| Power | 569 | 5.81% |
| Wonder | 568 | 5.80% |
| Joyful Activation | 0 | 0.00% |

**Step 2 (Phase 3)** — 학습된 모델의 LIRIS test (80 unseen films, 4,900 clips) 추론 분포 vs GT 분포 **JS-divergence** (symmetric, bounded [0, 1]):

```python
from scipy.spatial.distance import jensenshannon

pred_mood_hits = defaultdict(int)
for clip in liris_test_set:
    v_pred, a_pred = model(clip)
    mood_idx = va_to_mood_k7(v_pred, a_pred)
    pred_mood_hits[mood_idx] += 1

pred_dist = np.array([pred_mood_hits[i]/len(liris_test_set) for i in range(7)])
gt_dist   = np.array([3015, 2664, 1810, 1174, 569, 568, 0]) / 9800

# 2번째 label index 기준 정렬 필요 (Peacefulness=2 등)
js_div = jensenshannon(pred_dist, gt_dist, base=2)  # [0, 1]

# 기대 결과:
# - js_div ≈ 0.05~0.15 → 모델이 GT 분포 잘 복제
# - 두 분포 모두 JA ≈ 0 (LIRIS 구조적 특성)
```

### 19-5. 보고서 방어 문단 (V5-final)

> "LIRIS-ACCEDE Discrete의 scalar V/A value는 이론 범위 [1, 5]와 달리 실제로 Valence [1.33, 3.59] · Arousal [1.32, 4.54]에 분포한다(§2-1). 이는 Baveye et al. (2015, IEEE TAC)에 기술된 crowdsourced pairwise ranking → affective rating 회귀의 수치적 특성에 기인한다. 9,800 clip 전체에 대한 7-centroid Euclidean 매핑 집계(§2-5) 결과, 본 연구의 7 GEMS mood 중 **6개(Peacefulness 30.77%, Sadness 27.18%, Tenderness 18.47%, Tension 11.98%, Power 5.81%, Wonder 5.80%)는 충분한 학습 신호를 제공**하나 **Joyful Activation(V=+0.7, A=+0.6) 단 1개 centroid는 v_norm max +0.30 제약으로 0.00% 할당** — 즉 구조적 도달 불가이다. 본 연구는 이를 **LIRIS 데이터셋의 단일 구조적 커버리지 한계**로 명시적으로 보고하며, 학습된 모델의 LIRIS ID test (80 unseen films)에서 예측 7-centroid 분포와 GT 분포 간 **Jensen-Shannon divergence = X** (기대 <0.15)로 정량화한다. 외부 OOD 일반화는 본 연구 범위 밖이며, 후속 연구에서 supplementary positive-valence data (EmoMusic, DEAM, MediaEval 2015-2018)로 JA 커버리지 확장이 가능하다. 본 연구의 V3.2 §6-4 10-band Biquad peaking EQ 프리셋 7종 자체는 Juslin & Laukka (2003), Arnal (2015), Bowling (2017) 등 음향 심리학적 근거로 완전 도출되어 있어 JA EQ 프리셋도 데이터 확보 시 즉시 활용 가능하다."

### 19-6. 학술 기여 3가지
1. 데이터 감수 성실성 (분포 구조 + JS-div 정량 보고)
2. 후속 연구 방향 (multi-dataset 학습 명제, JA 보완)
3. LIRIS 기반 커뮤니티 공통 제약 공론화 — 단 기존 V5의 과대추정("4개 under-represented") 대비 **실측상 단 1개 class로 제한**된 정확한 진단

---

## 20. Runtime 추정 (V5-final, 21 run)

| Phase | 작업 | Compute | 문서/분석 | 누적 |
|---|---|---|---|---|
| 0 | PANNs sanity(§2-5 GT 집계 이미 완료) + variance 시각화 | 40분 | 10분 | 50분 |
| 1 | zip 해제, CSV, split, feature precompute (X-CLIP+PANNs) | 4시간 | 4시간 | ~1일 실작업 |
| 2a | 4 ablation × 40min (baseline 우선) | 3시간 | 4시간 | ~1일 |
| 2b | 21 run × 40min | 14시간 | 8시간 | ~3일 |
| 3 | 추론 + §19 JS-div + 청취 A/B + 보고서 (movies.zip 필요) | 1일 | 2일 | ~3일 |

**총 compute ≈ 22시간, 총 소요 ≈ 3주**

---

## 21. 🆕 최종 모델 선정 전략 (V5-final 신설, 사용자 목적 직접 명시)

### 21-1. 목적

본 연구의 **최종 산출물**: 여러 모델 변형을 체계적 OAT ablation으로 성능 비교하여 **LIRIS 기반 V/A 회귀 + EQ 시스템의 최종 모델**을 결정.

### 21-2. 진행 원칙

#### 원칙 1: **베이스라인 우선 실행 (사용자 결정)**
- Phase 2a-0 (V3.2 원래 설계: X-CLIP + PANNs + Gate + MT K=4) 먼저 완주
- val mean_CCC / per-axis CCC / loss term 수렴 / gate collapse / overfit gap 실측
- **사용자 sign-off 후 후속 ablation 방향 결정**

#### 원칙 2: OAT (One-At-a-Time)
- Phase 2a는 한 번에 한 축만 변경, 나머지 baseline 고정
- 승자 확정 → 다음 단계 baseline 승격
- 결합 효과는 Phase 2a 최종 구성으로 자동 확인

#### 원칙 3: 사용자 sign-off 게이트
각 Phase 2a 단계 종료마다 **사용자가 승자 확인 + 다음 단계 진행 결정**:
- 2a-0 완료 → "baseline 성능 OK? 2a-1 V/A 정규화 비교 진행?"
- 2a-1 완료 → "승자 확정 + 2a-2 Mood Head K 비교 진행?"
- 2a-2 완료 → "승자 확정 + 2a-3 Audio encoder 비교 진행?"
- 2a-3 완료 → "승자 확정 + 2a-4 Fusion 비교 진행? (선택)"

### 21-3. 최종 모델 채택 기준 (우선순위 순)

| 순위 | 기준 | 측정 |
|---|---|---|
| 1 | **Primary performance**: LIRIS val mean_CCC (Phase 2 기준) | point estimate + paired t-test |
| 2 | **Test generalization**: LIRIS test mean_CCC + film-level bootstrap 95% CI (Phase 3) | bootstrap |
| 3 | **Gate / training health**: gate_entropy, overfit gap, grad_norm balance | 로그 |
| 4 | **EQ 커버리지**: JS-divergence (예측 vs GT 7-centroid 분포, §19-4) | 추론 분석 |
| 5 | **추론 품질**: 샘플 영화 청취 A/B (Phase 3) | 평가자 5명 |

### 21-4. Phase 2a 비교 테이블 템플릿

Phase 2a 단계별 다음 테이블로 승자 결정:

| 단계 | A (baseline) | B (후보) | A val mean_CCC | B val mean_CCC | paired t p-value | 승자 |
|---|---|---|---|---|---|---|
| 2a-1 | 전략 A | 전략 B | — | — | — | — |
| 2a-2 | K=4 | K=7 | — | — | — | — |
| 2a-3 | PANNs | AST | — | — | — | — |
| 2a-4 | Gate | GMU | — | — | — | — |

Phase 2b 이후 역시 단계별 테이블 유지 → 보고서에 그대로 포함 가능.

### 21-5. 예상 Phase 2 최종 모델 (baseline 결과 전 예측)

**기대 Final Model**: X-CLIP + PANNs + Gate + MT K=4 (V3.2 원래 설계 그대로)
- 근거: V3.3 ablation(CCMovies)에서 AST/GMU는 small-data 대응이었고 LIRIS 규모에서 V3.2 원설계 우위 기대
- 단 Phase 2a 실측으로 확정. **예측과 실측이 다를 경우 실측 우선**

### 21-6. 모델 선정 보고서 구조 (draft)

최종 보고서에 포함할 7개 섹션:
1. 베이스라인 2a-0 성능 (val CCC, loss term trajectories, gate entropy, overfit 여부)
2. V/A 정규화 전략 비교 (2a-1)
3. Mood Head K 비교 (2a-2)
4. Audio encoder 비교 (2a-3)
5. Fusion 비교 (2a-4, 선택)
6. 하이퍼 튜닝 결과 (2b-1~6)
7. Phase 3 최종 모델 test 성능 + §19 JS-div + 청취 A/B + 결론

---

## 22. 🆕 다운로드 대기 병행 작업 (V5-final 신설)

**movies.zip (30.4GB) 다운로드 중 (2일 예상)** — 이 기간 동안 병행 가능:

### 22-1. 즉시 실행 가능 (movies.zip 불필요)
1. **Phase 0 §6-1 PANNs 4초 sanity** — data.zip의 9,800 clip 중 10개로 30분
2. **Phase 1 전체** — data.zip + annotations.zip만으로 완결
3. **Phase 2a-0 베이스라인** — feature precompute 완료 시 바로 가능
4. **코드 scaffolding** — `model/autoEQ/train_liris/` 디렉토리 생성 + config/dataset/model/loss/trainer 구현
5. **단위 테스트 9종 작성**

### 22-2. movies.zip 완료 후 가능
1. **Phase 3 샘플 영화 end-to-end 검증** — full movie로 `analyze → timeline.json → playback → remux` 최종 검증
2. **청취 A/B 평가** — full movie EQ 적용 결과 3~5명 평가자

### 22-3. 권장 진행 순서
```
T+0 (지금)        — V5-final 플랜 확정 (완료)
T+0 ~ T+1h        — Phase 0 §6-1 PANNs sanity
T+1h ~ T+2일      — Phase 1 (data 해제, preprocess, precompute) + 코드 scaffolding 병행
T+2일 (movies.zip 완료 예상) ← 이 시점에 Phase 2a-0 baseline 준비 완료
T+2일 ~ T+3일     — Phase 2a-0 baseline 실행 + 사용자 sign-off
T+3일 ~ T+7일     — Phase 2a-1~4 OAT (각 단계 sign-off)
T+7일 ~ T+14일    — Phase 2b 하이퍼 튜닝 (21 run)
T+14일 ~ T+17일   — Phase 3 추론 + §19 JS-div + 청취 A/B + 보고서
```

**핵심**: movies.zip 2일 대기가 총 일정에 블로킹되지 않음 — Phase 0~2a-0까지 모두 data.zip만으로 진행 가능.
