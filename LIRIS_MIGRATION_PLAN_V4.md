# LIRIS-ACCEDE 전환 플랜 V4 (COGNIMUSE 배제 + EQ 커버리지 학술 프레이밍)

## Context (왜 이 작업인가)

- V3.2 명세로 LIRIS-ACCEDE 학습 계획 → 데이터셋 확보 불가로 V3.3 축소 명세(CCMovies 9편 + pseudo-label) 전환 → V3.3 최종 모델(X-CLIP+AST+GMU+MT K=4) 학습 완료.
- **2026-04-20 LIRIS-ACCEDE 도착** → V3.3 진행분 취소, LIRIS로 재시작.
- 사용자 핵심 요구: V3.3의 학습 프로세스 외곽은 유지, 모델 내부는 V3.2 원래 설계(PANNs + Gate + Multi-task)로 복귀.
- V4는 V3에서 실측(9,800 clip 직접 확인)된 제약조건을 반영하고, 2차 피드백 리뷰에서 지적된 **EQ 프리셋 커버리지 한계의 학술적 정직성**을 플랜에 명시, **COGNIMUSE 배제 결정**까지 수용한 최종 확정본.

## 0. 변경 이력

### V3 → V4 (2026-04-20)
1. **COGNIMUSE 전면 배제** (사용자 결정) — OOD 검증은 LIRIS 공식 test split(80 films, 4,900 clips, 학습 시 미관측)이 film-level generalization 역할 수행. 외부 OOD는 Future Work.
2. **§19 신설 — EQ 프리셋 커버리지 한계의 학술적 프레이밍** — 실측 Valence positive 편향(v_norm ≤ +0.30)으로 JA/PE/TE/WO 4개 mood centroid 도달 불가. 은폐 대신 "LIRIS 데이터셋의 구조적 특성"으로 정직하게 보고 → 학술적 기여.
3. **§8 K=7 3조건 게이트 순서 통일** — 실측 근거로 Data sufficiency → Non-interference → Learnability로 fail-fast 순서 확정(비용 오름차순 × 예상 fail rate).
4. **§14-3 Loss 로깅 세분화** — L_va 내부 MSE/CCC 성분 독립 로깅으로 ccc_hybrid_w 적정성 실시간 진단.
5. **§13 Phase 2b-5 증강 off-ablation 6 run 테이블로 구체화** (a~f).
6. **§13 Phase 2b-6 Modality Dropout p ablation 신설** (p ∈ {0, 0.05, 0.10, 0.15}).
7. **§20 Runtime 추정 테이블** — compute vs 문서/분석 분리.

### V2 → V3 (참고 유지)
- 9,800 clip 실측으로 V/A 이론 범위 [1~5] → 실제 [1.33, 3.59]/[1.32, 4.54] 확인, K=4 default 전환.
- Variance threshold 0.15 폐기 → per-axis p75 AND 조건.
- Phase 2 → 2a(근본 OAT) + 2b(하이퍼 OAT) 분리.
- Film-level cluster bootstrap 구현 명시.
- Overfit auto-monitor (train−val CCC gap > 0.10).
- Phase 0 sanity "full clip vs 4s crop"으로 재설계.

### V1 → V2 (참고 유지)
- "Pareto" → "Lexicographic" 교정, "Label Smoothing" → "Target Shrinkage" 교정.
- CCC 단독 primary, Pearson/MAE는 diagnostic.
- V3.2 모델 내부(PANNs + Gate + Multi-task) 복귀.

---

## 1. 사용자 확정 결정사항

1. Congruence Head + Negative Sampling: 미복원
2. Mood Head: **K=4 quadrant default** (§8 게이트 통과 시 K=7 상향)
3. 코드 구조: 새 `model/autoEQ/train_liris/` 폴더
4. 베이스 모델: X-CLIP + PANNs CNN14 + Gate Network + Intermediate Fusion + Multi-task
5. 평가 프레이밍: CCC 단독 primary (AVEC), Pearson/MAE diagnostic
6. Early stopping: Lexicographic `(mean_CCC, mean_Pearson, −mean_MAE)`, patience=10
7. **OOD 검증: COGNIMUSE 배제** (V4 신규) — LIRIS 공식 test set(80 films, 학습 시 미관측)이 film-level generalization 평가를 수행. 외부 OOD는 Future Work로 명시.

---

## 2. LIRIS-ACCEDE 실측 요약 (V3 §3 유지, V4에서 재검증 불필요)

### 2-1. V/A scalar 실측 범위
| 축 | 이론 (README) | 실제 | (raw−3)/2 변환 후 |
|---|---|---|---|
| valenceValue | 1~5 | **1.33 ~ 3.59** | **[−0.84, +0.30]** |
| arousalValue | 1~5 | 1.32 ~ 4.54 | [−0.84, +0.77] |

**함의**: v_norm이 +0.30에서 잘려 있어 V3.2 Mood centroid 중 positive V 4개(JA/PE/TE/WO) 도달 불가. **K=4 quadrant가 default**. 이 편향은 §19에서 EQ 커버리지 한계로 학술 보고.

### 2-2. Variance 분포 (Target Shrinkage threshold 근거)
- valenceVariance: p50=0.108, p75=0.117, max=0.164
- arousalVariance: p50=0.150, p75=0.164, max=0.218
- V3 확정 threshold: `v_thr=0.117, a_thr=0.164, logic=AND` → 약 6~10% 발동

### 2-3. 사분면 분포 ([−1, +1] 정규화 후)
HVHA 11.0% · HVLA 40.2% · LVHA 18.6% · LVLA 30.2% → K=4 학습 가능, min class 11% > chance 25%는 아니지만 stratified sampling으로 완화.

### 2-4. LIRIS 공식 split 채택
| split | 영화 | clip | 역할 |
|---|---|---|---|
| learning (set=1) | 40 | 2,450 | train |
| validation (set=2) | 40 | 2,450 | val (early stop) |
| test (set=0) | 80 | 4,900 | **test (unseen films generalization)** |

test의 80 films는 train/val에 등장하지 않으므로 **film-level hold-out = in-domain OOD 역할**. COGNIMUSE 없이도 외부영화 일반화 평가 가능.

---

## 3. 구성 요약

| 계층 | 선택 |
|---|---|
| Visual encoder | X-CLIP base-patch32 (frozen, 512-d) |
| Audio encoder | PANNs CNN14 (frozen, 2048-d) + Linear Projection 2048→512 |
| Fusion | Gate Network (1024→256→2 softmax) + Intermediate Fusion → 1024-d |
| VA Head | 1024 → 256 → 2 |
| **Mood Head** | **1024 → 256 → 4 (K=4 quadrant, default)** · K=7 상향은 §8 게이트 통과 시 |
| Congruence Head | 없음 |
| Modality Dropout | p=0.05 default (Phase 2b-6 ablation {0, 0.05, 0.10, 0.15}) |
| Feature Noise | σ=0.03 대칭 |
| Quadrant Mixup | prob=0.5, α=0.4 |
| Target Shrinkage | per-axis p75 threshold (v=0.117, a=0.164), AND 조건, ε=0.05 |
| V/A loss | CCC hybrid w=0.3 (Phase 2b-3 ablation {0, 0.3, 1.0}) |
| Mood loss | CE(K=4), λ_mood=0.3 (Phase 2b-1 ablation) |
| Gate entropy | λ_ent=0.05 (Phase 2b-2 ablation {0, 0.05, 0.1}) |
| Early stopping | Lexicographic (mean_CCC, mean_Pearson, −mean_MAE), patience=10 |
| Primary metric | mean_CCC (AVEC) |
| Bootstrap CI | Film-level cluster resampling |
| Overfit 감지 | train−val mean_CCC gap > 0.10 자동 경고 |
| Loss 로깅 | total, L_va, **L_va_mse, L_va_ccc**, L_mood, L_gate_entropy |

---

## 4. V3.2 ↔ V3.3 ↔ V4 플랜 정합성 매트릭스

| 항목 | V3.2 | V3.3 (CCMovies) | V4 | 비고 |
|---|---|---|---|---|
| 학습 데이터 | LIRIS 9,800 | CCMovies 1,289 | LIRIS 9,800 | V3.2 복귀 |
| Split | 120/20/20 | LOMO 9-fold | **LIRIS 공식 40/40/80** | 타 논문 비교 가능 |
| OOD 검증 | COGNIMUSE 200/7편 | (legacy만) | **LIRIS test 80 films(in-domain unseen)** · COGNIMUSE 배제 · 외부 OOD는 Future Work | V4 사용자 결정 |
| V/A range | [-1,+1] | [-1,+1] | (raw−3)/2 변환, v∈[-0.84,+0.30] a∈[-0.84,+0.77] | §2-1 실측 |
| Per-sample std | — | ensemble std | valenceVariance/arousalVariance (p75 AND) | §2-2 |
| Video encoder | X-CLIP | X-CLIP | X-CLIP | 불변 |
| Audio encoder | PANNs | AST | **PANNs** | 사용자 결정 |
| Fusion | Gate + Intermediate | GMU | **Gate + Intermediate** | 사용자 결정 |
| Mood Head | K=7 | K=4 | **K=4 default** · K=7은 §8 통과 시 | §2-1 편향 |
| Congruence | 포함 | 제거 | 미복원 | 사용자 |
| V/A loss | MSE | CCC hybrid w=0.3 | CCC hybrid w=0.3 · 2b-3 ablation | — |
| Gate entropy | λ=0.05 | N/A (GMU) | λ=0.05 · 2b-2 ablation | — |
| Feature Aug | 명시 없음 | Noise+Mixup+Shrinkage | 동일 + p75 AND threshold + 2b-5 off-ablation | — |
| Modality Dropout | p=0.1 cong_label==0만 | p=0.05 전체 | **p=0.05 default · 2b-6 ablation** | V4 신규 |
| Early stopping | val CCC | "Pareto"(실제 lex) | Lexicographic | V2 교정 |
| Primary metric | CCC | CCC | **CCC 단독** · Pearson/MAE diagnostic | V2 재프레이밍 |
| Bootstrap | — | — | Film-level cluster 95% CI | V3 |
| Overfit monitor | — | — | train−val CCC gap > 0.10 | V3 |
| **Loss 로깅** | total만 | total만 | total, L_va, **L_va_mse, L_va_ccc**, L_mood, L_gate_ent | V4 |
| EQ 프리셋 | 7×10 Biquad | 동일 | 동일 + **§19 커버리지 한계 보고** | V4 |

---

## 5. Phase 구조 (V3 유지, V4는 Runtime 테이블만 추가)

| Phase | Compute | 문서/분석 | 총 소요 | 핵심 산출물 |
|---|---|---|---|---|
| **0** | ~40분 (PANNs sanity + V/A scatter + variance 분포) | 10분 | **~50분** | `runs/phase0_sanity/report.json` |
| **1** | ~4시간 (feature precompute 3h + K=7 게이트 1h) | ~4시간 (CSV/split/검증) | **~1일 실작업 / 1주 문서포함** | `data/features/liris_panns/*.pt`, `k7_gate_result.json`, `distribution_report.json` |
| **2a** | ~3시간 (4개 ablation × 40min) | ~4시간 | **~1일** | `runs/phase2a_*/` |
| **2b** | ~13시간 (19 run × 40min) | ~8시간 | **~3일** | `runs/phase2b_*/` |
| **3** | ~1일 (추론 통합 + 청취 A/B) | ~1일 (보고서) | **~3일** | `runs/final/`, `§19 커버리지 분석`, 청취 결과 |

**총 compute: ~20시간 / 총 소요: ~3주 (문서 포함)**

---

## 6. Phase 0 — Sanity Check (~50분)

### 6-1. PANNs 4초 입력 품질 (§2-7 재설계, full clip vs 4s crop)
```python
# LIRIS clip 10개 랜덤 선택
for clip in sampled_10:
    full_audio = load(clip, sr=16000)               # 8~12s
    crop_4s = center_crop(full_audio, 4.0)
    feat_full = panns_cnn14(full_audio)             # 2048-d
    feat_crop = panns_cnn14(crop_4s)                # 2048-d
    sims.append(cosine_sim(feat_full, feat_crop))
```
판정:
- mean ≥ 0.90 → 4s crop 그대로
- 0.80~0.90 → 주의. multi-chunk average 고려
- < 0.80 → 대안: 10s zero-padding / multi-chunk / AST 회귀

### 6-2. V/A 2D scatter + K=4/K=7 centroid별 count
`v_norm = (valenceValue − 3) / 2, a_norm = (arousalValue − 3) / 2` 전체 9,800 샘플 시각화 + centroid 할당 집계.

### 6-3. Variance 분포 확인
v/a variance histogram + p75 threshold 선 그리기 + AND/OR 발동 비율 표.

산출: `runs/phase0_sanity/{panns_sanity.json, va_scatter.png, variance_dist.json}`

---

## 7. Phase 1 — LIRIS 데이터 준비 (~1주)

1. zip 해제: `unzip LIRIS-ACCEDE-{data,annotations}.zip -d dataset/LIRIS_ACCEDE/`
2. 메타데이터 파싱 → `liris_metadata.csv` (id, name, film_id, set, v_raw, a_raw, v_var, a_var, v_norm, a_norm, quadrant_k4, mood_k7)
3. film_id는 `ACCEDEdescription.xml`의 `<movie>` 태그에서 추출
4. 공식 split 적용 (set=0/1/2) + 같은 film이 여러 split에 없음 assert
5. **§8 K=7 3조건 게이트 실행 (fail-fast, 비용 오름차순 통일)**
6. Feature precompute: X-CLIP (512-d) + PANNs (2048-d) → `data/features/liris_panns/`
7. Phase 1 완료 체크리스트: `liris_metadata.csv`, `splits/official_split.json`, `k7_gate_result.json`, feature precompute manifest, `distribution_report.json`

---

## 8. K=7 Mood Head 상향 3조건 게이트 (V4 순서 통일, fail-fast)

**통일된 순서** (V3의 §8/§11-3 불일치 해소 — Data sufficiency 1순위로 확정):

### 조건 1 (최우선, 1분): Data sufficiency
- LIRIS train(40 films, ~9,800 windows)에서 각 7-class count ≥ 1%
- 실측 예상(§2-1 Valence 편향): Joyful Activation, Tenderness, Peacefulness, Wonder 4개 실질 0% → **FAIL 유력**
- 비용: CSV 집계 1분
- FAIL 시 즉시 K=4 확정, 이하 스킵 (fail-fast)

### 조건 2 (30분): Non-interference
- K=7 vs K=4 짧은 10-epoch mini-run 대조
- `|val_mean_CCC_K7 − val_mean_CCC_K4| ≤ 0.01`
- FAIL 시 K=4 확정, 조건 3 스킵

### 조건 3 (30분, 조건 2 run 재활용): Learnability
- K=7 학습의 val `mood_F1_macro > 14.3%` (random baseline 1/7)
- 조건 2의 K=7 run에서 함께 측정

**순서 근거**: 비용 오름차순 × 예상 fail rate. V3 §2-1 실측으로 "Data sufficiency 가장 빨리 실패" 확정되어 이것을 1순위로. 원 피드백의 "non-interference 1순위" 제안은 일반적 사고 실험이었으나, LIRIS 실측 이후 Data sufficiency 우선이 실용적.

**게이트 의사코드**:
```python
# 1. Data sufficiency (1min)
ratios = class_count_per_k7(train) / len(train)
if ratios.min() < 0.01:
    return "K=4 (data_sufficiency FAIL: min class < 1%)"

# 2. Non-interference (30min)
ccc_k7 = short_run(num_classes=7, epochs=10).val_mean_ccc
ccc_k4 = short_run(num_classes=4, epochs=10).val_mean_ccc
if abs(ccc_k7 - ccc_k4) > 0.01:
    return f"K=4 (non_interference FAIL: ΔCCC={abs(ccc_k7-ccc_k4):.3f})"

# 3. Learnability (재활용)
if mood_f1_k7 <= 0.143:
    return "K=4 (learnability FAIL)"

return "K=7 PASS"
```

---

## 9. 구현 파일 구조 (V3 §12 유지, Loss 세분화만 추가)

### 9-1. 신규 `model/autoEQ/train_liris/`
| 파일 | 내용 |
|---|---|
| `config.py` | TrainLirisConfig (num_mood_classes={4,7}, λ_mood=0.3, λ_gate_entropy=0.05, ccc_hybrid_w=0.3, modality_dropout_p=0.05, feature_noise_std=0.03, mixup_prob=0.5, mixup_alpha=0.4, target_shrinkage_eps=0.05, v_var_threshold, a_var_threshold, shrinkage_logic="AND", batch_size=32, lr=1e-4, wd=1e-5, epochs=40, patience=10, grad_clip=1.0, warmup=500, seed=42, use_official_split=True, pad_audio_to_10s=auto) |
| `liris_preprocess.py` | 메타데이터 파싱, V/A 변환, film_id 추출, feature precompute, variance threshold 산출 |
| `dataset.py` | `PrecomputedLirisDataset` + `official_split()` |
| `model.py` | `AutoEQModelLiris` (V3.3 `train_pseudo/model_base/model.py` 기반, MoodHead 1024→256→{4,7}) |
| `losses.py` | `combined_loss_liris` — **L_va_mse, L_va_ccc 독립 반환 추가** (§10 로깅용) |
| `trainer.py` | Lexicographic early stopping + Overfit auto-monitor + per-term loss 로깅 |
| `metrics.py` | CCC/Pearson/MAE/RMSE per-axis + `film_level_bootstrap_ci` |
| `run_train.py` | CLI |
| `eval_test.py` | 최종 평가 (bootstrap CI) |
| `tests/` | 9종 (split 무결성, V/A round-trip, K4/K7 매핑, CCC/Pearson 공식, dropout/mixup shape, film_level_bootstrap correctness) |

### 9-2. 재사용
- `train_pseudo/dataset.py::va_to_mood` (7-class GEMS)
- `train_pseudo/model_base/*` (V3.2 baseline 참조)
- `train/encoders.py` (X-CLIP/PANNs util)
- `infer_pseudo/*` — VARIANTS에 `liris_base` 추가
- `playback/*`, `scripts/eq_response_check.py`

### 9-3. 폐기
- `pseudo_label/*`, CCMovies preprocess, **COGNIMUSE preprocess (V4 신규 폐기)**, σ-filter 코드, CCMovies data

---

## 10. 평가·모니터링 구현 (V4 Loss 세분화 추가)

### 10-1. Film-level cluster bootstrap
```python
def film_level_bootstrap_ci(preds, targets, film_ids, metric_fn,
                            n_resamples=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    unique_films = np.unique(film_ids)
    film_idx = {f: np.where(film_ids == f)[0] for f in unique_films}
    results = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique_films, len(unique_films), replace=True)
        indices = np.concatenate([film_idx[f] for f in sampled])
        results.append(metric_fn(preds[indices], targets[indices]))
    lo, hi = np.percentile(results, [(1-ci)/2*100, (1+ci)/2*100])
    return {'point': metric_fn(preds, targets), 'ci_lo': lo, 'ci_hi': hi}
```
보고 형식: `LIRIS test: mean_CCC = 0.550 [95% CI: 0.508, 0.591] (film-level n=80)`

### 10-2. Overfit auto-monitor
```python
# 매 epoch 종료 시
gap = train_mean_ccc - val_mean_ccc
wandb.log({'diagnostic/train_val_ccc_gap': gap})
if gap > 0.10:
    log.warning(f"Overfit @ epoch {epoch}: train={train_mean_ccc:.3f} val={val_mean_ccc:.3f}")
    # config.overfit_response: 'alert_only' | 'reduce_lr' | 'stop_early'
```

### 10-3. Loss term 독립 로깅 (V4 MSE/CCC 성분 추가)
```python
# losses.py에서 각 성분 독립 반환
def combined_loss_liris(va_pred, va_target, mood_logits, mood_label, gate_w, cfg):
    mse_term = F.mse_loss(va_pred, va_target)
    ccc_v, ccc_a = compute_ccc_per_axis(va_pred, va_target)
    ccc_term = 1 - (ccc_v + ccc_a) / 2
    L_va = (1 - cfg.ccc_hybrid_w) * mse_term + cfg.ccc_hybrid_w * ccc_term
    L_mood = F.cross_entropy(mood_logits, mood_label)
    L_ent = gate_entropy_loss(gate_w)
    L_total = L_va + cfg.lambda_mood * L_mood + cfg.lambda_gate_entropy * L_ent
    return L_total, {
        'loss_total': L_total, 'loss_va': L_va,
        'loss_va_mse': mse_term, 'loss_va_ccc': ccc_term,  # V4 신규
        'loss_mood': L_mood, 'loss_gate_entropy': L_ent,
    }

# trainer.py
wandb.log({f'train/{k}': v.item() for k, v in loss_components.items()})
```
→ wandb 대시보드에서 MSE와 CCC term 독립 curve 확인. ccc_hybrid_w=0.3 적정성 실시간 진단.

### 10-4. Gate 진단 + Head-wise grad norm
V3 §14-4, §14-5 유지.

---

## 11. Phase 2a — 근본 설계 ablation (OAT 순차, ~1일)

Baseline 2a-0: V/A 전략 A, K=4, PANNs, Gate, 기본 하이퍼.

| 단계 | 변경 축 | 조건 |
|---|---|---|
| 2a-1 | V/A 정규화: 전략 A vs B (min-max stretch) | 나머지 baseline |
| 2a-2 | Mood Head: K=4 vs K=7 (§8 게이트 통과 시만) | 2a-1 승자 |
| 2a-3 | Audio encoder: PANNs vs AST | 2a-2 승자 |
| 2a-4 | Fusion: Gate vs GMU (선택적) | 2a-3 승자 |

각 단계 승자를 다음 baseline으로 승격. paired t-test 기반 판정.

---

## 12. Phase 2b — 하이퍼 튜닝 ablation (OAT, ~3일)

Baseline 2b-0: Phase 2a 최종 승자.

| 단계 | 변경 축 | 값 | Run 수 |
|---|---|---|---|
| 2b-1 | λ_mood | {0, 0.1, 0.3, 0.5} | 4 |
| 2b-2 | λ_gate_entropy | {0, 0.05, 0.1} | 3 |
| 2b-3 | ccc_hybrid_w | {0, 0.3, 1.0} | 3 |
| 2b-4 | Batch size | {32, 64} | 2 |
| **2b-5** | **증강 off-ablation** | 6 run 테이블 (아래) | 6 |
| **2b-6** | **Modality Dropout p (V4 신규)** | {0, 0.05, 0.10, 0.15} | 4 |

### 2b-5 증강 off-ablation 테이블 (V4 구체화, OAT 엄수)
| Run | Modality Dropout | Feature Noise | Quadrant Mixup | Target Shrinkage |
|---|---|---|---|---|
| 2b-5-a | OFF | ON | ON | ON |
| 2b-5-b | ON | OFF | ON | ON |
| 2b-5-c | ON | ON | OFF | ON |
| 2b-5-d | ON | ON | ON | OFF |
| 2b-5-e (ALL_OFF) | OFF | OFF | OFF | OFF |
| 2b-5-f (ALL_ON, reference) | ON | ON | ON | ON |

**총 Phase 2b run: 4+3+3+2+6+4 = 22 run**, compute ≈ 14.7시간.

---

## 13. Phase 3 — 추론/재생 + 최종 평가 (~3일)

1. Phase 2 최종 모델 확정 → `infer_pseudo/model_inference.py::VARIANTS`에 `liris_base` entry 추가
2. LIRIS test set(80 films, 4,900 clips) 최종 평가 — **film-level bootstrap 95% CI 포함**
3. **§19 EQ 커버리지 분석** 실행 — LIRIS test set에 대한 모델 예측에서 각 mood centroid 발동 빈도 측정
4. 샘플 영화 1편 end-to-end (analyze → timeline.json → playback → remux)
5. 청취 A/B: α_d ∈ {0.3, 0.5, 0.7}
6. 최종 보고서 작성

---

## 14. Verification

### 14-1. 코드/데이터 무결성
- `pytest model/autoEQ/train_liris/tests/ -v` (9/9 PASS)
- `liris_preprocess --verify` (split 무결성, V/A round-trip, variance threshold 발동률 assert)

### 14-2. 학습 sanity (2 epoch mini-run)
- L_total 감소 + L_va_mse/L_va_ccc/L_mood/L_gate_ent 4개 성분 독립 감소 확인
- grad_norm head 간 10배 이내 균형
- train/val CCC gap 모니터링 작동

### 14-3. Primary metric 목표 (bootstrap CI 포함)
- LIRIS val: mean_CCC ≥ 0.30 (baseline) / ≥ 0.45 (stretch)
- LIRIS test (80 unseen films): mean_CCC ≥ 0.28 + 95% CI 보고
- 축별 ccc_v, ccc_a 각각 보고
- §19 커버리지 분석: 각 mood centroid 발동 빈도 집계

### 14-4. End-to-end
- 샘플 1편 `analyze → timeline.json → playback → remuxed.mp4` 무오류
- VAD 대사 보호 적용 확인

---

## 15. 리스크 및 트레이드오프 (V4 업데이트)

| # | 리스크 | 대응 |
|---|---|---|
| 1 | Valence positive 편향으로 K=7 불가 (§2-1) | K=4 default, §8 fail-fast 게이트, §19 학술 보고 |
| 2 | Variance threshold per-axis 조정 | v=0.117, a=0.164 AND 조건 (6~10% 발동) |
| 3 | PANNs 4초 입력 품질 | Phase 0 sanity 사전 차단 |
| 4 | Overfit (183만 param / 9,800 windows) | Overfit auto-monitor + 증강 4종 + patience 10 |
| 5 | **외부 OOD 부재 (COGNIMUSE 배제)** | **LIRIS test 80 unseen films가 film-level generalization 평가 수행. 외부 OOD(EmoMusic/DEAM/MediaEval)는 Future Work로 명시** |
| 6 | Congruence 미복원의 학술 축소 | 보고서 "실증적 단순화" 방어 |
| 7 | LIRIS split 1:1:2 skew (train 40 : val 40 : test 80) | 필요 시 `--use_full_learning_set`로 set 1+2 병합 옵션 |
| 8 | 증강 효과 약화 | Phase 2b-5 off-ablation (6 run) |
| 9 | **EQ 프리셋 커버리지 편향 (§19)** | **학술적 발견으로 정직 보고 + COGNIMUSE 대체로 LIRIS test에서 발동 분포 측정** |

---

## 16. 결론

**"V3.3 학습 프로세스 외곽 × V3.2 모델 내부 × LIRIS 실측 기반 교정"** 하이브리드.

V4 핵심 추가:
1. **§19 EQ 커버리지 한계 학술 프레이밍** — 은폐 대신 정직 보고로 학술 기여 전환
2. **COGNIMUSE 배제** — LIRIS 공식 test 80 films가 film-level 일반화 역할. 외부 OOD는 Future Work
3. **§8 K=7 게이트 순서 통일** — Data sufficiency 1순위 (실측 기반 fail-fast)
4. **Loss MSE/CCC 성분 독립 로깅**
5. **2b-5 증강 off-ablation 6 run 테이블**
6. **2b-6 Modality Dropout p ablation 신설**
7. **Runtime 테이블**

진행 순서: Phase 0 (50분) → Phase 1 (~1일 실작업) → Phase 2a (~1일) → Phase 2b (~3일) → Phase 3 (~3일). 총 compute ~20시간, 문서 포함 ~3주.

---

## 17. (V3에서 유지) Mood centroid 좌표

K=7 상향 시 사용. V3.2 §2-5 유지:
| Mood | V | A | LIRIS 도달성 |
|---|---|---|---|
| Tension | −0.6 | +0.7 | ✅ |
| Sadness | −0.6 | −0.4 | ✅ |
| Peacefulness | +0.5 | −0.5 | ❌ (v>+0.30 불가) |
| Joyful Activation | +0.7 | +0.6 | ❌ |
| Tenderness | +0.4 | −0.2 | ❌ |
| Power | +0.2 | +0.8 | ⚠️ (경계) |
| Wonder | +0.5 | +0.3 | ❌ |

---

## 18. (V3에서 유지) 데이터 증강 4종 상세

### 증강 1: Modality Dropout (model.forward)
`if p > 0: trig = rand < p; choice = randint(0,2); v[trig & choice==0]=0; a[trig & choice==1]=0`
V4: p=0.05 default, Phase 2b-6에서 {0, 0.05, 0.10, 0.15} ablation.

### 증강 2: Feature Noise (model.forward)
`x' = x + N(0, σ²), σ=0.03, 대칭, dropped sample은 bypass`

### 증강 3: Quadrant Mixup (collate_fn, prob=0.5)
같은 사분면 쌍, λ~Beta(0.4, 0.4)→[0.1, 0.9] shrink, visual/audio/v/a/v_var/a_var 동일 λ 선형결합, mood label은 primary 유지.

### 증강 4: Target Shrinkage (collate_fn)
```python
trigger = (v_var > 0.117) & (a_var > 0.164)  # AND, ~6-10% 발동
if trigger: v *= 0.95, a *= 0.95
```

### 순서
DataLoader → collate_fn [Mixup → Target Shrinkage] → model.forward [Modality Dropout → Feature Noise] → Gate → Fusion → Heads → Loss

---

## 19. 🆕 EQ 프리셋 커버리지 한계 — 학술적 정직 보고 (V4 핵심 신설)

### 19-1. 문제 진술

LIRIS-ACCEDE Discrete의 9,800 clip 실측(§2-1)에서 valenceValue는 이론 범위 [1, 5]와 달리 실제로 [1.33, 3.59]에 분포한다. 이는 Baveye et al. (2014)의 crowdsourced pairwise ranking → scalar regression 과정의 수치적 특성에 기인한다. `(v−3)/2` 변환 후 `v_norm ∈ [−0.84, +0.30]`로, V3.2 명세 §2-5의 7 GEMS mood centroid 중 **positive Valence 4개(Joyful Activation V=+0.7, Tenderness V=+0.4, Peacefulness V=+0.5, Wonder V=+0.5)에 대해 학습 신호가 구조적으로 불충분**하다.

### 19-2. 연쇄 영향 (학습 → 추론 → EQ)

```
[학습] LIRIS Valence positive 편향
  → 모델 V_pred ∈ [−0.84, +0.30] 범위로 수렴

[추론] 다양한 영화 씬 입력
  → V_pred ≤ +0.30만 출력 (외삽 제한)
  → V/A → 7-centroid Euclidean 매핑 시
  → Tension/Sadness/Power 쪽으로 편중 매핑

[EQ 적용] 실제 Joyful Activation 씬에 Sadness EQ 발동 위험
  → 감독 의도 반대 효과
```

### 19-3. 대응 — 은폐 대신 학술적 발견으로 보고

본 연구는 이 한계를 **LIRIS-ACCEDE 데이터셋의 구조적 특성**으로 명시적으로 보고한다. 이는 후속 LIRIS 기반 정서 회귀 연구에 공통되는 제약이며, 커뮤니티에 공론화하는 것이 장기적 학술 기여다.

### 19-4. 정량 측정 (Phase 3 실행)

Phase 3에서 학습된 모델이 LIRIS test set(80 unseen films, 4,900 clips)을 추론할 때 각 mood centroid가 발동되는 빈도를 집계:

```python
# eval_test.py
mood_hits = defaultdict(int)
for clip in liris_test_set:
    v_pred, a_pred = model(clip)
    mood = va_to_mood_k7(v_pred, a_pred)
    mood_hits[mood] += 1

# 예상 결과:
# Tension/Sadness/Power: ~90% 합산
# JA/TE/PE/WO: <10% 합산 (under-represented)
```

### 19-5. 보고서 방어 문단 (draft)

> "LIRIS-ACCEDE Discrete의 scalar V/A value는 이론 범위 [1, 5]와 달리 실제로는 Valence [1.33, 3.59] · Arousal [1.32, 4.54]에 분포한다(§2-1). 이는 Baveye et al. (2014)의 crowdsourced pairwise ranking을 MAC 2014 regression으로 변환하는 과정의 수치적 특성에 기인하며, V3.2 명세의 7 GEMS centroid 중 positive Valence 4개(Joyful Activation, Tenderness, Peacefulness, Wonder)에 대해 학습 신호가 구조적으로 불충분하다. 본 연구는 이를 은폐하지 않고 **LIRIS 데이터셋의 구조적 커버리지 한계**로 명시적으로 보고하며, 이 4개 mood의 EQ 프리셋은 학습된 모델에서 발동 빈도가 낮게 측정되었다(§19-4 분석: 전체 test set 기준 합산 X%). 이 한계는 LIRIS 기반 정서 회귀 연구에 공통되는 제약이며, 후속 연구에서 supplementary positive-valence data(예: EmoMusic, DEAM, MediaEval 2015-2018)를 병용한 multi-dataset 학습으로 커버리지를 확장할 수 있다. 본 연구가 제안한 10-band Biquad peaking EQ 프리셋 자체는 7 mood 전부에 대해 V3.2의 음향 심리학적 근거(Juslin & Laukka 2003, Arnal 2015, Bowling 2017 등)로부터 도출되어 있으므로, 데이터 확보 시 그대로 활용 가능한 형태로 보존되어 있다."

### 19-6. 이로 얻는 학술적 기여 3가지
1. 데이터 감수 성실성 (point estimate만 보고하지 않고 분포 구조 공개)
2. 후속 연구 방향 제시 (multi-dataset 학습 명제)
3. LIRIS 기반 커뮤니티 공통 제약 공론화

---

## 20. Runtime 추정 (V4 신규 테이블)

| Phase | 작업 | Compute | 문서/분석 | 누적 |
|---|---|---|---|---|
| 0 | PANNs sanity + V/A scatter + variance | 40분 | 10분 | 50분 |
| 1 | zip 해제, CSV, split, K=7 게이트, feature precompute | 4시간 | 4시간 | ~1일 실작업 |
| 2a | 4개 ablation × 40min | 3시간 | 4시간 | ~1일 |
| 2b | 22 run × 40min (λ_mood 4 + λ_ent 3 + ccc_w 3 + batch 2 + aug 6 + dropout_p 4) | 14.7시간 | 8시간 | ~3일 |
| 3 | 추론 통합 + §19 발동 분포 분석 + 청취 A/B + 보고서 | 1일 | 2일 | ~3일 |

**총 compute ≈ 23시간**, **총 소요 ≈ 3주 (문서/분석 포함)**.
