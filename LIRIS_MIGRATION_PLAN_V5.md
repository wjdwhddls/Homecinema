# LIRIS-ACCEDE 전환 플랜 V5 (용어 정확성 + 학술 인용 교정)

## Context (왜 이 작업인가)

- V3.2 명세로 LIRIS-ACCEDE 학습 계획 → 데이터셋 확보 불가로 V3.3 축소 명세(CCMovies 9편 + pseudo-label) 전환 → V3.3 최종 모델(X-CLIP+AST+GMU+MT K=4) 학습 완료.
- **2026-04-20 LIRIS-ACCEDE 도착** → V3.3 진행분 취소, LIRIS로 재시작.
- 사용자 핵심 요구: V3.3의 학습 프로세스 외곽은 유지, 모델 내부는 V3.2 원래 설계(PANNs + Gate + Multi-task)로 복귀.
- V5는 V4의 학술 정합성을 완성한 **minor revision**. 근본 설계는 V4 그대로, 용어·인용·실측 연결·내부 일관성만 교정.

## 0. 변경 이력

### V4 → V5 (2026-04-20, 학술 정확성 minor revision)
1. **§3-1 "in-domain OOD" 용어 모순 교정** — OOD와 in-domain은 반대 개념. LIRIS test 80 films는 **in-distribution (ID) hold-out**으로 정정. 외부 OOD(COGNIMUSE/EmoMusic/DEAM 등 다른 데이터셋)는 본 연구 범위 밖으로 명시.
2. **§3-2 Baveye 인용 수정** — `Baveye et al. (2014)` / `MAC 2014 regression` → **`Baveye et al. (2015, IEEE TAC)` / `affective rating 회귀`**로 통일. IEEE Transactions on Affective Computing 저널이 LIRIS-ACCEDE 데이터셋의 권위 인용.
3. **§3-3 §19-4 실측 연결** — "예상 ~90%" 추측 → Phase 0 §6-2에서 집계할 ground-truth 7-centroid 분포 표로 구조화. Phase 3에서 모델 예측 분포 vs ground truth 분포 KL-divergence 측정.
4. **§3-4 Runtime 일치** — §5와 §20의 2b run 수 불일치(19 vs 22) 해소. 실제 22 run에서 §3-6 중복 제거로 **21 run**, compute ≈ 14시간, 총 ~22시간으로 통일.
5. **§3-5 Bootstrap 범위 명시** — Phase 3 최종 평가(LIRIS test 80 films)에만 film-level cluster bootstrap 적용. Phase 2 ablation은 point estimate + paired t-test.
6. **§3-6 2b-5 ∩ 2b-6 중복 제거** — 2b-5-a(Modality Dropout OFF) ≡ 2b-6 p=0. 2b-6을 `p ∈ {0.05, 0.10, 0.15}` 3 run으로 축소, p=0 포인트는 2b-5-a 결과 재사용.
7. **§3-7 §8 순서 근거 이원화** — "결정력 기준"(Non-interference 1순위)과 "실행 효율 기준"(Data sufficiency 1순위)을 명시적으로 분리. LIRIS 실측상 Data sufficiency fail이 유력하므로 실행 효율 기준 채택.

### V3 → V4 (참고)
- COGNIMUSE 전면 배제
- §19 EQ 프리셋 커버리지 한계 학술 프레이밍 신설
- §8 K=7 게이트 순서 통일 (Data sufficiency 1순위)
- Loss 로깅에 L_va_mse/L_va_ccc 독립 성분 추가
- 2b-5 증강 off-ablation 6 run 테이블
- 2b-6 Modality Dropout p ablation 신설
- Runtime 테이블

### V2 → V3 / V1 → V2 (참고 유지)
- 9,800 clip 실측 → V/A 이론 [1~5] vs 실제 [1.33, 3.59]/[1.32, 4.54], K=4 default
- Variance threshold 0.15 폐기, per-axis p75 AND
- Phase 2 → 2a(근본 OAT) + 2b(하이퍼 OAT) 분리
- Film-level cluster bootstrap 구현 명시
- Overfit auto-monitor, Phase 0 sanity "full clip vs 4s crop"
- "Pareto" → "Lexicographic", "Label Smoothing" → "Target Shrinkage"
- CCC 단독 primary, V3.2 모델 내부 복귀

---

## 1. 사용자 확정 결정사항

1. Congruence Head + Negative Sampling: 미복원
2. Mood Head: **K=4 quadrant default** (§8 게이트 통과 시 K=7 상향)
3. 코드 구조: 새 `model/autoEQ/train_liris/` 폴더
4. 베이스 모델: X-CLIP + PANNs CNN14 + Gate Network + Intermediate Fusion + Multi-task
5. 평가 프레이밍: CCC 단독 primary (AVEC), Pearson/MAE diagnostic
6. Early stopping: Lexicographic `(mean_CCC, mean_Pearson, −mean_MAE)`, patience=10
7. **평가 범위 (V5 교정)**: LIRIS 공식 test(80 films)는 **in-distribution (ID) hold-out**이며 학습 분포 내 unseen films에 대한 일반화만 측정. **외부 OOD 평가**(COGNIMUSE/EmoMusic/DEAM/MediaEval 등 다른 데이터셋)는 본 연구 범위 밖, **Future Work**.

---

## 2. LIRIS-ACCEDE 실측 요약

### 2-1. V/A scalar 실측 범위
| 축 | 이론 (README) | 실제 | (raw−3)/2 변환 후 |
|---|---|---|---|
| valenceValue | 1~5 | **1.33 ~ 3.59** | **[−0.84, +0.30]** |
| arousalValue | 1~5 | 1.32 ~ 4.54 | [−0.84, +0.77] |

**함의**: v_norm이 +0.30에서 잘려 있어 V3.2 Mood centroid 중 positive V 4개(JA/PE/TE/WO) 도달 불가. **K=4 quadrant가 default**. 이 편향은 §19에서 EQ 커버리지 한계로 학술 보고.

### 2-2. Variance 분포 (Target Shrinkage threshold 근거)
- valenceVariance: p50=0.108, p75=0.117, max=0.164
- arousalVariance: p50=0.150, p75=0.164, max=0.218
- 확정 threshold: `v_thr=0.117, a_thr=0.164, logic=AND` → 약 6~10% 발동

### 2-3. 사분면 분포 ([−1, +1] 정규화 후)
HVHA 11.0% · HVLA 40.2% · LVHA 18.6% · LVLA 30.2% → K=4 학습 가능, min class 11%는 stratified sampling으로 완화.

### 2-4. LIRIS 공식 split 채택 (V5 용어 교정)
| split | 영화 | clip | 역할 |
|---|---|---|---|
| learning (set=1) | 40 | 2,450 | train |
| validation (set=2) | 40 | 2,450 | val (early stop) |
| test (set=0) | 80 | 4,900 | **test (LIRIS 분포 내 unseen films 일반화 평가)** |

**test의 80 films는 train/val에 등장하지 않으므로 film-level hold-out = in-distribution (ID) generalization 평가**. 학습 분포(LIRIS-ACCEDE) 내에서 unseen films에 대한 일반화만 측정되며, 외부 OOD(COGNIMUSE 등 다른 데이터셋) 평가는 본 연구 범위 밖 (Future Work).

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
| Modality Dropout | p=0.05 default (Phase 2b-6 ablation {0.05, 0.10, 0.15}; p=0은 2b-5-a에서 공유) |
| Feature Noise | σ=0.03 대칭 |
| Quadrant Mixup | prob=0.5, α=0.4 |
| Target Shrinkage | per-axis p75 threshold (v=0.117, a=0.164), AND 조건, ε=0.05 |
| V/A loss | CCC hybrid w=0.3 (Phase 2b-3 ablation {0, 0.3, 1.0}) |
| Mood loss | CE(K=4), λ_mood=0.3 (Phase 2b-1 ablation) |
| Gate entropy | λ_ent=0.05 (Phase 2b-2 ablation {0, 0.05, 0.1}) |
| Early stopping | Lexicographic (mean_CCC, mean_Pearson, −mean_MAE), patience=10 |
| Primary metric | mean_CCC (AVEC) |
| **Bootstrap CI** | **Phase 3 최종 평가에만 적용** (film-level cluster resampling) — Phase 2는 point estimate |
| Overfit 감지 | train−val mean_CCC gap > 0.10 자동 경고 |
| Loss 로깅 | total, L_va, L_va_mse, L_va_ccc, L_mood, L_gate_entropy |

---

## 4. V3.2 ↔ V3.3 ↔ V5 플랜 정합성 매트릭스

| 항목 | V3.2 | V3.3 (CCMovies) | V5 | 비고 |
|---|---|---|---|---|
| 학습 데이터 | LIRIS 9,800 | CCMovies 1,289 | LIRIS 9,800 | V3.2 복귀 |
| Split | 120/20/20 | LOMO 9-fold | **LIRIS 공식 40/40/80** | 타 논문 비교 가능 |
| 평가 범위 | COGNIMUSE OOD | (legacy만) | **LIRIS test 80 films (ID hold-out)** · 외부 OOD는 Future Work | V5 용어 교정 |
| V/A range | [-1,+1] | [-1,+1] | (raw−3)/2, v∈[-0.84,+0.30] a∈[-0.84,+0.77] | §2-1 |
| Per-sample std | — | ensemble std | valenceVariance/arousalVariance (p75 AND) | §2-2 |
| Video encoder | X-CLIP | X-CLIP | X-CLIP | 불변 |
| Audio encoder | PANNs | AST | **PANNs** | 사용자 |
| Fusion | Gate + Intermediate | GMU | **Gate + Intermediate** | 사용자 |
| Mood Head | K=7 | K=4 | **K=4 default** · K=7은 §8 통과 시 | §2-1 |
| Congruence | 포함 | 제거 | 미복원 | 사용자 |
| V/A loss | MSE | CCC hybrid | CCC hybrid w=0.3 · 2b-3 ablation | — |
| Gate entropy | λ=0.05 | N/A | λ=0.05 · 2b-2 ablation | — |
| Feature Aug | 명시 없음 | 3종 | 4종 + 2b-5 off-ablation | — |
| Modality Dropout | p=0.1 cong만 | p=0.05 전체 | p=0.05 default · 2b-6 {0.05,0.10,0.15} | V5 중복 제거 |
| Early stopping | val CCC | "Pareto" | Lexicographic | V2 |
| Primary metric | CCC | CCC | CCC 단독, Pearson/MAE diagnostic | V2 |
| Bootstrap | — | — | **Phase 3 최종만** film-level cluster 95% CI | V5 범위 명시 |
| Overfit monitor | — | — | gap > 0.10 경고 | V3 |
| Loss 로깅 | total | total | total, L_va, L_va_mse, L_va_ccc, L_mood, L_gate_ent | V4 |
| EQ 프리셋 | 7×10 Biquad | 동일 | 동일 + §19 커버리지 한계 보고 | V4 |

---

## 5. Phase 구조 (V5 Runtime 일치 교정)

| Phase | Compute | 문서/분석 | 총 소요 | 핵심 산출물 |
|---|---|---|---|---|
| **0** | ~40분 | 10분 | **~50분** | `runs/phase0_sanity/report.json` |
| **1** | ~4시간 (feature precompute 3h + K=7 게이트 1h) | ~4시간 | **~1일 실작업 / 1주 문서포함** | `data/features/liris_panns/*.pt`, `k7_gate_result.json`, `distribution_report.json` |
| **2a** | ~3시간 (4 ablation × 40min) | ~4시간 | **~1일** | `runs/phase2a_*/` |
| **2b** | ~14시간 (**21 run** × 40min) | ~8시간 | **~3일** | `runs/phase2b_*/` |
| **3** | ~1일 (추론 통합 + §19 커버리지 분석 + 청취 A/B) | ~1일 | **~3일** | `runs/final/`, 발동 분포 리포트, 청취 결과 |

**총 compute: ~22시간 / 총 소요: ~3주 (문서 포함)**

---

## 6. Phase 0 — Sanity Check (~50분)

### 6-1. PANNs 4초 입력 품질 (full clip vs 4s crop)
```python
for clip in sampled_10:
    full_audio = load(clip, sr=16000)       # 8~12s
    crop_4s = center_crop(full_audio, 4.0)
    feat_full = panns_cnn14(full_audio)     # 2048-d
    feat_crop = panns_cnn14(crop_4s)        # 2048-d
    sims.append(cosine_sim(feat_full, feat_crop))
```
판정: mean ≥ 0.90 OK / 0.80~0.90 주의 (multi-chunk 고려) / <0.80 대안 (10s pad / multi-chunk / AST)

### 6-2. V/A 2D scatter + ground-truth centroid 분포 집계
`v_norm = (valenceValue − 3) / 2, a_norm = (arousalValue − 3) / 2` 전체 9,800 샘플 시각화.

**K=4 quadrant ground-truth 분포** (실측 완료):
- HVHA 11.0% · HVLA 40.2% · LVHA 18.6% · LVLA 30.2%

**K=7 GEMS centroid ground-truth 분포** (Phase 0에서 집계 예정):
- Sadness (V=-0.6, A=-0.4): ~XX%  ← LIRIS 도달 가능
- Tension (V=-0.6, A=+0.7): ~XX%  ← LIRIS 도달 가능
- Power (V=+0.2, A=+0.8): ~XX%   ← 경계, 부분 도달
- Peacefulness (V=+0.5, A=-0.5): ~0%  ← v>+0.30 도달 불가
- Joyful Activation (V=+0.7, A=+0.6): ~0%  ← 도달 불가
- Tenderness (V=+0.4, A=-0.2): ~0%  ← 도달 불가
- Wonder (V=+0.5, A=+0.3): ~0%  ← 도달 불가

이 분포는 §19-4에서 학습 모델의 예측 분포와 비교됨 (Phase 3 KL-divergence 측정).

### 6-3. Variance 분포
v/a variance histogram + p75 threshold 선 + AND/OR 발동 비율 표.

산출: `runs/phase0_sanity/{panns_sanity.json, va_scatter.png, gt_centroid_dist.json, variance_dist.json}`

---

## 7. Phase 1 — LIRIS 데이터 준비 (~1일 실작업)

1. zip 해제: `unzip LIRIS-ACCEDE-{data,annotations}.zip -d dataset/LIRIS_ACCEDE/`
2. 메타데이터 파싱 → `liris_metadata.csv` (id, name, film_id, set, v_raw, a_raw, v_var, a_var, v_norm, a_norm, quadrant_k4, mood_k7)
3. film_id는 `ACCEDEdescription.xml`의 `<movie>` 태그에서 추출
4. 공식 split 적용 + 같은 film이 여러 split에 없음 assert
5. §8 K=7 3조건 게이트 실행 (fail-fast)
6. Feature precompute: X-CLIP (512-d) + PANNs (2048-d) → `data/features/liris_panns/`
7. 산출: `liris_metadata.csv`, `splits/official_split.json`, `k7_gate_result.json`, feature precompute manifest, `distribution_report.json`

---

## 8. K=7 Mood Head 상향 3조건 게이트 (V5 순서 근거 이원화)

### 통일된 실행 순서
**조건 1 (최우선, 1분): Data sufficiency** → **조건 2 (30분): Non-interference** → **조건 3 (30분, 재활용): Learnability**

### 조건 1 (1분): Data sufficiency
- LIRIS train(40 films, ~9,800 windows) 각 7-class count ≥ 1%
- 실측 예상(§2-1): JA/TE/PE/WO 4개 class 실질 0% → **FAIL 유력**
- FAIL 시 K=4 확정, 이하 스킵

### 조건 2 (30분): Non-interference
- K=7 vs K=4 10-epoch mini-run 대조, `|ΔCCC| ≤ 0.01`
- FAIL 시 K=4 확정

### 조건 3 (30분, 조건 2 run 재활용): Learnability
- K=7 학습 val `mood_F1_macro > 14.3%`

### 순서 근거 (V5 이원화 — 두 기준을 명시 분리)

**(a) 결정력 기준 (severity)** — 실패 시 치명성 순서:
- Non-interference (primary task 저해) > Learnability (본래 목적 미달) > Data sufficiency (구조적 불가능)
- 이 관점에서는 **Non-interference가 1순위**

**(b) 실행 효율 기준 (cost-benefit)** — 확인 비용 × 예상 fail 확률:
- Data sufficiency (1분, LIRIS 실측상 fail 유력) < Non-interference (30분, 중간 확률) < Learnability (30분, Non-interference 통과 시만 의미)
- 이 관점에서는 **Data sufficiency가 1순위**

**최종 결정**: §2-1 실측(v_norm ≤ +0.30)으로 Data sufficiency의 예상 fail rate가 매우 높고 확인 비용이 1분으로 최저이므로 **실행 효율 기준 채택**. 원 피드백의 "Non-interference 1순위" 제안은 결정력 기준에서 타당하나, LIRIS 실측으로 구조적 fail이 확정된 이상 fail-fast 실행 효율이 우선.

### 게이트 의사코드
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

## 9. 구현 파일 구조

### 9-1. 신규 `model/autoEQ/train_liris/`
| 파일 | 내용 |
|---|---|
| `config.py` | TrainLirisConfig (num_mood_classes={4,7}, λ_mood=0.3, λ_gate_entropy=0.05, ccc_hybrid_w=0.3, modality_dropout_p=0.05, feature_noise_std=0.03, mixup_prob=0.5, mixup_alpha=0.4, target_shrinkage_eps=0.05, v_var_threshold, a_var_threshold, shrinkage_logic="AND", batch_size=32, lr=1e-4, wd=1e-5, epochs=40, patience=10, grad_clip=1.0, warmup=500, seed=42, use_official_split=True, pad_audio_to_10s=auto) |
| `liris_preprocess.py` | 메타데이터 파싱, V/A 변환, film_id 추출, feature precompute, variance threshold 산출 |
| `dataset.py` | `PrecomputedLirisDataset` + `official_split()` |
| `model.py` | `AutoEQModelLiris` (V3.3 `train_pseudo/model_base/model.py` 기반) |
| `losses.py` | `combined_loss_liris` — L_va_mse, L_va_ccc 독립 반환 |
| `trainer.py` | Lexicographic early stopping + Overfit auto-monitor + per-term loss 로깅 |
| `metrics.py` | CCC/Pearson/MAE/RMSE per-axis + `film_level_bootstrap_ci` (Phase 3 전용) |
| `run_train.py` | CLI |
| `eval_test.py` | 최종 평가 (bootstrap CI + 7-centroid 발동 분포) |
| `tests/` | 9종 단위 테스트 |

### 9-2. 재사용
- `train_pseudo/dataset.py::va_to_mood` (7-class GEMS)
- `train_pseudo/model_base/*` (V3.2 baseline 참조)
- `train/encoders.py` (X-CLIP/PANNs util)
- `infer_pseudo/*` — VARIANTS에 `liris_base` 추가
- `playback/*`, `scripts/eq_response_check.py`

### 9-3. 폐기
- `pseudo_label/*`, CCMovies preprocess, COGNIMUSE preprocess, σ-filter 코드, CCMovies data

---

## 10. 평가·모니터링 구현

### 10-1. Film-level cluster bootstrap (V5 범위 명시)

**적용 범위**: **Phase 3 최종 평가(LIRIS test 80 films)에만 사용**. Phase 2 ablation은 point estimate + paired t-test로 비교 (bootstrap overhead 회피, 22회 fold-level 재샘플링은 의미 희박).

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
gap = train_mean_ccc - val_mean_ccc
wandb.log({'diagnostic/train_val_ccc_gap': gap})
if gap > 0.10:
    log.warning(f"Overfit @ epoch {epoch}: train={train_mean_ccc:.3f} val={val_mean_ccc:.3f}")
```

### 10-3. Loss term 독립 로깅
```python
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
        'loss_va_mse': mse_term, 'loss_va_ccc': ccc_term,
        'loss_mood': L_mood, 'loss_gate_entropy': L_ent,
    }
```

### 10-4. Gate 진단 + Head-wise grad norm
V3 §14-4, §14-5 유지.

---

## 11. Phase 2a — 근본 설계 ablation (OAT, ~1일)

Baseline 2a-0: V/A 전략 A, K=4, PANNs, Gate, 기본 하이퍼.

| 단계 | 변경 축 | 조건 |
|---|---|---|
| 2a-1 | V/A 정규화: 전략 A vs B (min-max stretch) | 나머지 baseline |
| 2a-2 | Mood Head: K=4 vs K=7 (§8 게이트 통과 시만) | 2a-1 승자 |
| 2a-3 | Audio encoder: PANNs vs AST | 2a-2 승자 |
| 2a-4 | Fusion: Gate vs GMU (선택적) | 2a-3 승자 |

각 단계 승자를 다음 baseline으로 승격. **paired t-test + point estimate** 기반 판정.

---

## 12. Phase 2b — 하이퍼 튜닝 ablation (OAT, ~3일, V5 21 run으로 최적화)

Baseline 2b-0: Phase 2a 최종 승자.

| 단계 | 변경 축 | 값 | Run 수 |
|---|---|---|---|
| 2b-1 | λ_mood | {0, 0.1, 0.3, 0.5} | 4 |
| 2b-2 | λ_gate_entropy | {0, 0.05, 0.1} | 3 |
| 2b-3 | ccc_hybrid_w | {0, 0.3, 1.0} | 3 |
| 2b-4 | Batch size | {32, 64} | 2 |
| 2b-5 | 증강 off-ablation | 6 run 테이블 | 6 |
| **2b-6** | **Modality Dropout p** (V5: p=0은 2b-5-a와 중복이므로 결과 재사용) | **{0.05, 0.10, 0.15}** | **3** |

### 2b-5 증강 off-ablation 테이블 (OAT 엄수)
| Run | Modality Dropout | Feature Noise | Quadrant Mixup | Target Shrinkage |
|---|---|---|---|---|
| 2b-5-a | **OFF** | ON | ON | ON |
| 2b-5-b | ON | OFF | ON | ON |
| 2b-5-c | ON | ON | OFF | ON |
| 2b-5-d | ON | ON | ON | OFF |
| 2b-5-e (ALL_OFF) | OFF | OFF | OFF | OFF |
| 2b-5-f (ALL_ON, reference) | ON | ON | ON | ON |

**V5 최적화**: 2b-5-a = 2b-6 p=0이므로 2b-6에서 p=0 포인트를 제거하고 2b-5-a 결과 재사용.

**총 Phase 2b run: 4+3+3+2+6+3 = 21 run**, compute ≈ 14시간.

---

## 13. Phase 3 — 추론/재생 + 최종 평가 (~3일)

1. Phase 2 최종 모델 확정 → `infer_pseudo/model_inference.py::VARIANTS`에 `liris_base` entry 추가
2. **LIRIS test set(80 unseen films, 4,900 clips) 최종 평가** — film-level bootstrap 95% CI
3. **§19 EQ 커버리지 분석** — 모델 예측 7-centroid 발동 분포 vs ground-truth 분포 비교 (KL-divergence)
4. 샘플 영화 1편 end-to-end (analyze → timeline.json → playback → remux)
5. 청취 A/B: α_d ∈ {0.3, 0.5, 0.7}
6. 최종 보고서 작성

---

## 14. Verification

### 14-1. 코드/데이터 무결성
- `pytest model/autoEQ/train_liris/tests/ -v` (9/9 PASS)
- `liris_preprocess --verify` (split 무결성, V/A round-trip, variance threshold 발동률 assert)

### 14-2. 학습 sanity (2 epoch mini-run)
- L_total 감소 + L_va_mse/L_va_ccc/L_mood/L_gate_ent 4개 성분 독립 감소
- grad_norm head 간 10배 이내 균형
- train/val CCC gap monitor 작동

### 14-3. Primary metric 목표 (V5 bootstrap 범위 명시)
**Phase 2 (point estimate + paired t-test)**:
- LIRIS val (early stop 기준): mean_CCC ≥ 0.30 (baseline) / ≥ 0.45 (stretch)

**Phase 3 (최종 평가, bootstrap CI 포함)**:
- LIRIS test (80 unseen films, ID hold-out): mean_CCC ≥ 0.28 + **95% CI (film-level cluster bootstrap)**
- 축별 ccc_v, ccc_a 각각 보고
- §19 커버리지 분석: 예측 분포 vs ground-truth 분포 KL-divergence

### 14-4. End-to-end
- 샘플 1편 `analyze → timeline.json → playback → remuxed.mp4` 무오류
- VAD 대사 보호 적용 확인

---

## 15. 리스크 및 트레이드오프 (V5 용어 교정)

| # | 리스크 | 대응 |
|---|---|---|
| 1 | Valence positive 편향으로 K=7 불가 (§2-1) | K=4 default, §8 fail-fast 게이트, §19 학술 보고 |
| 2 | Variance threshold per-axis 조정 | v=0.117, a=0.164 AND 조건 (6~10% 발동) |
| 3 | PANNs 4초 입력 품질 | Phase 0 sanity 사전 차단 |
| 4 | Overfit (183만 param / 9,800 windows) | Overfit auto-monitor + 증강 4종 + patience 10 |
| 5 | **외부 OOD(Out-of-Distribution) 평가 부재** — **LIRIS test는 ID(In-Distribution) hold-out이므로 distribution shift robustness는 검증되지 않음** | LIRIS test 80 unseen films로 film-level within-distribution generalization 평가만 수행. Cross-dataset OOD 평가(EmoMusic/DEAM/MediaEval/COGNIMUSE)는 Future Work로 명시 |
| 6 | Congruence 미복원의 학술 축소 | 보고서 "실증적 단순화" 방어 |
| 7 | LIRIS split 1:1:2 skew (40/40/80) | 필요 시 `--use_full_learning_set`로 set 1+2 병합 옵션 |
| 8 | 증강 효과 약화 | Phase 2b-5 off-ablation (6 run) |
| 9 | EQ 프리셋 커버리지 편향 (§19) | 학술적 발견으로 정직 보고 + LIRIS test에서 발동 분포 측정 |

---

## 16. 결론

**"V3.3 학습 프로세스 외곽 × V3.2 모델 내부 × LIRIS 실측 기반 교정 × V5 학술 정확성 완성"** 하이브리드.

V5 핵심 교정:
1. **"in-domain OOD" 용어 모순 해소** — LIRIS test는 **in-distribution (ID) hold-out**, 외부 OOD는 Future Work
2. **Baveye 인용 수정** — 2015 IEEE TAC로 통일, MAC 2014 표현 제거
3. **§19-4 실측 연결** — ground-truth 7-centroid 분포 + Phase 3 KL-divergence 측정
4. **Runtime 일치** — 21 run (중복 제거 후), compute ~14시간, 총 ~22시간
5. **Bootstrap 범위** — Phase 3 최종 평가 전용, Phase 2는 point estimate
6. **2b-6 중복 제거** — p=0을 2b-5-a와 공유, 4→3 run
7. **§8 순서 근거 이원화** — 결정력 vs 실행 효율 기준 명시 분리

진행 순서: Phase 0 (50분) → Phase 1 (~1일 실작업) → Phase 2a (~1일) → Phase 2b (~3일, 21 run) → Phase 3 (~3일). **총 compute ~22시간, 문서 포함 ~3주**.

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
V5: p=0.05 default, Phase 2b-6 ablation {0.05, 0.10, 0.15} (p=0은 2b-5-a 공유).

### 증강 2: Feature Noise (model.forward)
`x' = x + N(0, σ²), σ=0.03, 대칭, dropped sample은 bypass`

### 증강 3: Quadrant Mixup (collate_fn, prob=0.5)
같은 사분면 쌍, λ~Beta(0.4, 0.4)→[0.1, 0.9] shrink, visual/audio/v/a/v_var/a_var 동일 λ 선형결합, mood label primary 유지.

### 증강 4: Target Shrinkage (collate_fn)
```python
trigger = (v_var > 0.117) & (a_var > 0.164)  # AND, ~6-10% 발동
if trigger: v *= 0.95, a *= 0.95
```

### 순서
DataLoader → collate_fn [Mixup → Target Shrinkage] → model.forward [Modality Dropout → Feature Noise] → Gate → Fusion → Heads → Loss

---

## 19. 🆕 EQ 프리셋 커버리지 한계 — 학술적 정직 보고 (V5 인용·실측 연결 교정)

### 19-1. 문제 진술

LIRIS-ACCEDE Discrete의 9,800 clip 실측(§2-1)에서 valenceValue는 이론 범위 [1, 5]와 달리 실제로 [1.33, 3.59]에 분포한다. 이는 **Baveye et al. (2015, IEEE Transactions on Affective Computing)** "LIRIS-ACCEDE: A Video Database for Affective Content Analysis"에 기술된 crowdsourced pairwise ranking → affective rating 회귀 과정의 수치적 특성에 기인한다. `(v−3)/2` 변환 후 `v_norm ∈ [−0.84, +0.30]`로, V3.2 명세 §2-5의 7 GEMS mood centroid 중 **positive Valence 4개(Joyful Activation V=+0.7, Tenderness V=+0.4, Peacefulness V=+0.5, Wonder V=+0.5)에 대해 학습 신호가 구조적으로 불충분**하다.

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

### 19-4. 정량 측정 — Phase 0 실측 연결 + Phase 3 분석 (V5 교정)

**Step 1 (Phase 0, §6-2에서 집계 완료 예정)**: 9,800 clip 전체의 ground-truth V/A 좌표를 7-centroid Euclidean으로 매핑한 class 분포 표 작성.

```python
# Phase 0 §6-2 산출 예시 (실측 기반)
ground_truth_k7 = {
    'Sadness':        'XX%',   # LIRIS 도달 가능 (V=-0.6)
    'Tension':        'XX%',   # LIRIS 도달 가능 (V=-0.6, A=+0.7)
    'Power':          'XX%',   # 경계 (V=+0.2)
    'Peacefulness':   '~0%',   # 도달 불가 (V=+0.5)
    'Joyful Activation': '~0%', # 도달 불가 (V=+0.7)
    'Tenderness':     '~0%',   # 도달 불가 (V=+0.4)
    'Wonder':         '~0%',   # 도달 불가 (V=+0.5)
}
```

**Step 2 (Phase 3)**: 학습된 모델이 LIRIS test set (80 unseen films, 4,900 clips, ID hold-out)을 추론할 때 각 mood centroid 발동 분포를 집계하고, **ground-truth 분포와 KL-divergence 측정**:

```python
# eval_test.py
pred_mood_hits = defaultdict(int)
for clip in liris_test_set:
    v_pred, a_pred = model(clip)
    mood = va_to_mood_k7(v_pred, a_pred)
    pred_mood_hits[mood] += 1

gt_dist   = normalize(ground_truth_k7)  # Phase 0 집계
pred_dist = normalize(pred_mood_hits)   # Phase 3 추론
kl_div    = scipy.stats.entropy(pred_dist, gt_dist)

# 기대 결과:
# - gt_dist ≈ pred_dist (모델이 ground-truth 분포를 잘 복제) → kl_div 작음
# - 단 두 분포 모두 JA/TE/PE/WO < 10% (under-represented, LIRIS 구조적 특성)
```

### 19-5. 보고서 방어 문단 (V5 인용·용어 교정 draft)

> "LIRIS-ACCEDE Discrete의 scalar V/A value는 이론 범위 [1, 5]와 달리 실제로는 Valence [1.33, 3.59] · Arousal [1.32, 4.54]에 분포한다(§2-1). 이는 **Baveye et al. (2015, IEEE TAC)**에 기술된 crowdsourced pairwise ranking을 affective rating으로 변환하는 회귀 과정의 수치적 특성에 기인하며, V3.2 명세의 7 GEMS centroid 중 positive Valence 4개(Joyful Activation, Tenderness, Peacefulness, Wonder)에 대해 학습 신호가 구조적으로 불충분하다. 본 연구는 이를 은폐하지 않고 **LIRIS 데이터셋의 구조적 커버리지 한계**로 명시적으로 보고하며, 이 4개 mood의 EQ 프리셋은 학습된 모델의 **LIRIS ID test(80 unseen films)**에서 발동 빈도가 낮게 측정되었다(§19-4 분석: ground-truth 대비 예측 분포 KL-divergence = X, JA/TE/PE/WO 합산 < Y%). 외부 OOD 일반화(다른 데이터셋)는 본 연구 범위 밖이며, 후속 연구에서 supplementary positive-valence data(EmoMusic, DEAM, MediaEval 2015-2018)를 병용한 multi-dataset 학습으로 커버리지를 확장할 수 있다. 본 연구가 제안한 10-band Biquad peaking EQ 프리셋 자체는 7 mood 전부에 대해 V3.2의 음향 심리학적 근거(Juslin & Laukka 2003, Arnal 2015, Bowling 2017 등)로부터 도출되어 있으므로, 데이터 확보 시 그대로 활용 가능한 형태로 보존되어 있다."

### 19-6. 학술적 기여 3가지
1. 데이터 감수 성실성 (분포 구조 공개 + KL-divergence 정량 보고)
2. 후속 연구 방향 제시 (multi-dataset 학습 명제)
3. LIRIS 기반 커뮤니티 공통 제약 공론화

---

## 20. Runtime 추정 (V5 21 run 반영)

| Phase | 작업 | Compute | 문서/분석 | 누적 |
|---|---|---|---|---|
| 0 | PANNs sanity + V/A scatter + K7 centroid 집계 + variance | 40분 | 10분 | 50분 |
| 1 | zip 해제, CSV, split, K=7 게이트, feature precompute | 4시간 | 4시간 | ~1일 실작업 |
| 2a | 4개 ablation × 40min | 3시간 | 4시간 | ~1일 |
| 2b | **21 run × 40min** (λ_mood 4 + λ_ent 3 + ccc_w 3 + batch 2 + aug 6 + dropout_p 3) | **14시간** | 8시간 | ~3일 |
| 3 | 추론 통합 + §19 KL-divergence 분석 + 청취 A/B + 보고서 | 1일 | 2일 | ~3일 |

**총 compute ≈ 22시간**, **총 소요 ≈ 3주 (문서/분석 포함)**.
