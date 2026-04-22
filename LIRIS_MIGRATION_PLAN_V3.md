# LIRIS-ACCEDE 전환 플랜 V3 (실측 기반 교정 + 피드백 2차 반영)

## 0. 변경 이력

### V2 → V3 (2026-04-20, 실측 기반 대규모 교정)

1. **§3 신설: LIRIS 9,800 clip 전체 실측 결과 반영** — V/A 실제 범위 [1.33, 3.59] / [1.32, 4.54] 확인. V2의 "이론 범위 1~5" 가정은 valence positive 쪽에서 무너짐.
2. **V/A 정규화 공식 재설계**: 단순 `(v−3)/2` 대신 **per-axis 실제 분포 기반 재스케일** 옵션 추가. K=7 centroid도 LIRIS 분포 내 재배치 검토.
3. **variance threshold 0.15 폐기**: 실측에서 `max(v_var, a_var) > 0.15`가 50.6% 발동 → 무의미. **per-axis p75 기반 threshold** 재설계.
4. **§8 K=7 3조건 게이트 순서 재정렬 (fail-fast)**: (1) Non-interference → (2) Data sufficiency → (3) Learnability.
5. **§11 Phase 0 sanity 재설계**: "4s vs 4s padded" → **"full clip(8~12s) vs 4s center crop"** (production 시나리오 재현).
6. **§13 Phase 2 → 2a/2b 분리 + OAT 명시**: 2a(근본 설계, 순차 OAT) → 2b(하이퍼 튜닝, 2a 확정 후 한 축씩).
7. **§14 Bootstrap을 film-level cluster resampling으로** 구현 명시.
8. **§14 Overfit auto-monitor** 추가: `train_mean_ccc − val_mean_ccc > 0.10` 감지.
9. **§15 Loss term 개별 로깅** 명시: L_va, L_mood, L_gate_ent wandb 독립 metric.
10. **§17 K=4 fallback을 default 가정**: 실측 분포상 K=7 게이트 통과 가능성 낮음. K=7은 "상향 검토" 위치로 이동.

### V1 → V2 변경 (참고)
- "Pareto" → "Lexicographic" 용어 교정
- "Label Smoothing" → "Target Shrinkage" 용어 교정
- CCC 단독 primary, Pearson/MAE는 diagnostic
- V3.2 원래 설계(PANNs + Gate + Multi-task) 모델 내부 복귀

---

## 1. Context

원래 V3.2 명세서로 LIRIS-ACCEDE 학습 계획이었으나 데이터 확보 불가로 V3.3 축소 명세(CCMovies 9편 + pseudo-label)로 전환. 2026-04-20 LIRIS 도착 → 재시작.

- 사용자 요구: **V3.3 학습 프로세스 외곽 유지 + V3.2 모델 내부(PANNs + Gate + Multi-task) 복귀**
- V3는 피드백 2차 라운드에서 지적된 **실측 데이터 가정 오류**를 교정

---

## 2. 사용자 확정 결정사항 (V1~V3 누적)

1. Congruence Head + Negative Sampling: 미복원
2. Mood Head: K=7 GEMS 복원 (단 §8 3조건 게이트 통과 시, 실측상 K=4 fallback이 default 예상)
3. 코드 구조: 새 `model/autoEQ/train_liris/` 폴더
4. 베이스 모델: **X-CLIP + PANNs CNN14 + Gate Network + Intermediate Fusion + Multi-task**
5. 평가 프레이밍: **CCC 단독 primary (AVEC 표준)**, Pearson·MAE는 diagnostic
6. Early stopping: **Lexicographic** `(mean_CCC, mean_Pearson, −mean_MAE)`, patience=10

---

## 3. LIRIS-ACCEDE 실측 결과 (V3 신규, 9,800 clip 전체 통계)

### 3-1. V/A scalar value 실제 범위 — 이론과 괴리 존재

| 축 | 이론 (README) | 실제 min | 실제 max | mean | std |
|---|---|---|---|---|---|
| valenceValue | 1 ~ 5 | **1.33** | **3.59** | 2.80 | 0.63 |
| arousalValue | 1 ~ 5 | 1.32 | 4.54 | 2.46 | 0.95 |

**Percentile 상세**:
- valenceValue: p25=2.35, p50=3.02, p75=3.32, p95=3.52, p99=3.58
- arousalValue: p25=1.59, p50=2.22, p75=3.20, p95=4.25, p99=4.48

**→ 중요 발견**: Valence의 positive 쪽이 **사실상 잘려 있음** (p99=3.58, max=3.59). Arousal은 [1.32, 4.54]로 더 넓지만 여전히 positive extreme 제한.

### 3-2. V/A 변환 후 범위 (본 플랜 선택 공식: `(raw − 3) / 2`)

| 축 | 변환 후 min | max | mean | [-1, +1] 범위 내 비율 |
|---|---|---|---|---|
| v_norm | **−0.84** | **+0.30** | −0.10 | 100% |
| a_norm | −0.84 | +0.77 | −0.27 | 100% |

**→ 함의**: V3.2 Mood centroid와 비교:
| Centroid | V 좌표 | A 좌표 | LIRIS에서 도달 가능? |
|---|---|---|---|
| Tension | −0.6 | +0.7 | ✅ (v_min=-0.84, a_max=0.77) |
| Sadness | −0.6 | −0.4 | ✅ |
| Peacefulness | +0.5 | −0.5 | ❌ (v_max=+0.30) |
| **Joyful Activation** | **+0.7** | **+0.6** | ❌ (v 도달 불가) |
| **Tenderness** | **+0.4** | −0.2 | ❌ (v 경계 근처) |
| **Power** | **+0.2** | +0.8 | ⚠️ (v 가능, a_max=0.77로 근소 미달) |
| **Wonder** | **+0.5** | +0.3 | ❌ (v 도달 불가) |

**→ 결론**: 본 플랜의 default 정규화 `(v−3)/2`를 사용하면 LIRIS에서 **Joyful Activation / Peacefulness / Tenderness / Wonder 4개 class가 실질적으로 비어 있음**. K=7 Mood Head는 구조적으로 학습 불가. → **K=4 quadrant fallback이 default 가정**.

### 3-3. Variance 분포 (Target Shrinkage threshold 재설계 근거)

| 컬럼 | min | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|
| valenceVariance | 0.093 | 0.103 | 0.108 | 0.117 | 0.131 | 0.164 |
| arousalVariance | 0.116 | 0.135 | 0.150 | 0.164 | 0.194 | 0.218 |

**V2 설정 `threshold=0.15` 발동 비율**:
- `valenceVariance > 0.15`: **1.9%** (너무 적음)
- `arousalVariance > 0.15`: **49.6%** (너무 많음)
- `max(v_var, a_var) > 0.15`: **50.6%** (절반)

**→ 무의미**: V3.3의 CCMovies pseudo-label std와 LIRIS multi-annotator variance는 scale이 다름. 0.15 고정 threshold는 LIRIS에서 per-axis 분포 차이를 무시 → "conditional"의 의미 상실.

**V3 재설계 (per-axis p75 기반)**:
```python
# Phase 1 전처리에서 자동 계산
v_thr = np.percentile(valenceVariance, 75)   # ≈ 0.117
a_thr = np.percentile(arousalVariance, 75)   # ≈ 0.164

# 샘플 발동 조건 (OR)
trigger = (v_var > v_thr) | (a_var > a_thr)
# 결과: 약 25% + 25% − overlap ≈ 40~45% 중 보수적 적용 필요
# 또는 AND 조건으로 ~6~10% 발동
```
또는 **upper quartile 기반 AND 조건** (더 엄격):
```python
trigger = (v_var > v_thr) & (a_var > a_thr)   # 약 6~10% 발동
```
Phase 1 분포 시각화 후 AND vs OR 선택. **원칙**: "conditional"의 목적은 **고-uncertainty 샘플만** 대상이므로 발동 비율 **10% 내외** 유지.

### 3-4. 4-사분면 분포 (정규화 후 `v_norm=0, a_norm=0` 기준)

| 사분면 | count | 비율 |
|---|---|---|
| HVHA (+V, +A) | 1,074 | **11.0%** (최저) |
| HVLA (+V, −A) | 3,944 | 40.2% (최다) |
| LVHA (−V, +A) | 1,819 | 18.6% |
| LVLA (−V, −A) | 2,963 | 30.2% |

**→ 함의**: 
- K=4 quadrant는 min class 11%로 충분히 학습 가능 (chance baseline 25% 대비 macro F1 target 현실적)
- 단, HVHA(+V, +A)가 실제로는 v_norm ∈ [0, +0.30] 좁은 영역에 몰려 있음 (2D scatter 필요)
- Stratified sampling 기반 split은 가능, 단 test set에서 HVHA variance 높을 수 있음

---

## 4. V/A 정규화 전략 (V3 재설계)

§3의 실측 결과를 바탕으로 3가지 전략 비교. Phase 1에서 선택:

### 전략 A: 단순 `(raw − 3) / 2` (V2 default)
- **장점**: V3.2 명세 원문과 직관적 일관성, Mood centroid 좌표 그대로 사용
- **단점**: V 양성 쪽 `[+0.30, +1.0]` 영역이 비어 K=7의 4개 class 학습 불가
- **적용 시**: K=4 quadrant fallback을 default로 확정

### 전략 B: per-axis min-max stretch
```python
v_norm = 2 * (v_raw - v_raw.min()) / (v_raw.max() - v_raw.min()) - 1
# v_raw ∈ [1.33, 3.59] → v_norm ∈ [-1, +1]
a_norm = 2 * (a_raw - a_raw.min()) / (a_raw.max() - a_raw.min()) - 1
# a_raw ∈ [1.32, 4.54] → a_norm ∈ [-1, +1]
```
- **장점**: 데이터 활용도 최대화, K=7 centroid 도달 가능
- **단점**: V/A 비대칭 스케일 → Mood centroid 좌표의 의미 왜곡 (V3.2 §2-5 매핑 불변 가정 훼손)
- **적용 시**: Mood centroid도 비례 재스케일 필요

### 전략 C: symmetric stretch (대칭 scale 유지)
```python
scale = max(abs(raw.min() - 3), abs(raw.max() - 3))  # 가장 큰 이탈
v_norm = (v_raw - 3) / scale
# → v_norm ∈ [-1, v_max_ratio], mean preservation
```
- **장점**: 중심(3)을 0에 고정, 대칭성 유지
- **단점**: V의 positive 쪽은 여전히 [+0.30] 근처에서 멈춤 (scale=2 유지 시 전략 A와 동일)

**V3 권장**: **전략 A를 default**로 채택 + **Phase 2a ablation에서 전략 B와 비교** (K=7 학습 가능성 테스트를 위한 실험적 대안).

---

## 5. 구성 요약 ("V3.3 외곽" × "V3.2 내부")

| 계층 | 선택 | 출처 |
|---|---|---|
| Visual encoder | X-CLIP base-patch32 (frozen, 512-d) | 공통 |
| Audio encoder | PANNs CNN14 (frozen, 2048-d) + Linear Projection 2048→512 | V3.2 원복 |
| Fusion | Gate Network (1024→256→2 softmax) + Intermediate Fusion → 1024-d | V3.2 원복 |
| VA Head | 1024 → 256 → 2 | 공통 |
| **Mood Head** | **1024 → 256 → 4 (K=4 quadrant, default)** · K=7 상향은 §8 게이트 통과 시 | §3-2 실측 기반 재평가 |
| Congruence Head | 없음 | 사용자 결정 |
| Modality Dropout | p=0.05 전체 무조건 | V3.3 |
| Feature Noise | σ=0.03 대칭 | V3.3 |
| Quadrant Mixup | prob=0.5, α=0.4 | V3.3 |
| **Target Shrinkage** | **per-axis p75 threshold**, ε=0.05 | §3-3 재설계 |
| V/A loss | CCC hybrid w=0.3 (Phase 2b ablation: {0, 0.3, 1.0}) | V3.3 + §2-5 |
| Mood loss | CE(K=4), λ_mood=0.3 (K=7 승격 시 λ 재검토) | V3.3 구조 |
| Gate entropy | λ_ent=0.05 (Phase 2b ablation: {0, 0.05, 0.1}) | V3.2 복원 + §2-8 |
| Early stopping | **Lexicographic** (mean_CCC, mean_Pearson, −mean_MAE), patience=10 | §2-2 교정 |
| Primary metric | **mean_CCC** (AVEC 표준) | §2-3 |
| Bootstrap CI | **Film-level cluster resampling** | §2-3 교정 |
| **Overfit 감지** | **train−val mean_CCC gap > 0.10 자동 경고** | §2-5 |

---

## 6. 실제 파일 구조 (V2 §3 유지, 변환 공식만 §4에서 재선택)

| 파일 | 내용 |
|---|---|
| `ACCEDEranking.txt` (9,800 rows) | id, name, valenceRank, arousalRank, **valenceValue** [1.33, 3.59], **arousalValue** [1.32, 4.54], valenceVariance, arousalVariance |
| `ACCEDEsets.txt` | 공식 split: set=0(test 4,900/80영화), set=1(learn 2,450/40), set=2(val 2,450/40) |
| `ACCEDEdescription.xml` | 각 clip의 `<movie>` 태그 → film_id 추출 소스 |

**본 플랜 split 정책**: **LIRIS 공식 split(40/40/80)을 기본값**, `--use_official_split=True`. 타 LIRIS 논문과 비교 가능성 확보.

---

## 7. V3.2 ↔ V3.3 ↔ V3 플랜 정합성 매트릭스 (V2 매트릭스 + V3 교정 반영)

| 항목 | V3.2 요구 | V3.3 구현(CCMovies) | V3 플랜 | 비고 |
|---|---|---|---|---|
| 학습 데이터 | LIRIS 9,800 / 160편 | CCMovies 1,289 / 9편 | **LIRIS 9,800 / 160편** | V3.2 복귀 |
| Split | Train 120/Val 20/Test 20 | LOMO 9-fold | **LIRIS 공식 40/40/80** (기본), 자체 split은 opt-in | §6 |
| **V/A range** | [-1, +1] | [-1, +1] | **전략 A `(v−3)/2` 기본, v_norm ∈ [-0.84, +0.30] · a_norm ∈ [-0.84, +0.77]** · Phase 2a에서 전략 B ablation | §3-2 · §4 |
| Per-sample std | — | ensemble std | **valenceVariance / arousalVariance (p75 기반 threshold)** | §3-3 |
| Video encoder | X-CLIP | X-CLIP | X-CLIP | 불변 |
| Audio encoder | PANNs | AST | **PANNs** | 사용자 결정 |
| Fusion | Gate + Intermediate | GMU | **Gate + Intermediate** | 사용자 결정 |
| **Mood Head** | K=7, 실패 시 K=4 | K=4 | **K=4 default** · K=7은 §8 게이트 상향 검토 | §3-2 실측 Valence 편향 |
| Congruence | 포함 | 제거 | 미복원 | 사용자 |
| V/A loss | MSE | CCC hybrid w=0.3 | CCC hybrid w=0.3 · 2b ablation {0, 0.3, 1.0} | §2-5 |
| Gate entropy | λ=0.05 | N/A | λ=0.05 · 2b ablation {0, 0.05, 0.1} | §2-8 |
| **Feature Aug** | 명시 없음 | Noise + Mixup + Shrinkage | **동일 + Shrinkage threshold per-axis p75** | §3-3 |
| Early stopping | val CCC | "Pareto" (실제 lex) | **Lexicographic** (CCC, Pearson, −MAE) | §2-2 교정 |
| **Primary metric** | CCC | CCC | **CCC 단독 primary** · Pearson/MAE diagnostic | §2-3 |
| **Bootstrap** | — | — | **Film-level cluster bootstrap 95% CI** | §2-3 |
| **Overfit monitor** | — | — | **train−val CCC gap > 0.10 경고** | §2-5 |
| **Loss 로깅** | total만 | total만 | **L_va · L_mood · L_gate_ent 독립 wandb metric** | §2-6 |
| EQ 프리셋 | 7×10 Biquad | 동일 | 동일 | 불변 |
| VAD / 대사 보호 | Silero VAD | 동일 | 동일 | 재사용 |
| JSON timeline | v1.0 | 구현 완료 | 그대로 재사용 | — |

---

## 8. Mood K=7 유지 3조건 게이트 (V3 재설계, fail-fast 순서)

§3-2 실측 결과 Valence positive 쪽 도달 불가가 확인되어 **K=4가 default**. K=7 상향은 아래 **3조건을 모두 통과**한 경우만:

### 조건 1 (최우선): Non-interference
- K=7 vs K=4 짧은 **10-epoch mini-run** 대조
- `abs(val_mean_CCC_K7 − val_mean_CCC_K4) ≤ 0.01`
- **실패 시 즉시 K=4 확정, 이후 조건 skip** (fail-fast)
- 근거: primary V/A task를 저해하면 K=7 채택 의미 상실

### 조건 2: Data sufficiency
- LIRIS train split(40 films, ~2,450 clip × 3~5 window ≈ ~9,800 windows)에서 각 7-class sample count **≥ 전체의 1%**
- **실측 예상**: v_norm 최대 +0.30으로 Joyful Activation(+0.7, +0.6) 등 4개 class가 실질 0% 할당 → 실패 유력

### 조건 3: Learnability
- K=7 학습 후 val `mood_F1_macro > random baseline 14.3% (1/7)`
- 조건 2 통과 시에만 체크 (논리적으로)

**게이트 코드**:
```python
# 순차 실행, fail-fast
if abs(ccc_k7 - ccc_k4) > 0.01:
    return "K=4 (non-interference 실패)"

class_ratios = count_per_class(train_data, k=7) / len(train_data)
if class_ratios.min() < 0.01:
    return "K=4 (data sufficiency 실패)"

if k7_mood_f1_macro <= 0.143:
    return "K=4 (learnability 실패)"

return "K=7 PASS"
```

---

## 9. Phase 구조 개요 (V3 재구성)

| Phase | 기간 | 목적 | 핵심 산출물 |
|---|---|---|---|
| **Phase 0** | 30분~1시간 | 실측 sanity check (full clip vs 4s crop, V/A 2D scatter, variance 분포 시각화) | `runs/phase0_sanity/report.json` |
| **Phase 1** | 1주 | LIRIS 전처리, feature precompute, K=7 3조건 게이트 실행, split 확정 | `data/features/liris_panns/*.pt`, `k7_gate_result.json` |
| **Phase 2a** | 1주 | 근본 설계 ablation (OAT): Audio encoder, Fusion, V/A 정규화 전략 | `runs/phase2a_*/` |
| **Phase 2b** | 1주 | 하이퍼 튜닝 ablation (OAT): λ_mood, λ_gate_ent, ccc_hybrid_w, batch_size, aug on/off | `runs/phase2b_*/` |
| **Phase 3** | 3~5일 | 최종 모델 선택, COGNIMUSE OOD 평가, 추론/재생 파이프라인 통합 | `runs/final/`, 청취 A/B 결과 |

---

## 10. Phase 0 — Sanity Check (30분~1시간, Phase 1 착수 전 필수)

### 10-1. PANNs 4초 입력 품질 검증 (§2-7 재설계)

**V2 오류**: "4s crop vs 4s padded to 10s" 비교는 padding이 silence이므로 정보량 동일 → 유사도 과대평가.

**V3 수정**: **"full clip (8~12s) vs 4s center crop"** 비교 — 실제 production 손실 측정.

```python
import torch, numpy as np
from scipy.spatial.distance import cosine

def panns_sanity():
    clips = random.sample(liris_clips, 10)
    sims = []
    for clip in clips:
        full_audio = load_audio(clip, sr=16000)          # 8~12s
        t_full = len(full_audio) / 16000
        # center crop 4s
        start = int((t_full - 4) / 2 * 16000)
        crop = full_audio[start : start + 4 * 16000]

        feat_full = panns_cnn14(full_audio)              # 2048-d
        feat_crop = panns_cnn14(crop)                    # 2048-d
        sims.append(1 - cosine(feat_full, feat_crop))
    return {
        'mean_cos_sim': np.mean(sims),
        'std_cos_sim': np.std(sims),
        'per_clip': sims,
    }
```

**판정**:
- `mean ≥ 0.90` → 4s crop 그대로 진행 (V3 default)
- `0.80 ≤ mean < 0.90` → 주의. 4초 feature 3개 average pooling으로 12s 커버 고려
- `mean < 0.80` → PANNs 4초 부적합. **대안 3개**:
  1. 10초 zero-padding (V2 default, 정보는 복원 안 됨)
  2. Multi-chunk average (2~3개 4s chunk → PANNs 평균)
  3. AST로 회귀 (가변 길이 robust)

### 10-2. V/A 2D scatter + 7-centroid 할당 집계

```python
v_norm = (valenceValue - 3) / 2
a_norm = (arousalValue - 3) / 2

# 7-centroid Euclidean 매핑
mood_k7 = [va_to_mood(v, a) for v, a in zip(v_norm, a_norm)]
class_counts = Counter(mood_k7)

# 각 class별 비율 → K=7 data sufficiency 선제 판정
```

산출: `runs/phase0_sanity/va_scatter.png`, `class_counts_k7.json`.

### 10-3. Variance 분포 시각화

```python
plt.hist(valenceVariance, bins=50)
plt.axvline(np.percentile(valenceVariance, 75), color='r')  # p75 threshold
plt.hist(arousalVariance, bins=50, alpha=0.5)

# 각 threshold 발동 비율 테이블 생성
thresholds = [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20]
for t in thresholds:
    v_trig = (valenceVariance > t).mean()
    a_trig = (arousalVariance > t).mean()
    print(f"threshold={t}: v_trig={v_trig:.1%}, a_trig={a_trig:.1%}")
```
산출: Target Shrinkage threshold 재설정 근거 표.

---

## 11. Phase 1 — LIRIS 데이터 준비 (1주)

### 11-1. zip 해제 + 메타데이터 파싱
```bash
unzip /Users/jongin/Downloads/LIRIS-ACCEDE-{data,annotations}.zip -d dataset/LIRIS_ACCEDE/
```

### 11-2. 전처리 CSV 생성
`liris_preprocess.py --build_csv` 산출:
- `id, name, film_id, set, v_raw, a_raw, v_var, a_var, v_norm, a_norm, quadrant_k4, mood_k7`
- v_norm/a_norm: 전략 A (`(v−3)/2`) 기본, `--normalization=B_stretch`로 전략 B 선택
- film_id: `ACCEDEdescription.xml`의 `<movie>` 태그에서 추출
- set: `ACCEDEsets.txt`에서 로드

### 11-3. K=7 3조건 게이트 실행 (§8, fail-fast)
1. (조건 2 우선 체크, 비용 최저) Data sufficiency: 7-class count ≥ 1%
   - **실측 예상**: Joyful Activation/Wonder/Tenderness/Peacefulness 4개 class 실질 0 → FAIL 유력
   - FAIL 시 K=4 확정, 조건 1·3 skip
2. (조건 1) Non-interference mini-run: K=7 vs K=4 10 epoch 대조, ΔCCC ≤ 0.01
3. (조건 3) Learnability: F1_macro > 14.3%

**실제 실행 순서 권장**: 조건 2 → 조건 1 → 조건 3 (조건 2 비용이 CSV 집계 1분으로 가장 저렴). §8의 "non-interference 1순위"는 **조건 2 통과 가정 하**의 순서.

### 11-4. Feature precompute
- X-CLIP (512-d) + PANNs (2048-d), Phase 0에서 확정한 입력 길이 정책 적용
- 출력: `data/features/liris_panns/{liris_visual.pt, liris_audio.pt, liris_metadata.pt, manifest.json}`
- 예상 시간: MPS 기준 9,800 clip × ~1s ≈ **3시간**, resumable manifest

### 11-5. Variance threshold 확정
§3-3 분포 기반 per-axis p75 threshold 계산 → config에 고정값 저장:
```python
{
    "target_shrinkage": {
        "v_var_threshold": 0.117,   # ≈ p75(valenceVariance)
        "a_var_threshold": 0.164,   # ≈ p75(arousalVariance)
        "logic": "AND",              # 보수적: 약 6~10% 발동
        "eps": 0.05
    }
}
```

### 11-6. 산출물 체크리스트
- [ ] `liris_metadata.csv` (9,800 rows)
- [ ] `splits/official_split.json`
- [ ] `k7_gate_result.json` (fail/pass + 이유)
- [ ] `feature precompute manifest`
- [ ] `distribution_report.json`

---

## 12. 구현 파일 구조

### 12-1. 신규 (`model/autoEQ/train_liris/`)
| 파일 | 내용 |
|---|---|
| `config.py` | `TrainLirisConfig` (feature_dir, audio_raw_dim=2048, fused_dim=1024, num_mood_classes={4,7}, λ_mood=0.3, λ_gate_entropy=0.05, ccc_hybrid_w=0.3, modality_dropout_p=0.05, feature_noise_std=0.03, mixup_prob=0.5, mixup_alpha=0.4, **target_shrinkage_eps=0.05, v_var_threshold, a_var_threshold, shrinkage_logic="AND"**, batch_size=32, lr=1e-4, wd=1e-5, epochs=40, patience=10, grad_clip=1.0, warmup=500, seed=42, **normalization={"A_linear","B_stretch"}**, **use_official_split=True**, **pad_audio_to_10s=auto**) |
| `liris_preprocess.py` | 메타데이터 파싱, V/A 변환 (전략 A/B), film_id 추출, feature precompute, variance threshold 자동 산출 |
| `dataset.py` | `PrecomputedLirisDataset` + `official_split()` + `film_level_split()` (opt-in) |
| `model.py` | `AutoEQModelLiris` (V3.3 `model_base/` 기반, MoodHead 1024→256→{4,7}) |
| `losses.py` | `combined_loss_liris` (CCC hybrid VA + K CE + gate entropy, 각 term 독립 반환) |
| `trainer.py` | **Lexicographic early stopping** + **Overfit auto-monitor** + **per-term loss 로깅** |
| `metrics.py` | `compute_ccc` / `pearson` / `mae` / `rmse` per-axis + **`film_level_bootstrap_ci`** |
| `run_train.py` | CLI |
| `eval_test.py` | 최종 평가 (bootstrap CI 자동) |
| `tests/` | 9종 (split 무결성, V/A 변환 round-trip, K4/K7 매핑, CCC/Pearson 공식, dropout shape, mixup shape, **film_level_bootstrap correctness**) |

### 12-2. 재사용 (변경 없음)
- `train_pseudo/dataset.py::va_to_mood` (7-class GEMS)
- `train_pseudo/model_base/*` (V3.2 baseline 참조)
- `train/encoders.py` (X-CLIP / PANNs encoder utils)
- `infer_pseudo/*` (VARIANTS에 `liris_base` 추가)
- `playback/*`, `scripts/eq_response_check.py`

### 12-3. 폐기
- `pseudo_label/*`, CCMovies/COGNIMUSE preprocess, σ-filter 코드, CCMovies data

---

## 13. Phase 2 — 학습 ablation (V3 재설계, 2a/2b 분리 + OAT)

**OAT 원칙**: 한 번에 한 축만 변경, 나머지는 baseline 고정. 한 단계 승자를 다음 단계 baseline으로 승격.

### 13-1. Phase 2a — 근본 설계 ablation (순차 실행)

**Baseline (2a-0)**: V/A = 전략 A, K=4, PANNs, Gate, 기본 하이퍼 → **reference run**

| 단계 | 변경 축 | 조건 | 승자 판정 |
|---|---|---|---|
| **2a-1** | V/A 정규화: 전략 A vs B | 나머지 baseline | val mean_CCC 높은 쪽 + K=7 가능성 |
| 2a-2 | Mood Head: K=4 vs K=7 (조건부) | 2a-1 승자 + §8 게이트 통과 시만 | 승자 확정 |
| 2a-3 | Audio encoder: PANNs vs AST | 2a-2 승자 | paired t-test |
| 2a-4 | Fusion: Gate vs GMU (선택적) | 2a-3 승자 | paired t-test |

각 단계 완료 후 wandb report 생성, sign-off 후 다음 단계.

### 13-2. Phase 2b — 하이퍼 튜닝 ablation (2a 확정 후)

**Baseline (2b-0)**: Phase 2a 최종 승자 config

| 단계 | 변경 축 | 값 |
|---|---|---|
| 2b-1 | `λ_mood` | {0, 0.1, 0.3, 0.5} |
| 2b-2 | `λ_gate_entropy` | {0, 0.05, 0.1} |
| 2b-3 | `ccc_hybrid_w` | {0, 0.3, 1.0} |
| 2b-4 | Batch size | {32, 64} |
| 2b-5 | 증강 on/off | 4종 증강 각각 OFF 조합 |

### 13-3. Phase 2 공통 실행 명령
```bash
python3 -m model.autoEQ.train_liris.run_train \
  --feature_dir data/features/liris_panns \
  --num_mood_classes 4 \
  --lambda_mood 0.3 --lambda_gate_entropy 0.05 \
  --ccc_hybrid_w 0.3 \
  --mixup_prob 0.5 --target_shrinkage_eps 0.05 \
  --epochs 40 --use_wandb \
  --output_dir runs/phase2a_0_baseline
```

---

## 14. 평가·모니터링 구현 (V3 신규 명시)

### 14-1. Film-level cluster bootstrap (§2-3 교정)

**문제**: sample-level resampling은 한 film 내 window를 독립으로 가정 → CI 과소추정.

**해결**: film 단위 resampling.
```python
def film_level_bootstrap_ci(
    preds: np.ndarray,           # (N,)
    targets: np.ndarray,         # (N,)
    film_ids: np.ndarray,        # (N,) — 각 sample의 film
    metric_fn: callable,         # ccc, pearson, mae
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    unique_films = np.unique(film_ids)
    n_films = len(unique_films)

    # film → window index 매핑 (사전 계산)
    film_to_indices = {f: np.where(film_ids == f)[0] for f in unique_films}

    results = []
    for _ in range(n_resamples):
        sampled_films = rng.choice(unique_films, size=n_films, replace=True)
        indices = np.concatenate([film_to_indices[f] for f in sampled_films])
        results.append(metric_fn(preds[indices], targets[indices]))

    results = np.array(results)
    lo = np.percentile(results, (1 - ci) / 2 * 100)
    hi = np.percentile(results, (1 + ci) / 2 * 100)
    return {
        'point_estimate': metric_fn(preds, targets),
        'ci_lower': lo,
        'ci_upper': hi,
        'ci_width': hi - lo,
    }
```
**보고 형식**:
```
LIRIS test: mean_CCC = 0.550 [95% CI: 0.508, 0.591]  (film-level n=80)
COGNIMUSE OOD: mean_CCC = 0.280 [95% CI: 0.219, 0.342] (film-level n=7)
```

### 14-2. Overfit auto-monitor (§2-5)

```python
# trainer.py 매 epoch 종료 시
train_mean_ccc = compute_ccc_on_train_subset()  # train 10% 샘플링
val_mean_ccc   = compute_ccc_on_val()

gap = train_mean_ccc - val_mean_ccc
wandb.log({'diagnostic/train_val_ccc_gap': gap})

if gap > 0.10:
    log.warning(
        f"[Overfit suspected @ epoch {epoch}] "
        f"train_CCC={train_mean_ccc:.3f} val_CCC={val_mean_ccc:.3f} gap={gap:.3f}"
    )
    # 자동 대응 (config.overfit_response에 따라):
    # 'alert_only' (default): 로그만
    # 'reduce_lr':  optimizer.param_groups[0]['lr'] *= 0.5
    # 'stop_early': patience = 0, trigger stop
```

### 14-3. Loss term 독립 로깅 (§2-6)

```python
# trainer.py 매 step 종료 시
losses = {
    'loss_total': L_total.item(),
    'loss_va':    L_va.item(),
    'loss_mood':  L_mood.item(),
    'loss_gate_entropy': L_ent.item(),
}
wandb.log({f'train/{k}': v for k, v in losses.items()}, step=global_step)

# validation 종료 시 val/loss_* 동일 로깅
```
JSON history에도 동일 기록. wandb 대시보드에서 4개 독립 차트로 발산/수렴 확인.

### 14-4. Gate 진단 로깅

```python
# 매 validation
wandb.log({
    'gate/w_v_mean': gate_weights[:, 0].mean(),
    'gate/w_a_mean': gate_weights[:, 1].mean(),
    'gate/w_v_var':  gate_weights[:, 0].var(),
    'gate/w_a_var':  gate_weights[:, 1].var(),
    'gate/entropy':  compute_gate_entropy(gate_weights),  # 평균
})
# 경고 조건: mean(w_v) > 0.9 or < 0.1 → modality collapse
```

### 14-5. Head-wise grad norm

```python
for head_name, head in [('va', va_head), ('mood', mood_head), ('gate', gate_net)]:
    norm = sum(p.grad.norm(2).item()**2 for p in head.parameters() if p.grad is not None) ** 0.5
    wandb.log({f'grad_norm/{head_name}': norm})
# V3.2 §4-1-1: 모든 head가 10배 이내 차이 → OK
```

---

## 15. 데이터 증강 4종 상세 (V3 업데이트 — §3-3 반영)

### 증강 1: Modality Dropout (model.forward 내)
```python
if self.training and self.modality_dropout_p > 0:
    trig = torch.rand(B, device=v.device) < self.modality_dropout_p  # p=0.05
    choice = torch.randint(0, 2, (B,), device=v.device)
    v[trig & (choice == 0)] = 0
    a[trig & (choice == 1)] = 0
```

### 증강 2: Feature Noise (model.forward 내)
```python
if self.training and self.feature_noise_std > 0:
    v_live = (v.abs().sum(-1, keepdim=True) > 0).float()
    a_live = (a.abs().sum(-1, keepdim=True) > 0).float()
    v = v + torch.randn_like(v) * self.feature_noise_std * v_live
    a = a + torch.randn_like(a) * self.feature_noise_std * a_live
```

### 증강 3: Quadrant Mixup (collate_fn)
- 같은 사분면 쌍만 섞음
- λ ~ Beta(0.4, 0.4) → 0.1 + 0.8·λ shrink
- visual, audio, v, a, v_var, a_var 모두 동일 λ 선형 결합
- mood label은 primary 유지

### 증강 4: Target Shrinkage (V3 재설계, §3-3)
```python
# V2 오류: threshold 0.15 고정 → LIRIS에서 50% 발동
# V3: per-axis p75, AND 조건 (default)
def apply_target_shrinkage(batch, eps, v_thr, a_thr, logic='AND'):
    if logic == 'AND':
        trigger = (batch.v_var > v_thr) & (batch.a_var > a_thr)  # ~6~10%
    else:  # 'OR'
        trigger = (batch.v_var > v_thr) | (batch.a_var > a_thr)  # ~40%
    batch.v[trigger] *= (1 - eps)  # 0.95
    batch.a[trigger] *= (1 - eps)
    return batch
```
Phase 1에서 `v_thr`, `a_thr` 확정값 저장 → trainer에서 로드.

### 증강 적용 순서
```
DataLoader → collate_fn [Mixup → Target Shrinkage] → model.forward [ModalityDropout → FeatureNoise]
→ Gate → Fusion → Heads → Loss
```

---

## 16. Verification (완료 확인)

### 16-1. 코드/데이터 무결성
- `pytest model/autoEQ/train_liris/tests/ -v` (9/9 PASS)
- `liris_preprocess --verify` → split 무결성, V/A round-trip, variance threshold 발동 비율 assert

### 16-2. 학습 sanity
- 2 epoch mini-run: L_total 감소 + 4개 loss term 독립 감소 + grad_norm 10배 이내 + train/val CCC gap 모니터링 정상

### 16-3. Primary metric 목표 (Bootstrap CI 포함)
**LIRIS 공식 split 기반**:
- val: mean_CCC ≥ 0.30 (baseline), ≥ 0.45 (stretch), 95% CI 보고
- test: mean_CCC ≥ 0.28 (OOD 허용), 95% CI 보고
- 축별 보고: ccc_v, ccc_a 각각 포함

**COGNIMUSE OOD**:
- mean_CCC ≥ 0.20, min(CCC_V, CCC_A) > 0
- film-level bootstrap CI 보고

### 16-4. End-to-end
- 샘플 1편: `analyze → timeline.json → playback → remuxed.mp4` 무오류
- VAD 대사 보호 적용 확인

---

## 17. 리스크 및 트레이드오프 (V3 업데이트)

| # | 리스크 | 대응 |
|---|---|---|
| 1 | **V/A Valence positive 편향으로 K=7 불가능** (§3-2) | **K=4 default**, K=7은 §8 게이트 상향 검토 / Phase 2a-1 전략 B ablation |
| 2 | **Variance threshold 0.15 부적합** (§3-3) | per-axis p75 threshold, AND 조건 default (6~10% 발동) |
| 3 | PANNs 4초 입력 품질 (§2-6) | Phase 0 sanity로 사전 차단 |
| 4 | Overfit 위험 (183만 파라미터 / 9,800 windows ≈ 187:1) | Overfit auto-monitor + patience 10 + 증강 4종 |
| 5 | Congruence 미복원의 학술 메시지 축소 | 보고서 "실증적 단순화 우위" 방어 |
| 6 | LIRIS 공식 split이 train:test = 1:2로 skew | 필요 시 `--use_full_learning_set`으로 set 1+2 합침 |
| 7 | 증강 효과 약화 (LIRIS ~9,800 vs CCMovies 722) | Phase 2b-5 증강 off-ablation |

---

## 18. 결론

**V3 핵심 교정 요약**:

1. **실측 기반**: LIRIS V/A 실제 범위 [1.33, 3.59] · [1.32, 4.54] 확인. V2의 "이론 1~5" 가정 파기.
2. **K=4 default**: Valence positive 편향으로 K=7 centroid 4개가 도달 불가 → K=4 quadrant를 기본값으로, K=7은 §8 게이트 통과 시 상향.
3. **Variance threshold per-axis p75**: 0.15 고정 폐기. v_thr≈0.117, a_thr≈0.164, AND 조건 default.
4. **Phase 2 OAT 분리**: 2a(근본 설계 순차) + 2b(하이퍼 튜닝). 각 단계 승자 확정 후 다음.
5. **Film-level bootstrap**: cluster resampling으로 CI 과소추정 회피.
6. **Overfit auto-monitor + per-term loss 로깅**: Multi-task 수렴 진단 강화.
7. **Phase 0 sanity 재설계**: "full clip vs 4s crop"으로 production 손실 직접 측정.
8. **K=7 게이트 fail-fast 순서**: 비용 최저(class count 집계) → Non-interference mini-run → Learnability.

**학위 논문 수준 준비 체크**:
- ✅ 용어 정확성: lexicographic, target shrinkage, AVEC-compliant CCC primary framing
- ✅ 통계적 근거: film-level bootstrap 95% CI
- ✅ 데이터 정합성: LIRIS 공식 split, V/A 실측 범위 반영, variance 분포 기반 threshold
- ✅ 실험 설계: OAT ablation, fail-fast K=7 게이트
- ✅ 학습 모니터링: overfit 자동 감지, per-term loss 로깅, head-wise grad norm, gate 진단

**진행 순서**: Phase 0 (30분~1시간) → Phase 1 (1주) → Phase 2a (1주) → Phase 2b (1주) → Phase 3 (3~5일).

---

## 부록 A. 실측 원시 데이터 (참고)

```
ACCEDEranking.txt 9800 rows, 8 cols
valenceValue  [1.33, 3.59]  mean=2.80  std=0.63
arousalValue  [1.32, 4.54]  mean=2.46  std=0.95
valenceVariance  [0.093, 0.164]  p50=0.108  p75=0.117
arousalVariance  [0.116, 0.218]  p50=0.150  p75=0.164

(v-3)/2 변환 후:
v_norm ∈ [-0.836, +0.297]  mean=-0.099
a_norm ∈ [-0.841, +0.771]  mean=-0.272

사분면 분포:
HVHA 1074 (11.0%) · HVLA 3944 (40.2%) · LVHA 1819 (18.6%) · LVLA 2963 (30.2%)
```
