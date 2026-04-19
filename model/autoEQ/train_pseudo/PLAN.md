# train_pseudo: CogniMuse 전용 학습 파이프라인 구현 계획

## Context

원 계획은 `model/autoEQ/train/`(LIRIS-ACCEDE 160편/9,800클립 기반)에서 학습, CogniMuse(7편/3.5h 연속 주석)를 OOD 검증에 쓰는 것이었음. LIRIS-ACCEDE 확보 불가로 CogniMuse만으로 학습·검증 모두 수행해야 하나, 기존 `train/`은 LIRIS 복구 대비 **그대로 보존**하고 `train_pseudo/`를 신규 생성해 CogniMuse 전용 파이프라인을 구축한다.

**근본 제약**:
- `train/negative_sampler.py`의 cross-film swap은 7편으로는 작동 불가 → **Congruence head + negative sampling 전면 제외**
- 40ms 연속 V/A → 4s 윈도우로 재집계 필요
- 7편으로는 전통적 film-level split 불가 → **LOMO 7-fold**

## 핵심 설계 결정 (15개, 누적)

1. **LOMO 교차검증** (Leave-One-Movie-Out, 7-fold)
2. **Modality dropout은 전체 샘플에 확률 p로 적용** (cong_label 의존 제거)
3. **experienced V/A 주석 사용** (피실험자 평균)
4. **4s 윈도우/2s stride + 윈도우 내 V/A 평균·표준편차 집계**
5. **Val은 time-based within-movie holdout** — 각 train 영화의 시간축 마지막 15% windows + gap 2 windows. Test는 1편 전체.
6. **Phase 0 분포 게이트** — mood class <1% 시 학습 차단
7. **Modality dropout p=0.05** (이전 실효 비율 5%와 등가)
8. **Movie ID 하드코딩**: `COGNIMUSE_MOVIES = ["BMI","CHI","CRA","DEP","FNE","GLA","LOR"]`
9. **Within-window σ metadata 필수 필드** (`valence_std`, `arousal_std`)
10. **명세서 V3.3 신규 작성** (`docs/specification_v3_3.md` + CHANGELOG)
11. **Early stopping**: patience 5, max_epochs 30
12. **재사용 전략: Import 공유** — train_pseudo는 train/의 frozen 자산만 import (`encoders.py`, `utils.py` 일부, `precompute.py`의 tensor 함수, `dataset.py`의 `MOOD_CENTERS`/`va_to_mood`)
13. **Config: 독립 `TrainCogConfig` dataclass** (TrainConfig 상속 없음, cong 필드 애초 없음)
14. **Class naming: Cog suffix** (`AutoEQModelCog`, `TrainerCog`, `PrecomputedCogDataset`)
15. **Plan 문서 위치: `train_pseudo/PLAN.md`**. 기존 `train/COGNIMUSE_TRANSITION_PLAN.md`는 historical reference로 사용자가 수동 정리

## 평가 지표 체계 (CCC + MAE + RMSE 트리오)

| 지표 | 역할 | 계산식 |
|---|---|---|
| **CCC** | Primary, early stopping 주 기준 | 기존 `compute_ccc` |
| **MAE** | 합격 게이트 + early stopping tiebreaker | `mean(|pred - target|)` |
| **RMSE** | 학술 보고 (AVEC 컨벤션) | `sqrt(mean((pred - target)²))` |
| `mean_mae` | tiebreaker·게이트 | `0.5 * (mae_v + mae_a)` (산술 평균) |
| `mean_rmse` | stretch 보고 | `0.5 * (rmse_v + rmse_a)` (산술 평균, quadratic 아님) |

**Best model 선정**: 튜플 비교 `(mean_ccc, -mean_mae) > (best_ccc, -best_mae)`로 CCC 주·MAE tiebreaker.

**지표 축 위계**: 합격 판정 = V/A Primary AND V/A Safety. Mood는 informative metric (판정 미포함, Safety 미달 시 보고서에 원인 분석).

**합격 기준 (3축 × 3단계)**:

| 단계 | V/A (pass/fail) | Mood (informative) |
|---|---|---|
| Primary | `mean_ccc` ≥ 0.20 **AND** `mean_mae` ≤ 0.25 | `f1_macro` ≥ 1.75/K **AND** `kappa` ≥ 0.15 |
| Safety | ≥ 6/7 folds: `min(ccc_v, ccc_a) > 0` **AND** `max(mae_v, mae_a) ≤ 0.30` | ≥ 6/7 folds: `f1_macro` ≥ 1.4/K |
| Stretch | `mean_ccc` ≥ 0.30 **AND** `mean_mae` ≤ 0.20 **AND** `mean_rmse` ≤ 0.28 | `f1_macro` ≥ 2.45/K **AND** `kappa` ≥ 0.25 |

K=7 → f1 {0.25, 0.20, 0.35}, K=4 → f1 {0.44, 0.35, 0.61}. "6/7 pass" 시 실패 fold는 원인 분석 1문단 필수.

## 폴더 구조

```
model/autoEQ/train_pseudo/
├── __init__.py
├── config.py                          # TrainCogConfig (독립 dataclass)
├── model.py                           # AutoEQModelCog (+ AudioProjectionCog, GateNetworkCog, VAHeadCog, MoodHeadCog)
├── losses.py                          # combined_loss_cog (3-term)
├── dataset.py                         # PrecomputedCogDataset, lomo_splits_with_time_val, apply_sigma_filter, create_dataloaders_cog, MOOD_CENTERS_4Q
├── trainer.py                         # TrainerCog (negative_sampler 의존 없음, CCC+MAE tiebreaker)
├── cognimuse_preprocess.py            # 원본 I/O 독립 구현 (annotation 로드, MP4 프레임/오디오 추출, 윈도우 aggregation)
├── analyze_cognimuse_distribution.py  # Phase 0 게이트
├── run_train.py                       # 단일 fold 학습 CLI
├── run_lomo.py                        # 7-fold orchestrator
├── PLAN.md                            # 이 문서
└── tests/
    ├── __init__.py
    ├── test_lomo_split.py
    ├── test_cognimuse_window_aggregation.py
    ├── test_movie_id_determinism.py
    ├── test_distribution_gate.py
    ├── test_cog_model_shapes.py
    ├── test_cog_losses.py
    ├── test_cog_modality_dropout.py
    └── test_cog_pipeline.py
```

기존 `train/tests/`는 **무수정**.

## 재사용 자산 (train/에서 import)

| 모듈 | import 경로 | 비고 |
|---|---|---|
| `XCLIPEncoder`, `PANNsEncoder` | `from ..train.encoders import ...` | frozen, 그대로 |
| `split_into_windows`, `resample_for_panns`, `encode_window_batch`, `save_features` | `from ..train.precompute import ...` | tensor-level 함수만 재사용 (I/O stub은 재사용 안 함) |
| `compute_ccc`, `compute_mean_ccc`, `compute_mood_metrics`, `compute_head_grad_norms` | `from ..train.utils import ...` | 그대로 |
| `compute_va_regression_metrics` | 〃 | **RMSE 필드 추가 필요 — train/utils.py "무수정" 원칙의 유일한 예외** |
| `MOOD_CENTERS`, `va_to_mood` | `from ..train.dataset import ...` | K=7 기본. K=4 축소 시 train_pseudo 내부 `MOOD_CENTERS_4Q` 정의 |

**RMSE 추가만은 train/utils.py 수정이 불가피** (단일 반환 dict이라 별도 함수 만들면 중복). 추가는 기존 테스트 호환(반환 dict에 키만 늘어남)이라 안전. 명세서 CHANGELOG에 "train/utils.py `compute_va_regression_metrics`에 `rmse_valence`, `rmse_arousal` 필드 추가" 명시.

## CogniMuse 원본 구조 가정 (실제 데이터 수령 시 검증)

```
<cognimuse_dir>/
├── BMI/
│   ├── video.mp4               # 영상+오디오 embedded
│   ├── experienced/
│   │   ├── valence.txt         # 40ms 간격, 한 줄 = float
│   │   └── arousal.txt
│   └── intended/ (동일)
├── CHI/, CRA/, DEP/, FNE/, GLA/, LOR/  (동일 구조)
```

- 주석 sampling rate: 25 Hz (40ms) — CogniMuse 문헌 기준
- 주석 범위: `[-1, 1]` 가정 (실제 데이터 수령 시 첫 확인 지점)
- 비디오 FPS: 영화별 상이, torchvision.io 자동 인식
- 오디오: MP4 embedded, moviepy/ffmpeg로 추출

**구조가 다를 경우 수정 지점**:
- 단일 CSV 주석 → `load_va_annotation()` 수정
- 별도 WAV 파일 → `load_audio_from_mp4()` → `load_audio_from_wav()`
- Likert [1,7] 등 → `_normalize_va()` 재정의 (아래)

**V/A 정규화 공식**:
```python
def _normalize_va(raw, src_range=(-1.0, 1.0)):
    a, b = src_range
    if (a, b) == (-1.0, 1.0): return raw.astype(np.float32)
    normed = 2.0 * (raw - a) / (b - a) - 1.0
    oob_pct = float((np.abs(normed) > 1.0).mean())
    assert oob_pct < 0.001, f"V/A normalize: {oob_pct:.3%} out of [-1,1]"
    return normed.astype(np.float32)
```

## 파일별 작성 사양

### 1. `train_pseudo/config.py` — TrainCogConfig (독립)

필드 구성: 기존 TrainConfig 기반 minus cong 전부, plus CogniMuse/LOMO 필드.
- Feature dims: `visual_dim=512`, `audio_raw_dim=2048`, `audio_proj_dim=512`, `fused_dim=1024`
- Task heads: `num_mood_classes=7`, `gate_hidden_dim=256`, `head_hidden_dim=256` (**cong 필드 전부 없음**)
- Optimizer: `lr=1e-4`, `weight_decay=1e-5`, `batch_size=32`, `epochs=30`, `warmup_steps=500`, `grad_clip_norm=1.0`
- Loss: `lambda_va=1.0`, `lambda_mood=0.5`, `lambda_gate_entropy=0.05`, `use_ccc_loss=True`, `ccc_loss_weight=0.3`
- Modality dropout: `modality_dropout_p=0.05`
- Early stop: `early_stop_patience=5`
- Input specs: `num_frames=8`, `frame_size=224`, `audio_sr=16000`, `audio_sec=4`
- Encoder: `xclip_model="microsoft/xclip-base-patch32"`, `panns_checkpoint=""` (자동 다운로드)
- CogniMuse: `cognimuse_dir=""`, `cognimuse_annotation="experienced"`, `cognimuse_window_sec=4`, `cognimuse_stride_sec=2`
- LOMO: `lomo_fold=-1`, `val_tail_ratio=0.15`, `val_gap_windows=2`
- σ-filter: `sigma_filter_threshold=-1.0` (음수=비활성)
- Wandb: `use_wandb=False`, `wandb_project="moodeq_cog"`, `wandb_run_name=""`
- Property: `audio_samples` = `audio_sr * audio_sec`

### 2. `train_pseudo/model.py` — AutoEQModelCog

기존 `train/model.py` 대비 차이:
- `CongruenceHead` 없음
- `self.cong_head` 초기화 없음
- `forward` 시그니처: `forward(self, visual_feat, audio_feat)` (cong_label 인자 삭제)
- `_apply_modality_dropout`: cong_label 무관, 전체 배치에 `modality_dropout_p` 확률 적용
- 반환 dict: `{"va_pred", "mood_logits", "gate_weights"}` (cong_logits 없음)
- Cong noise 로직 없음

하위 클래스 이름: `AudioProjectionCog`, `GateNetworkCog`, `VAHeadCog`, `MoodHeadCog`.

### 3. `train_pseudo/losses.py`

- `va_mse_loss`, `va_hybrid_loss`, `mood_ce_loss`, `gate_entropy_loss`: train/losses.py와 동일 로직 복제 (import 순환 방지)
- `cong_ce_loss` 없음
- `combined_loss_cog(outputs, va_target, mood_target, config)`: cong_target 인자 없음, total = λ_va·L_va + λ_mood·L_mood + λ_gate·L_gate
- `loss_dict`: `{"va", "mood", "gate_entropy", "total"}` (+ `va_mse`, `va_ccc` if CCC hybrid)

### 4. `train_pseudo/dataset.py`

**상수**:
```python
COGNIMUSE_MOVIES = ["BMI", "CHI", "CRA", "DEP", "FNE", "GLA", "LOR"]  # 영구 고정

MOOD_CENTERS_4Q = torch.tensor([  # K=4 축소 시 사용
    [ 0.6,  0.6],  # HVHA (Joy/Power)
    [ 0.6, -0.4],  # HVLA (Peace/Tenderness)
    [-0.6,  0.6],  # LVHA (Tension)
    [-0.6, -0.4],  # LVLA (Sadness)
], dtype=torch.float32)
```

**클래스**:
- `PrecomputedCogDataset`: `{split}_visual.pt`, `{split}_audio.pt`, `{split}_metadata.pt` 로드. `__getitem__` 반환: `{visual_feat, audio_feat, valence, arousal, mood, movie_id, valence_std, arousal_std}` (cong_label 없음)

**함수**:
- `lomo_splits_with_time_val(metadata, fold, val_tail_ratio=0.15, gap_windows=2) -> (train_ids, val_ids, test_ids)`:
  - `test_movie = COGNIMUSE_MOVIES[fold]`, test = 해당 영화 전체 windows
  - 나머지 6편 각각: `t0` 오름차순 정렬 → `N_val = round(N * val_tail_ratio)`, `N_train = N - N_val - gap_windows`, `N_train > 0` assert
  - `train = windows[:N_train]`, `val = windows[N_train + gap_windows:]` (gap_windows 개 drop)
  - Rounding: `round()` (banker's). 주의: Python `round(4.5) == 4`
  - **Gap-induced time buffer**: `diff_sec = stride * (gap + 1) - window`.
    - `window=4, stride=2, gap=0` → diff=−2s (2초 overlap 있음)
    - `gap=1` → diff=0s (인접, overlap 없음)
    - `gap=2` → diff=+2s (2초 버퍼 — overlap 제거 + scene 연속성 완화)
    - 즉 **gap=2는 "4s 완전 분리"가 아니라 "overlap 제거 후 2s 추가 버퍼"**. 이 계산이 원 plan 문서에서 "4s 분리"로 잘못 기술됐음 — 실제는 2s 버퍼이나 window-overlap leakage 차단이라는 핵심 목적은 달성.
- `apply_sigma_filter(window_ids, metadata, threshold) -> list[str]`:
  - `threshold <= 0`이면 no-op
  - `max(valence_std, arousal_std) > threshold`인 window 제외
  - **train split에만 호출 권장** (val/test는 원본 분포 유지해야 평가 타당성 확보)
- `create_dataloaders_cog(dataset, train_ids, val_ids, config) -> (train_loader, val_loader)`: window_id 리스트 기반 Subset + DataLoader

### 5. `train_pseudo/trainer.py` — TrainerCog

기존 Trainer와 차이:
- `negative_sampler` import·초기화·호출 없음
- `train_one_epoch`: forward는 `self.model(visual, audio)` (cong_label 없음). total_losses에 `"cong"` 키 없음. heads grad norm은 `{"va", "mood"}`만.
- `validate`: batch에서 `cong_label` 로드 안 함. all_cong_* 없음. `total_losses`에 `"cong"` 없음. RMSE 포함, `mean_rmse = 0.5 * (rmse_v + rmse_a)` 저장.
- `check_early_stopping`: `self.best_mean_mae=inf` 상태 추가. `(mean_ccc, -mean_mae) > (best_ccc, -best_mae)` 튜플 비교로 갱신.
- `_log_to_wandb`: scalar_keys에서 cong 관련 전부 제거, `"rmse_valence"`, `"rmse_arousal"`, `"mean_rmse"` 추가.

### 6. `train_pseudo/cognimuse_preprocess.py` — 독립 I/O

**함수**:
- `_normalize_va(raw, src_range=(-1.0, 1.0))`: V/A 선형 변환 + oob assert
- `load_va_annotation(movie_dir, source)`: `source ∈ {"experienced", "intended", "mean"}`. experienced/intended는 valence.txt/arousal.txt 로드. mean은 둘의 산술 평균. Returns `(v_arr, a_arr, sr_hz=25.0)`
- `aggregate_window_va(v_arr, a_arr, sr_hz, t0, t1) -> (mean_v, mean_a, std_v, std_a)`: `int(t0*sr)`부터 `int(t1*sr)`까지 slice 후 통계.
- `load_frames_from_mp4(video_path, timestamps) -> Tensor`: torchvision.io.read_video 또는 decord로 timestamp에 가장 가까운 frame 선택. ImageNet normalize 적용. Returns `(num_frames=8, 3, 224, 224)`
- `load_audio_from_mp4(video_path, t0, t1, target_sr=16000) -> Tensor`: moviepy 또는 ffmpeg로 구간 추출, 16kHz mono 리샘플. Returns `(T,)`
- `compute_manifest_sha(cognimuse_dir) -> dict[code, sha256]`: 7편 MP4 SHA256 해시
- `preprocess_cognimuse(cognimuse_dir, output_dir, annotation_source, window_sec=4, stride_sec=2, xclip, panns, batch_size=16)`:
  1. 각 영화별 `load_va_annotation` + duration 획득
  2. `split_into_windows` (train/precompute.py 재사용)
  3. 윈도우별 프레임 timestamp: `[t0 + 0.25 + 0.5*i for i in range(8)]`
  4. 프레임·오디오 로드 → **배치 단위로** `encode_window_batch` 호출 (성능)
  5. `window_id = f"{movie_code}_{idx:05d}"`
  6. metadata 스키마: `{movie_id, movie_code, valence, arousal, valence_std, arousal_std, t0, t1, annotation_source}`
  7. `save_features` (train/precompute.py 재사용) + `cognimuse_preprocess_manifest.json` 별도 저장

**Manifest 스키마** (`cognimuse_preprocess_manifest.json`):
```json
{
  "annotation_source": "experienced",
  "window_sec": 4, "stride_sec": 2,
  "movies": {"BMI": {"window_count": N, "duration_sec": D}, ...},
  "file_sha": {"BMI": "abc123...", ...},
  "cognimuse_movies_constant": ["BMI", ...]
}
```

### 7. `train_pseudo/analyze_cognimuse_distribution.py` — Phase 0 게이트

`GATE_THRESHOLD = 0.01` (1%). 산출물:
- `runs/phase0/distribution_report.json`:
```json
{
  "n_windows": N, "movie_counts": {...},
  "va_quadrant_pct": {"HVHA": ..., "HVLA": ..., "LVHA": ..., "LVLA": ...},
  "mood_class_pct": {"0": 0.12, "1": 0.08, ...},
  "sigma_valence": {"p50": ..., "p90": ..., "p99": ...},
  "sigma_arousal": {"p50": ..., "p90": ..., "p99": ...},
  "movie_va_mean": {"BMI": [mean_v, mean_a], ...},
  "gate_passed": true|false,
  "min_mood_class_pct": ...,
  "K": 7
}
```
- `runs/phase0/distribution_plots/*.png` (V/A 2D 산점도, σ 히스토그램, mood class bar)

**게이트 로직**: 어떤 mood class 비율 < 1% → `sys.exit(1)` + 대응 가이드 출력 (lambda_mood 하향 / 4-quadrant 축소 / mood head 제거).

### 8. `train_pseudo/run_train.py` — 단일 fold 학습

CLI:
```
python -m model.autoEQ.train_pseudo.run_train \
    --feature_dir data/features/cognimuse \
    --split_name cognimuse \
    --lomo_fold 0 \
    --epochs 30 \
    [--modality_dropout_p 0.05] \
    [--sigma_filter_threshold 0.3] \
    [--val_tail_ratio 0.15] \
    [--val_gap_windows 2] \
    [--seed 42] \
    [--output_dir runs/latest] \
    [--use_wandb]
```

**저장물** (`--output_dir`):
- `best_model.pt`
- `history.json`
- `fold_mapping.json`:
```json
{
  "run_id": "...", "timestamp": "...",
  "fold": 0, "test_movie_code": "BMI", "test_movie_id": 0,
  "train_movies": ["CHI","CRA","DEP","FNE","GLA","LOR"],
  "val_tail_ratio": 0.15, "val_gap_windows": 2,
  "dropped_window_ids": [...],
  "window_counts": {"train": N, "val": M, "test": L},
  "sigma_filter_threshold": -1.0, "sigma_filtered_out_count": 0,
  "modality_dropout_p": 0.05, "seed": 42,
  "preprocess_manifest_sha": {...},
  "cognimuse_movies_constant": [...]
}
```

### 9. `train_pseudo/run_lomo.py` — 7-fold orchestrator

**동작 규칙**:
- 순차 실행 (GPU 메모리 제약으로 병렬 생략)
- 각 fold 독립 seed: `base_seed + fold_idx`
- 중간 fold 실패 시 `logs/fold_<k>_error.log` 기록 후 **나머지 fold 계속 진행**
- 최종 산출물:
  - `runs/<run_id>/lomo_report.json`: fold별 best metric 배열 + 평균/표준편차 + Pass/Fail 판정
  - `runs/<run_id>/lomo_report.md`: 학술 보고용 표 (CCC/MAE/RMSE/F1/κ × 7 folds + 평균±std)

### 10. 테스트 사양

**`test_lomo_split.py`**:
- `test_test_is_full_movie()`: fold k의 test_ids가 `COGNIMUSE_MOVIES[k]` 전체 windows인지
- `test_train_val_disjoint()`: train_ids ∩ val_ids = ∅
- `test_gap_windows_dropped()`: train 끝 window의 t_end와 val 첫 window의 t_start 간격이 gap_windows * stride_sec 이상
- `test_assertion_on_short_movie()`: N - N_val - gap_windows <= 0이면 AssertionError

**`test_cognimuse_window_aggregation.py`**: 
- 결정적 V/A 배열(예: sine wave) + 고정 window로 aggregate_window_va 결과를 numpy 계산과 대조
- 빈 슬라이스 (t0 > duration) edge case

**`test_movie_id_determinism.py`**:
- `COGNIMUSE_MOVIES` 알파벳 정렬 상태 확인
- `COGNIMUSE_MOVIES.index("BMI") == 0` 등 6개 체크
- 동일 preprocess 2회 실행 시 manifest file_sha 동일

**`test_distribution_gate.py`**:
- Mock metadata에서 특정 mood class <1% 강제 → exit code 1
- 모든 class >= 1% → exit code 0

**`test_cog_model_shapes.py`**:
- Forward 반환 dict 키 = `{"va_pred", "mood_logits", "gate_weights"}` (cong_logits 부재 확인)
- 각 tensor shape: `(B, 2)`, `(B, K)`, `(B, 2)`

**`test_cog_losses.py`**:
- `combined_loss_cog` loss_dict 키 = `{"va", "mood", "gate_entropy", "total"}` (+ va_mse, va_ccc if CCC)
- `"cong"` 키 부재
- 가중치 합 검증: `total ≈ λ_va·va + λ_mood·mood + λ_gate·gate_entropy` (부호 포함, gate는 음수)

**`test_cog_modality_dropout.py`**:
- training=True + fixed seed: 약 `modality_dropout_p` 비율의 샘플이 visual 또는 audio에 0 벡터
- training=False: 입출력 identity
- cong_label 인자 받지 않음 (TypeError)

**`test_cog_pipeline.py`**: 합성 metadata → `lomo_splits_with_time_val(fold=0)` → 2 epoch 학습 → history len = 2, best_state_dict 존재

### 11. X-CLIP visual_dim 사전 검증

첫 preprocess 실행 전 1-batch smoke test로 `xclip(dummy_frames).shape[-1] == 512` 확인. 불일치 시 `TrainCogConfig.visual_dim` 조정 + model head 입력 차원 재검증.

## 작업 순서

| 순 | 대상 | 비고 |
|---|---|---|
| 1 | `train_pseudo/__init__.py`, `config.py` | 최소 seed |
| 2 | `train_pseudo/model.py` | cong 없는 구조 |
| 3 | `train_pseudo/losses.py` | 3-term |
| 4 | `train/utils.py`에 RMSE 필드 추가 | 기존 무수정 원칙의 유일한 예외 |
| 5 | `train_pseudo/dataset.py` | LOMO 스플리터 + apply_sigma_filter |
| 6 | `train_pseudo/trainer.py` | CCC+MAE tiebreaker |
| 7 | `train_pseudo/cognimuse_preprocess.py` | I/O 독립 구현 |
| **8** | **CogniMuse 원본 파일 수령 → 구조 assumption 검증** | **여기서 실제 파일 구조 확인하고 plan assumption 블록 수정** |
| 9 | `train_pseudo/analyze_cognimuse_distribution.py` + **Phase 0 실제 실행 (게이트)** | K=7/4 확정, `lambda_mood` 확정, `MOOD_CENTERS_4Q` 사용 여부 확정 |
| 10 | `train_pseudo/run_train.py`, `run_lomo.py` (Phase 0 결과 반영 default) | fold_mapping.json 저장 |
| 11 | `train_pseudo/tests/` 일괄 작성 | 8개 파일 |
| 12 | 7-fold LOMO 전체 실행 | `runs/<run_id>/lomo_report.{json,md}` |
| 13 | `docs/specification_v3_3.md` + CHANGELOG 작성 | Phase 0 리포트 수치 인용 |

**Phase 0 결과에 따른 Revert 경계**:
- K=7 유지 → 이후 수정 없음
- `lambda_mood` 하향 → `train_pseudo/config.py` 1곳
- K=4 축소 → `train_pseudo/config.py` `num_mood_classes=4` + `train_pseudo/dataset.py`에서 `MOOD_CENTERS_4Q` 활성 + `test_cog_model_shapes.py` K 수정

## 학습 프로세스 다이어그램

```
CogniMuse 원본(7편 MP4 + 40ms V/A .txt, experienced)
    ↓ train_pseudo/cognimuse_preprocess.py (독립 I/O)
        · load_va_annotation → _normalize_va → aggregate_window_va
        · load_frames_from_mp4 + load_audio_from_mp4
        · encode_window_batch (train/precompute.py의 tensor 함수 재사용)
        · save_features + manifest.json
    ↓
data/features/cognimuse/{cognimuse_visual.pt, _audio.pt, _metadata.pt}
    ↓ train_pseudo/analyze_cognimuse_distribution.py (Phase 0 게이트)
    ↓
PrecomputedCogDataset
    ↓ lomo_splits_with_time_val(fold=k, gap_windows=2)
    ↓ apply_sigma_filter (train only)
    ↓
AutoEQModelCog (cong 없음, dropout 전역)
    ↓
combined_loss_cog: L = 1.0·L_va + 0.5·L_mood + 0.05·L_gate
    ↓
TrainerCog (early stopping: CCC 주 + MAE tiebreaker)
    ↓
fold별 best_model.pt + fold_mapping.json
    ↓ run_lomo.py × 7
    ↓
runs/<run_id>/lomo_report.{json,md}
```

## 명세서 V3.3 개정

**산출물** (`docs/` 하위):
1. `specification_v3_3.md`: 본문 개정
2. `spec_v3_2_to_v3_3_changelog.md`: 섹션별 diff
3. `cognimuse_distribution_report.md`: Phase 0 결과 표/그래프

**섹션별 개정**:

| 섹션 | V3.3 |
|---|---|
| 2-1 학습 데이터 | CogniMuse 7편, 4s/2s 윈도우 약 N windows (Phase 0 확정 후 기입) |
| 2-2 검증 | LOMO 7-fold (test=1편), time-based within-movie val + gap 2 |
| 2-3 Split | LOMO + 하드코딩 매핑 표 |
| 2-4 윈도우 | 40ms experienced V/A → 4s/2s 평균+σ |
| 3-3 Negative Sampling | **삭제**, 데이터셋 제약 근거 문단 대체 |
| 5 Multi-task | V/A(주) + Mood(보조) 2-task |
| 6 손실 가중치 | `λ=(va 1.0, mood 0.5, gate 0.05)` |
| 7-1 하이퍼파라미터 | epochs 30, patience 5, dropout 0.05 |
| 7-2 정량 평가 | LOMO 7-fold 표: CCC/MAE/RMSE + F1/κ. CCC primary / MAE gate / RMSE 보고 |
| 8-1 한계 | + 7편 소규모 + 연속→윈도우 변환 왜곡 + 주석자 7명 편향 |
| Phase 체크리스트 | Phase 0 신규, Phase 2에서 negative sampler 항목 삭제 |

**학술 방어 문구** (8-1 말미):
> 원 계획은 LIRIS-ACCEDE 학습 + CogniMuse OOD 이원화였다. 데이터 확보 제약으로 학습·검증 모두 CogniMuse로 수행하였으며, (a) cross-film 자기지도 Congruence 학습 제외, (b) LOMO 7-fold 평가로 변경하였다. 데이터 규모 축소 대신 film-level 일반화를 엄격히 측정하는 이점이 있으나, 감정 분포 편향이 7편 수준 분산에 종속되는 한계를 가진다.

## 검증 방법 (End-to-End)

### 0. Phase 0 게이트
```bash
python -m model.autoEQ.train_pseudo.cognimuse_preprocess \
    --cognimuse_dir /path/to/cognimuse \
    --output_dir data/features/cognimuse \
    --annotation experienced

python -m model.autoEQ.train_pseudo.analyze_cognimuse_distribution \
    --feature_dir data/features/cognimuse \
    --split_name cognimuse \
    --output_dir runs/phase0
```
기대: exit 0, distribution_report.json 생성. exit 1이면 lambda_mood/K 조정 후 재실행.

### 1. 단위 테스트
```bash
pytest model/autoEQ/train_pseudo/tests -x
```

### 2. 단일 fold 스모크
```bash
python -m model.autoEQ.train_pseudo.run_train \
    --feature_dir data/features/cognimuse \
    --lomo_fold 0 --epochs 30
```
기대: 학습 수렴, best_model.pt + fold_mapping.json 저장.

### 3. 7-fold LOMO
```bash
python -m model.autoEQ.train_pseudo.run_lomo \
    --feature_dir data/features/cognimuse --epochs 30
```
기대: lomo_report.json의 평균 지표가 합격 기준 테이블 Primary 충족.

### 4. Gate 건전성 + 합격 판정
- fold별 `gate_w_v`, `gate_w_a` ∈ [0.3, 0.7]
- V/A Primary: mean_ccc ≥ 0.20 AND mean_mae ≤ 0.25
- V/A Safety: 6/7 folds에서 min(ccc_v,ccc_a) > 0 AND max(mae_v,mae_a) ≤ 0.30
- Mood는 informative (판정 미포함), Safety 미달 시 원인 분석

## 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| 7편 학습 과적합 | LOMO + early stop(patience 5) + modality dropout + gate_entropy |
| 연속→윈도우 평균 왜곡 | σ metadata 필수 + Phase 0 σ 분포 시각화 + `sigma_filter_threshold` 옵션 (train only) |
| Val 1편 불안정 | Time-based within-movie val holdout (gap 2) + test CCC만 최종 권위 |
| Mood class degeneration | Phase 0 게이트 + K-독립 임계 (`1.75/K` 등) + K=4 축소 대응 준비 |
| CogniMuse 원본 구조 불일치 | Plan에 assumption 블록 명시 + 파일 수령 시 검증 단계(작업 순서 8번)에서 assumption만 수정 |
| V/A 정규화 오인 | `_normalize_va` 내 oob pct < 0.1% assertion |
| Gate 신호 약화 | gate_entropy 유지 + modality dropout 전역 적용 |
| Movie ID 재현 불일치 | COGNIMUSE_MOVIES 하드코딩 + manifest file_sha + fold_mapping.json |
| X-CLIP visual_dim 불일치 | 첫 preprocess 전 1-batch smoke |
| train/utils.py RMSE 추가 영향 | 필드 추가만 (기존 키 무변경) — train/ 테스트 통과 유지 |
| 코드-명세서 drift | V3.3 개정을 작업 순서 13번에 포함, CHANGELOG 병행 |

## 기존 `train/` 처리 방침

- `train/` 폴더 **무수정** 원칙, 단 `train/utils.py`의 `compute_va_regression_metrics` 에 RMSE 필드 추가는 예외 (CHANGELOG 명시).
- `train/COGNIMUSE_TRANSITION_PLAN.md`: historical reference로 보존하거나 `archive/`로 사용자 수동 이동. 이 새 PLAN.md가 권위 문서.

## 학술 보고서 변경 요약

> "원 계획은 LIRIS-ACCEDE(~160편/9,800클립)를 학습, CogniMuse(7편)를 OOD 검증에 쓰는 것이었으나 LIRIS 미확보로 CogniMuse 단일 데이터셋 전환을 불가피하게 수행했다. (1) cross-film negative sampling 기반 Congruence head를 제거하고, (2) modality dropout을 전체 샘플에 확률 p=0.05로 적용하며, (3) LOMO 7-fold 교차검증으로 평가 방식을 변경했다. V/A 회귀(주태스크), Mood 분류(보조), Adaptive Gating + gate entropy는 V3.2 명세서 대비 변경 없이 유지된다. 평가 지표는 CCC(primary) + MAE(gate) + RMSE(report) 트리오를 채용하며, 합격 판정은 V/A Primary AND Safety로 정의하고 Mood는 informative로 보고한다."
