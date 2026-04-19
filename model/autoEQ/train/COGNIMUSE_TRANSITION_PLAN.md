# CogniMuse 데이터셋 전환 학습 파이프라인 개편 계획

## Context

현재 MoodEQ 학습 파이프라인은 **LIRIS-ACCEDE**(≈160편/9,800클립, 클립당 단일 V/A 라벨) 기준으로 설계되어 있음. 그러나 LIRIS-ACCEDE 다운로드가 불가해져, 원래 OOD 검증 세트로 쓰려던 **CogniMuse**(7편/~3.5시간, 40ms 해상도 연속 V/A 주석)를 학습에도 사용해야 함.

**근본 제약**:
- `negative_sampler.py`는 "다른 영화의 오디오와 교체"하는 cross-film swap 기반(라인 15, 102-108). 7편으로는 후보 풀이 너무 작아 스타일 leakage 위험 + congruence 신호의 self-supervised 학습이 의미를 잃음.
- CogniMuse의 감정 주석은 시계열(40ms/frame) → LIRIS식 "클립당 단일 라벨" 구조로 재집계 필요.
- Film-level split을 7편으로 나누면 test/val 크기가 너무 작고 분산이 큼.

**원칙**: 명세서 V3.2의 학습 구조(X-CLIP + PANNs + Gate + V/A 회귀 + Mood 분류)를 최대한 유지하고, 데이터셋 의존적 부분(negative sampling, congruence head, film-level disjoint 가정)만 수정.

**사용자 결정**:
1. **LOMO 교차검증** (Leave-One-Movie-Out, 7-fold)
2. **Modality dropout은 전체 샘플에 확률 p로 적용** (cong_label 의존 제거)
3. **experienced V/A 주석 사용** (피실험자 평균)
4. **4s 윈도우/2s stride + 윈도우 내 V/A 평균 집계 + σ 추적**

**피드백 반영 결정(2026-04-18 추가)**:
5. **Val은 time-based within-movie holdout** — 각 train 영화의 시간축 마지막 15% windows를 val pool로. Test(outer LOMO holdout)는 1편 전체 windows로 분리. Early stopping은 val CCC(trigger용), 최종 리포트는 test CCC만 권위.
6. **Phase 0 분포 분석 게이트** — `analyze_cognimuse_distribution.py` 완료 + mood class별 비율이 최소 1% 이상임을 확인한 뒤에만 학습 착수.
7. **Modality dropout p=0.05로 시작** — 이전 실효 비율(5%)과 동등화. {0.05, 0.075, 0.1} ablation CLI 옵션 유지.
8. **Movie ID 하드코딩** — `COGNIMUSE_MOVIES = ["BMI","CHI","CRA","DEP","FNE","GLA","LOR"]` 영구 고정. metadata에 `movie_code` 추가. fold mapping json 저장.
9. **Within-window σ를 metadata 필수 필드로** — `valence_std`, `arousal_std` 포함. 분석 스크립트에 σ 분포 출력. 학습 활용은 옵션(CLI `--sigma_filter_threshold`, 기본 비활성).
10. **명세서 V3.3 신규 작성** — `docs/specification_v3_3.md` + `docs/spec_v3_2_to_v3_3_changelog.md`. V3.2는 보존.
11. **Early stopping 안전장치** — `early_stop_patience: 10 → 5`, `max_epochs: 50 → 30`.

---

## 학습 프로세스(최종 형태)

```
CogniMuse 원본(MP4 + V/A 연속 .txt, experienced)
    ↓ [NEW] cognimuse_preprocess.py
        · 4s 윈도우 / 2s stride
        · 윈도우 시간구간 V/A 평균 → 클립식 라벨
        · V/A ∈ [-1, 1] 정규화 확인
        · movie_id = 7편 중 하나
    ↓
precompute.py (기존 로직 그대로)
    · X-CLIP (frozen) → 512-dim visual
    · PANNs CNN14 (frozen) → 2048-dim audio
    · {split}_visual.pt / {split}_audio.pt / {split}_metadata.pt
    ↓
PrecomputedFeatureDataset (cong_label 필드 제거)
    ↓
[NEW] LOMO 스케줄러: 7-fold 루프, fold k는 영화 k를 test로
    ↓
AutoEQModel(수정): cong_head 제거, modality dropout 무조건 확률 적용
    ↓
combined_loss(수정): L = λ_va·L_va + λ_mood·L_mood + λ_gate·L_gate
    ↓
Trainer(수정): negative_sampler 호출 제거, cong 관련 로깅/메트릭 제거
    ↓
fold별 best_mean_ccc 기록 → 7-fold 평균 리포트
```

---

## 변경 사항(파일별)

### 1. `model/autoEQ/train/config.py`
삭제할 필드:
- 라인 14 `num_cong_classes`
- 라인 17-18 `cong_head_input_dim`, `cong_head_hidden_dim`
- 라인 31 `lambda_cong`
- 라인 42-44 `neg_congruent_ratio`, `neg_slight_ratio`, `neg_strong_ratio`
- 라인 56 `cong_noise_std`

추가할 필드:
- `cognimuse_annotation: str = "experienced"` (experienced/intended/mean)
- `cognimuse_window_sec: int = 4` (이미 `audio_sec`와 동일하지만 명시성)
- `cognimuse_stride_sec: int = 2`
- `lomo_fold: int = -1` (-1=전체 학습, 0~6=fold 지정)
- `val_tail_ratio: float = 0.15` — train 영화 시간축 마지막 15%를 val pool로
- `val_gap_windows: int = 2` — train/val 경계의 시간축 leakage 차단용 drop 수 (window=4s, stride=2s → gap=2는 4s 완전 분리)
- `sigma_filter_threshold: float = -1.0` — 음수면 비활성, 양수면 σ 초과 windows 제외

수정할 필드:
- `modality_dropout_p: 0.1 → 0.05` (전역 적용으로 바뀌어 실효 비율 유지)
- `early_stop_patience: 10 → 5`
- `epochs: 50 → 30`

### 2. `model/autoEQ/train/model.py`
- **CongruenceHead 클래스 삭제** (라인 74-89)
- `AutoEQModel.__init__`에서 `self.cong_head` 제거 (라인 106)
- `forward`의 cong 계산 블록 삭제 (라인 185-196)
- 반환 dict에서 `"cong_logits"` 삭제 (라인 201)
- `forward` 시그니처에서 `cong_label: Tensor | None = None` 제거
- `_apply_modality_dropout` 수정: `cong_label` 인자 제거하고 모든 샘플에 확률 p로 적용
  ```python
  def _apply_modality_dropout(self, v, a):
      if not self.training:
          return v, a
      B = v.size(0)
      drop_choice = torch.randint(0, 2, (B,), device=v.device)
      drop_trigger = torch.rand(B, device=v.device) < self.config.modality_dropout_p
      drop_visual = drop_trigger & (drop_choice == 0)
      drop_audio = drop_trigger & (drop_choice == 1)
      v = v.clone(); a = a.clone()
      v[drop_visual] = 0.0; a[drop_audio] = 0.0
      return v, a
  ```

### 3. `model/autoEQ/train/losses.py`
- **`cong_ce_loss` 함수 삭제** (라인 39-44)
- `combined_loss` 시그니처에서 `cong_target` 제거
- 내부에서 `l_cong` 계산 및 total 합산 항 삭제
- `loss_dict`에서 `"cong"` 키 삭제

### 4. `model/autoEQ/train/trainer.py`
- **`from .negative_sampler import NegativeSampler` 삭제** (라인 13)
- `from .utils import` 에서 `compute_cong_accuracy` 삭제 (라인 15)
- `self.negative_sampler = NegativeSampler(config)` 삭제 (라인 68)

`train_one_epoch`:
- 라인 134-138 negative sampling 호출 블록 전체 삭제
- 라인 141 forward 호출을 `outputs = self.model(visual, audio)` 로 변경
- 라인 144-146 `combined_loss`에서 `cong_target` 제거
- 라인 158-162 gradient 측정 dict에서 `"cong"` 키 삭제
- `total_losses` / `grad_norms_accum` 초기화에서 `"cong"` 삭제 (라인 119-120)

`validate`:
- 라인 221 `cong_target` 로드 제거
- 라인 223 forward 호출을 `cong_label=None` 인자 제거로 변경
- 라인 225-227 combined_loss에 cong_target 전달 제거
- `all_cong_logits`, `all_cong_target` 관련 블록 전부 삭제 (라인 208-209, 241-242, 267-269)
- `total_losses`에서 `"cong"` 키 삭제

`_log_to_wandb`:
- `scalar_keys` 에서 `"cong"`, `"cong_accuracy"` 삭제 (라인 97, 104)
- `scalar_keys`에 `"rmse_valence"`, `"rmse_arousal"` 추가

`check_early_stopping` 수정:
- `self.best_mean_mae: float = float("inf")` 상태 추가
- 갱신 조건을 튜플 비교로:
  ```python
  mean_ccc = val_metrics.get("mean_ccc", 0.0)
  mean_mae = 0.5 * (val_metrics.get("mae_valence", 1.0) + val_metrics.get("mae_arousal", 1.0))
  if (mean_ccc, -mean_mae) > (self.best_mean_ccc, -self.best_mean_mae):
      self.best_mean_ccc = mean_ccc
      self.best_mean_mae = mean_mae
      self.patience_counter = 0
      self.best_state_dict = copy.deepcopy(self.model.state_dict())
      return False
  ```

### 5. `model/autoEQ/train/dataset.py`
- `SyntheticAutoEQDataset`: `self.cong_labels` 초기화 삭제(라인 196-197), `__getitem__` 반환에서 `"cong_label"` 삭제(라인 214)
- `PrecomputedFeatureDataset.__getitem__`: `"cong_label"` 필드 삭제(라인 285). V/A σ 필드 노출(`valence_std`, `arousal_std`) — 선택 사용
- **새 함수 추가**:
  ```python
  def lomo_splits_with_time_val(
      window_metadata: dict[str, dict],
      fold: int,
      val_tail_ratio: float = 0.15,
      gap_windows: int = 2,
  ) -> tuple[list[str], list[str], list[str]]:
      """Outer LOMO + time-based val holdout + temporal gap.

      fold k: test = COGNIMUSE_MOVIES[k]의 모든 window_ids (전체, gap 불필요).
      나머지 6편: 각 영화의 window들을 t0 오름차순 정렬 후
        - val = 뒤 val_tail_ratio 비율
        - train = 앞 (N - N_val - gap_windows)
        - 사이 gap_windows 개는 drop하여 시간축 leakage 차단
      Train/val은 window-level disjoint이며, window 간 overlap(stride=2s, window=4s → 2s 중첩)
      으로 인한 프레임/오디오 공유가 gap_windows=2일 때 4s 분리로 완전 제거됨.

      gap_windows=2 근거: stride=2s 기준 4s 시간 분리 = overlap 제거 + scene 연속성 완화.
      gap=1은 overlap만 제거(여유 없음), gap=3+는 train 데이터 과도 손실.

      Returns: (train_window_ids, val_window_ids, test_window_ids)
      Raises: AssertionError if any train movie has N - N_val - gap_windows <= 0.
      """
  ```
  - `gap_windows` 값과 drop된 window_ids는 `runs/<run_id>/fold_mapping.json`에 기록
  - σ-filter는 split 확정 후 각 split 내부에서 독립 적용 (gap 계산 전에 필터하지 않음)
- 기존 `film_level_split`, `stratified_film_level_split`는 유지(합성/향후 LIRIS용)
- `create_dataloaders` 시그니처를 window_id 리스트 기반으로 확장하거나 별도 `create_loaders_from_window_ids` 추가

### 6. `model/autoEQ/train/utils.py`
- **`compute_cong_accuracy` 함수 삭제** (라인 114-117)
- **`compute_va_regression_metrics` 확장**: 반환 dict에 `rmse_valence`, `rmse_arousal` 추가
  ```python
  err = va_pred - va_target  # (N, 2)
  rmse_v = float(torch.sqrt((err[:, 0] ** 2).mean()))
  rmse_a = float(torch.sqrt((err[:, 1] ** 2).mean()))
  ```

### 7. `model/autoEQ/train/negative_sampler.py`
- **파일 전체 삭제**

### 8. `model/autoEQ/train/run_train.py`
- import에서 `negative_sampler` 흔적 제거(필요 시)
- `build_splits` 수정: `--lomo` 플래그면 `lomo_splits_with_time_val(metadata, fold, val_tail_ratio)` 호출
- CLI 인자 추가:
  - `--dataset {cognimuse, liris, synthetic}`
  - `--lomo`, `--lomo_fold INT`
  - `--cognimuse_dir PATH`
  - `--val_tail_ratio 0.15`
  - `--modality_dropout_p 0.05`
  - `--sigma_filter_threshold -1.0`
- 매 run 시작 시 `runs/<run_id>/fold_mapping.json` 저장 (movie_code ↔ movie_id ↔ fold role)
- **새 entry**: `run_lomo.py` — 7번 반복 학습 후 fold별 **test mean_ccc** 평균/표준편차 리포트 (val CCC는 early stopping용이므로 리포트에서 제외)

### 9. **NEW: `model/autoEQ/train/cognimuse_preprocess.py`**
CogniMuse 원본 → precompute.py가 기대하는 `(frames, waveform, metadata)` 스트림 형태로 변환.
- **하드코딩 상수**:
  ```python
  COGNIMUSE_MOVIES = ["BMI", "CHI", "CRA", "DEP", "FNE", "GLA", "LOR"]  # 영구 고정
  # movie_id = COGNIMUSE_MOVIES.index(code)
  ```
- 입력: `--cognimuse_dir` (7편 MP4 + experienced V/A .txt 경로), `--annotation {experienced,intended,mean}`
- 파이프라인:
  1. 각 영화를 `window_sec=4`, `stride_sec=2`로 슬라이딩
  2. 윈도우 시간 [t0, t1] 구간의 40ms V/A 프레임들 **평균 및 표준편차** 계산 → `(mean_v, mean_a, std_v, std_a)`
  3. 비디오 8 프레임 균등 샘플, 오디오 16kHz/4s 추출
  4. `window_id = f"{movie_code}_{window_idx:05d}"` 규약 (예: `BMI_00042`)
  5. 기존 `precompute.py`의 `encode_window_batch` 호출
  6. `cognimuse_visual.pt`, `cognimuse_audio.pt`, `cognimuse_metadata.pt` 저장
     (metadata 스키마: `{movie_id: int, movie_code: str, valence: float, arousal: float, valence_std: float, arousal_std: float, t0: float, t1: float, annotation_source: str}`)
- 또한 `cognimuse_preprocess_manifest.json` 저장: 영화별 window 수, 전체 duration, 결정성 증명용 SHA of MP4 files

### 9-a. **NEW: `model/autoEQ/train/analyze_cognimuse_distribution.py`** (Phase 0 게이트)
`cognimuse_metadata.pt` 로드 후:
- V/A 2D 히스토그램 (quadrant별 %)
- 7 mood class별 샘플 수와 비율
- 영화별 V/A 평균/분산, quadrant 편향 표
- `valence_std`, `arousal_std` 분포 히스토그램, 상위 10% windows 예시 10개
- **게이트 판정**: 어떤 mood class가 <1% 이면 `sys.exit(1)`로 실패 신호 + 대응 가이드 출력
  - 가이드: (a) `lambda_mood` 하향, (b) 4-quadrant 분류로 축소, (c) mood head 제거 중 선택
- 산출물: `runs/phase0/distribution_report.json` + `distribution_plots/*.png`

### 10. 테스트 수정

**파일 전체 삭제**:
- `tests/test_negative_sampling.py`

**수정**:
- `test_losses.py`: cong_ce_* 테스트 삭제, `combined_loss` 호출부에서 `cong_target` 제거
- `test_model_shapes.py`: `test_cong_head_shape` 삭제, full forward shape 검증에서 `"cong_logits"` 제거
- `test_training_sanity.py`: `cong_target` 생성/전달 제거
- `test_gradient_norm.py`, `test_gradient_flow.py`: `cong_head` 항목 제거
- `test_label_consistency.py`: `cong_label` 범위 검증 삭제
- `test_modality_dropout.py`: cong_label 의존 테스트를 "모든 샘플 확률 적용" 버전으로 재작성 (입력 시그니처 변경)
- `test_pipeline_integration.py`: `"cong"` loss 키 검증 삭제
- `test_analyze_va.py`: `test_recommend_neg_sampler_thresholds` 삭제
- `analyze_va.py`: `recommend_negative_sampler_thresholds()` 함수는 남기되 CLI 기본 출력에서는 제외(학술용 데이터 분석 도구로 유지 가능)

**신규**:
- `tests/test_lomo_split.py`: 7-fold LOMO가 disjoint, test는 1편 전체 windows, val은 나머지 6편의 time-tail 15%이며 train과 window-level disjoint인지 검증
- `tests/test_cognimuse_window_aggregation.py`: 40ms 연속 V/A → 4s 윈도우 평균/σ 로직 단위 테스트 (결정적 입력 → 기대 출력)
- `tests/test_movie_id_determinism.py`: `COGNIMUSE_MOVIES` 상수가 정렬 상태이고, 같은 파일셋에 두 번 돌려도 movie_id 매핑이 동일한지 검증
- `tests/test_distribution_gate.py`: mood class가 <1% 비율일 때 `analyze_cognimuse_distribution.py`가 exit code 1을 반환하는지 smoke test

---

## 손실 함수 변경 후 형태

**이전**:
```
L = 1.0·L_va + 0.5·L_mood + 0.5·L_cong + 0.05·L_gate
```

**이후**:
```
L = 1.0·L_va + 0.5·L_mood + 0.05·L_gate
```

gate_entropy는 그대로 유지 → gate 균등화 regularization은 보존됨.
Modality dropout은 이제 cong_label 없이 작동하므로 gate가 단일 모달리티에 붕괴하는 것을 막는 2차 regularization 역할 수행.

---

## Critical Files (작업 순서)

1. `model/autoEQ/train/config.py` — 필드 정리
2. `model/autoEQ/train/model.py` — cong_head 제거, dropout 시그니처 수정
3. `model/autoEQ/train/losses.py` — `combined_loss` 시그니처 변경
4. `model/autoEQ/train/utils.py` — `compute_cong_accuracy` 삭제
5. `model/autoEQ/train/trainer.py` — negative_sampler/cong 로직 제거
6. `model/autoEQ/train/dataset.py` — `cong_label` 제거 + `lomo_splits_with_time_val` 추가
7. `model/autoEQ/train/negative_sampler.py` — 파일 삭제
8. `model/autoEQ/train/cognimuse_preprocess.py` — **신규** (CogniMuse 로딩)
9. `model/autoEQ/train/analyze_cognimuse_distribution.py` + **8.5 Phase 0 게이트 실제 실행** — 분포 리포트 산출, mood class <1% 시 차단. 이 결과에 따라 `num_mood_classes`, `lambda_mood`, `MOOD_CENTERS` 최종 확정. 실행 후 plan에 `distribution_report.md` 단락 추가 + V3.3 명세서 2-1/7-2 수치 기입.
10. `model/autoEQ/train/run_train.py` — CLI 확장 (**Phase 0 결과 반영한 default**)
11. 테스트 파일 일괄 수정
12. `docs/specification_v3_3.md` + `docs/spec_v3_2_to_v3_3_changelog.md` 작성 (Phase 0 리포트 인용)

**Revert 경계** (Phase 0 결과에 따른 최소 수정 범위):
- 7 mood class 유지 → 이후 단계 수정 없음
- `lambda_mood` 하향만 필요 → `config.py`의 기본값 1곳 + `run_train.py` CLI default 1곳
- 4-quadrant 축소 필요 → `config.num_mood_classes=4` + `dataset.MOOD_CENTERS` 4개로 재정의 + 테스트 `test_label_consistency.py` mood 범위 수정. `model.py`/`trainer.py`는 `num_mood_classes` 파라미터화되어 있어 구조 변경 불필요.

---

## 기존 재사용 자산

- `encoders.py`의 `XCLIPEncoder`, `PANNsEncoder`: 변경 없이 그대로 사용
- `precompute.py`의 `split_into_windows`, `encode_window_batch`, `save_features`: 그대로 재사용 (CogniMuse 로더가 입력만 바꿔 호출)
- `utils.py`의 `compute_ccc`, `compute_mean_ccc`, `compute_va_regression_metrics`, `compute_mood_metrics`: 그대로 사용
- `losses.py`의 `va_hybrid_loss`, `va_mse_loss`, `mood_ce_loss`, `gate_entropy_loss`: 그대로 사용
- `dataset.py`의 `MOOD_CENTERS`, `va_to_mood`, `compute_movie_va`, `stratified_film_level_split`(보조 유틸로 유지): 그대로 사용

---

## 명세서 V3.3 개정(필수, 코드 작업과 병행)

**원본 V3.2 보존**: `/Users/jongin/Downloads/specification_v3_2.md` 그대로 유지 (히스토리 증거)

**신규 산출물** (모두 `/Users/jongin/workspace/Homecinema/docs/` 하위):
1. `specification_v3_3.md` — 개정된 본문(아래 섹션 반영)
2. `spec_v3_2_to_v3_3_changelog.md` — 섹션별 변경 diff와 변경 근거(LIRIS 미확보)
3. `cognimuse_distribution_report.md` — Phase 0 분석 결과를 본문에 인용할 수 있도록 표/그래프 정리

**섹션별 개정 내용**:

| 섹션 | V3.2 내용 | V3.3 개정 |
|---|---|---|
| 2-1 학습 데이터 | LIRIS-ACCEDE ~9,800 클립, ~160편 | CogniMuse 7편, 4s/2s 윈도우 약 N windows (Phase 0 확정 후 수치 기입) |
| 2-2 검증 데이터 | LIRIS hold-out + CogniMuse OOD | LOMO 7-fold (test=1편 전체), time-based val holdout (train 영화 각 15% tail) |
| 2-3 Film-level Split | 75/12.5/12.5 영화 분할 | **LOMO 7-fold + 하드코딩 movie 매핑 표 첨부** (BMI/CHI/CRA/DEP/FNE/GLA/LOR → id 0~6) |
| 2-4 윈도우 구성 | 8-12s 클립 → 4s/2s 윈도우 | **40ms 연속 experienced V/A → 4s/2s 윈도우 평균(+σ 메타데이터)**. σ 분포와 학습 필터 옵션 설명 |
| 3-3 Negative Sampling | 자기지도 cross-film 오디오 교체(50/25/25) | **삭제**. 대체 문단: "CogniMuse 7편으로는 cross-film 후보 풀이 부족하여 본 연구에서는 자기지도 congruence 학습을 제외함. 대신 modality dropout의 전역 확률 적용으로 단일 모달리티 의존을 억제" |
| 5 Multi-task | V/A + Mood + Congruence(3-task) | **V/A(주) + Mood(보조) 2-task**. Congruence head 삭제 근거 명시 |
| 6 손실 가중치 | λ=(va 1.0, mood 0.5, cong 0.5, gate 0.05) | **λ=(va 1.0, mood 0.5, gate 0.05)**. Phase 0 결과 따라 mood 하향 가능성 각주 |
| 7-1 학습 하이퍼파라미터 | epochs 50, patience 10, dropout 0.1 | **epochs 30, patience 5, dropout p=0.05** + ablation {0.05,0.075,0.1} |
| 7-2 정량 평가 | LIRIS Test + CogniMuse OOD | **LOMO 7-fold test 결과 표: CCC / MAE / RMSE (V/A 회귀) + F1-macro / κ (Mood)**, fold별 breakdown + 평균±표준편차. CCC primary / MAE 게이트 / RMSE 보고 역할 명시 |
| 8-1 데이터 한계 | 서양 영화 | + **7편 소규모 학습**, + **연속→윈도우 평균 변환의 감정 전이 왜곡** (σ 분포 근거), + **주석자 7명 풀의 편향** |
| Phase 체크리스트 | LIRIS 기준 | **Phase 0 (분포 분석 게이트) 신규**, Phase 2에서 negative sampler 항목 삭제, Phase 4 평가 항목에서 OOD 문장 대체 |

**학술 방어 문구(명세서 8-1 데이터 한계 말미에 삽입)**:
> 원 계획은 LIRIS-ACCEDE를 학습, CogniMuse를 OOD 검증으로 이원화하는 것이었다. 데이터 확보 제약으로 학습·검증 모두 CogniMuse로 수행하였으며, 그 결과 (a) cross-film 기반 자기지도 Congruence 학습을 제외하고, (b) LOMO 7-fold로 평가 방식을 변경하였다. 이는 데이터 규모가 줄어든 대신 film-level 일반화를 엄격하게 측정하는 이점이 있으나, 학습 데이터의 감정 분포 편향이 7편 수준의 분산에 직접 종속된다는 한계를 가진다.

---

## 검증 방법(End-to-End)

### 0. Phase 0 분포 게이트 (학습 전 필수)
```bash
python -m model.autoEQ.train.cognimuse_preprocess \
    --cognimuse_dir /path/to/cognimuse \
    --output_dir data/features/cognimuse \
    --annotation experienced

python -m model.autoEQ.train.analyze_cognimuse_distribution \
    --feature_dir data/features/cognimuse \
    --split_name cognimuse \
    --output_dir runs/phase0
```
기대: exit code 0(모든 mood class ≥ 1%), `distribution_report.json` 생성, σ 분포 히스토그램 확인. exit code 1이면 `lambda_mood` 또는 mood 구조 조정 후 재실행.

### 1. 합성 데이터 스모크 (기존 테스트가 깨지지 않는지)
```bash
pytest model/autoEQ/train/tests -x
python -m model.autoEQ.train.run_train --use_synthetic --epochs 2
```
기대: 모든 테스트 통과, 학습 2 epoch 완주, `cong` 관련 키 부재 확인.

### 2. CogniMuse 전처리 단독
```bash
python -m model.autoEQ.train.cognimuse_preprocess \
    --cognimuse_dir /path/to/cognimuse \
    --output_dir data/features/cognimuse \
    --annotation experienced
```
기대: `cognimuse_visual.pt`, `cognimuse_audio.pt`, `cognimuse_metadata.pt` 생성.
metadata 샘플링 검증: 각 window의 V/A가 [-1,1] 내, movie_id가 0~6, window 수가 영화별 ~수백 개 수준.

### 3. 단일 fold LOMO 학습
```bash
python -m model.autoEQ.train.run_train \
    --feature_dir data/features/cognimuse \
    --split_name cognimuse \
    --lomo --lomo_fold 0 \
    --epochs 30
```
기대: 학습 loss 수렴, val mean_ccc 진행, best 모델 저장, early stop 정상 동작, 출력 dict에서 `cong` 키 부재.

### 4. 전체 7-fold LOMO
```bash
python -m model.autoEQ.train.run_lomo \
    --feature_dir data/features/cognimuse \
    --split_name cognimuse \
    --epochs 30
```
기대: 7 fold 모두 완주, fold별 `best_mean_ccc` 리스트 + 평균/표준편차 리포트 저장.

### 5. Gate / 평가 지표 합격 기준 (피드백 반영, 2026-04-18 최종)

**평가 지표 체계 — CCC + MAE + RMSE 트리오**:

| 지표 | 역할 | 이유 |
|---|---|---|
| **CCC** | Primary, early stopping 주 기준 | 분포 일치도(상관+bias+variance 종합). degenerate "mean 예측" 해 감지 |
| **MAE** | 합격 게이트 + early stopping tiebreaker | 원 단위 해석 직관, outlier robust → LOMO 7-fold 분산 안정 |
| **RMSE** | 학술 보고 표 (AVEC/MediaEval 컨벤션) | √MSE로 원 단위 복원, 제곱 오차 감도 유지. 외부 연구 비교 가능 |

세 지표 모두 **매 validation에서 로깅**, 학습 손실(MSE 기반 hybrid)은 현행 유지.

**Best model 선정 (`check_early_stopping`)**:
- CCC 개선 시 best 갱신 (주 기준)
- CCC 동률일 때 MAE가 개선된 epoch로 갱신 (Pareto 후퇴 방지 tiebreaker)
- 튜플 비교 `(mean_ccc, -mean_mae) > (best_ccc, -best_mae)`로 구현

**Gate 건전성**:
- 각 fold 학습 로그에서 `gate_w_v`, `gate_w_a`가 0.3~0.7 범위 유지 (단일 모달리티 붕괴 없음)

**지표 축 위계 (합격 판정 규칙)**:
> 합격 판정 = **V/A Primary AND V/A Safety** (V/A는 pass/fail gate). Mood 지표는 **informative metric**으로 fold별 분포·평균을 보고하되 합격 판정에 포함하지 않음. 단 Mood Safety 미달(2+ folds가 chance 수준)은 학습 불안정 신호 → 보고서에 원인 분석 필수.
> 근거: Mood head는 inference 시 미사용(`va_to_mood`가 결정론적) auxiliary task. EQ 프리셋은 V/A에만 의존.

**합격 기준 (3축 × 3단계)**:

| 단계 | V/A (pass/fail) | Mood (informative) |
|---|---|---|
| **Primary** | `mean_ccc` ≥ 0.20 **AND** `mean_mae` ≤ 0.25 | `f1_macro` ≥ **1.75/K** **AND** `kappa` ≥ 0.15 |
| **Safety** | **≥ 6/7 folds**: `min(ccc_v, ccc_a) > 0` **AND** `max(mae_v, mae_a) ≤ 0.30` | **≥ 6/7 folds**: `f1_macro` ≥ **1.4/K** |
| **Stretch** | `mean_ccc` ≥ 0.30 **AND** `mean_mae` ≤ 0.20 **AND** `mean_rmse` ≤ 0.28 | `f1_macro` ≥ **2.45/K** **AND** `kappa` ≥ 0.25 |

`K = config.num_mood_classes` (Phase 0 결과에 따라 K=7 또는 K=4). 수치 참조:

| K | f1 Primary | f1 Safety | f1 Stretch |
|---|---|---|---|
| 7 | 0.25 | 0.20 | 0.35 |
| 4 | 0.44 | 0.35 | 0.61 |

**Safety 규칙 보강**: "6/7 fold 통과" 방식 선택 이유 — median fold 방식은 fold 분포를 은폐, 6/7은 투명. **실패 허용된 fold는 보고서에 원인 분석 1문단 필수** (예: "FNE fold ccc=0.08, 원인: 유일 애니메이션 → visual feature 분포 OOD"). 기계적 무시가 아니라 **반드시 설명**.

**임계값 근거** (V/A 스팬 [-1, 1], 폭 2.0):
- MAE 0.25 = 스팬 12.5% = mood quadrant 폭 0.5의 절반 (변별력 최소선)
- MAE 0.30 = 15% (경계, 이 이상이면 EQ 프리셋 오선택 빈발)
- RMSE 0.28 = AVEC 2017~2019 continuous emotion baseline 수준
- κ 0.15 = Landis & Koch "slight→fair agreement" 경계 (chance 보정 후 최소선)
- **f1_macro chance 배수**: `1.75/K` (Primary) / `1.4/K` (Safety) / `2.45/K` (Stretch) — K 독립성 확보로 Phase 0 게이트 결과에 자동 대응. uniform predictor의 기대 f1_macro ≈ 1/K 기준.

**미달 시 대응**:
- `mean_ccc` 0.15~0.20 구간 또는 `mean_mae` 0.25~0.30: ablation 필수 — dropout p ∈ {0.05, 0.075, 0.1}, `lambda_mood` 하향, mood head를 4-quadrant로 축소 중 선택
- `mean_ccc` < 0.15: 학습 설계 근본 재검토. Visual-only / Audio-only ablation으로 문제 위치 식별
- `ccc_valence`만 낮고 `ccc_arousal`이 높은 경우: 일반적 현상. 보고서 Discussion에 Juslin & Laukka 문헌 근거로 설명

### 6. 회귀 테스트
```bash
pytest model/autoEQ/train/tests/test_lomo_split.py -v
pytest model/autoEQ/train/tests/test_cognimuse_window_aggregation.py -v
pytest model/autoEQ/train/tests/test_modality_dropout.py -v
```

---

## 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| 7편 학습만으로는 과적합 | LOMO + early stop + modality dropout으로 일반화 신호 확보. 명세서 V3.3 8-1에 한계 명시 |
| 연속 주석 → 윈도우 평균 시 감정 전이 구간 왜곡 | `valence_std`, `arousal_std` metadata 필수 저장 + Phase 0에서 σ 분포 시각화 + CLI `--sigma_filter_threshold` 옵션 제공 |
| **Val 1편 기반 early stopping 변동성** | Time-based within-movie val holdout로 val pool 확대(수백~수천 windows). 최종 리포트는 test CCC만 사용 |
| **Mood class degeneration (CogniMuse 분포 편향)** | Phase 0 게이트로 사전 차단. <1% 시 `lambda_mood` 하향 또는 4-quadrant 축소 결정 |
| Congruence head 제거로 gate 학습 신호 약화 | `lambda_gate_entropy=0.05` 유지 + modality dropout(p=0.05) 전역 적용으로 gate 균등화 |
| **Movie ID 재현 불일치** | `COGNIMUSE_MOVIES` 하드코딩 + run별 `fold_mapping.json` + manifest SHA 기록 |
| `compute_per_movie_va` 등 영화 단위 유틸이 7편에서 의미 축소 | LOMO에선 `lomo_splits_with_time_val` 사용. stratified 경로는 합성/향후 LIRIS용 보존 |
| **코드-명세서 정합성 훼손** | 명세서 V3.3 신규 작성을 코드 변경과 **동일 PR에 포함**. V3.2→V3.3 CHANGELOG 별도 문서화 |

---

## 학술 보고서용 변경 요약 문구

> "원 계획은 LIRIS-ACCEDE(~160편/9,800클립)를 학습, CogniMuse(7편/200클립)를 OOD 검증에 사용하는 것이었으나, LIRIS-ACCEDE 확보가 불가해져 학습과 검증 모두 CogniMuse로 진행하였다. 이에 따라 (1) cross-film negative sampling에 기반한 자기지도 Congruence head를 제거하고, (2) modality dropout을 congruence 라벨 의존성 없이 전체 샘플에 확률 p로 적용하는 형태로 수정하였으며, (3) film-level Leave-One-Movie-Out (7-fold) 교차검증으로 평가 방식을 변경하였다. V/A 회귀(주태스크), Mood 분류(보조), Adaptive Gating + 엔트로피 정규화는 V3.2 명세서 대비 변경 없이 유지된다."
