# model_ast — AST 오디오 인코더 변형 (계획)

## 목적
`model_base/` (PANNs CNN14)와 **오직 오디오 인코더만 다른** 변형으로, CNN 기반
vs Transformer 기반 오디오 백본이 영화 감정 예측 (특히 arousal CCC)에 미치는
영향을 분리 측정한다. 나머지 구성요소 — X-CLIP 비디오 백본, gated concat
fusion, VA / Mood (K=4) / Gate entropy heads, loss, dataset, trainer, LOMO
9-fold 프로토콜 — 는 `model_base/`와 동일.

## 비교 구도
> "CNN 기반 PANNs 대비 Transformer 기반 AST가 영화 감정 예측
> (특히 arousal)에 더 적합한가?"

- 공정 비교 조건: 동일 pretrain 데이터셋(**AudioSet**), 동일 film split,
  동일 seed, 동일 optimizer/lr/epoch.
- 모델 규모 차이: PANNs CNN14 ~80M vs AST ~86M (거의 동일).

## 기대 효과
- 베이스라인 약점 = arousal CCC **0.403**. AST는 self-attention으로
  window 전체를 통합하므로 long-range cue (긴장 빌드업, 지속 저역)를
  더 잘 포착할 가능성.
- 목표 Δ: mean CCC 0.47 → 0.49+ (BEATs 대비 보수적이지만 유의미 수준).

## 구현 상태
- [x] 폴더 구조 + 공유 config/model 재사용 설계 (`model_base/`에 factory 인자 추가)
- [x] `config.py` — `TrainCogConfigAST(TrainCogConfig)`: audio_raw_dim=768,
  ast_model_name, ast_max_length
- [x] `model.py` — `AutoEQModelAST` (AutoEQModelCog 서브클래스; AudioProjection
  이 config.audio_raw_dim을 소비하므로 추가 코드 불요)
- [x] `run_train.py` — `_base_run_train.main(..., model_cls=AutoEQModelAST,
  config_cls=TrainCogConfigAST)` 래퍼
- [x] `run_lomo.py` — `_base_run_lomo.main(..., run_train_main=run_train.main)` 래퍼

## 남은 작업

### 1. AST feature 사전 계산 (`scripts/precompute_ast_features.py`)
- 입력: `dataset/autoEQ/CCMovies/windows/<film>/<wid>.wav` (기존 PANNs feature
  추출에서 이미 쓰던 16 kHz mono 파일)
- 모델: `transformers.ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")`
- 추출: `last_hidden_state[:, 0]` ([CLS] token, 768-dim) 또는 mean-pool
- 출력: `data/features/ccmovies_ast/<wid>.pt` — `{"visual": ..., "audio": (768,), "labels": ...}`
- 중요: **visual feature는 기존 `data/features/ccmovies/<wid>.pt`의 visual
  텐서와 동일한 값**을 재사용해야 공정 비교 성립. precompute 스크립트에서
  기존 pt를 읽어 audio 필드만 교체하는 방식이 가장 안전.

### 2. LOMO 9-fold 학습 실행
```
python -m model.autoEQ.train_pseudo.model_ast.run_lomo \
    --feature_dir data/features/ccmovies_ast \
    --split_name ccmovies --movie_set ccmovies \
    --epochs 30 \
    --output_dir runs/ablation_ast_lomo9 \
    --base_seed 42 \
    --use_wandb --wandb_project moodeq_cog_ast
```
- 베이스라인과 동일 argv로 돌리면 오직 모델 클래스 + feature_dir만 차이.

### 3. 베이스라인 대비 통계 비교 (`scripts/compare_ablation.py`)
- 입력: `runs/ccmovies_lomo9/lomo_report.json`,
  `runs/ablation_ast_lomo9/lomo_report.json`
- 페어링 기준: 동일 test_movie_code 별 fold 매칭
- 산출:
  - per-fold Δ CCC 표
  - paired t-test (n=9)
  - 최종 요약 (mean Δ, p-value, 사분면 정확도 비교 등)

### 4. Human test_gold 재평가
- `model/autoEQ/train_pseudo/eval_testgold.py`에 AST 체크포인트 경로 넘겨서
  인간 GT 200 window 평가 재수행. 베이스라인 human CCC=0.432, 사분면 acc=0.44
  대비 비교.

## 실행 체크리스트
- [ ] `pip install transformers>=4.45` (ASTModel, ASTFeatureExtractor 가용)
- [ ] `scripts/precompute_ast_features.py` 작성 + 9편 feature 추출
  (M1/M2 MPS 권장, 총 1시간 이내 예상)
- [ ] `model_ast.run_lomo` 실행 (9 fold, 각 30 epoch, 총 수 시간)
- [ ] `scripts/compare_ablation.py` 실행 → 리포트
- [ ] Δ CCC > 0.03 이면 "유의미한 개선"으로 간주

## 가드레일
- **visual feature 공정성**: AST 실험에서도 동일한 X-CLIP visual feature를
  써야 함. precompute 스크립트가 기존 pt를 그대로 복사 후 audio 필드만
  바꾸도록 강제.
- **체크포인트 버전 고정**: transformers 모델 허브는 snapshot이 잠금되지만
  실험 재현을 위해 리포트에 transformers 버전과 model revision SHA 기록.
- **ASTFeatureExtractor 기본 설정**: `do_normalize=True`, `sampling_rate=16000`,
  `num_mel_bins=128` — 수동 override 금지 (AudioSet fine-tune과 동일 조건).

## Phase 1 검증 참고
EQ preset의 frequency-response는 `scripts/eq_response_check.py`로 이미
spec V3.3 §5-7 (±3.5 dB 예산) 이내 확인 완료 (Power만 boundary tier로
Phase 3 청취 검증 대기 중) — 오디오 인코더 교체와 독립. 본 ablation은
순수 V/A 예측 성능에만 영향.
