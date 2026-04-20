# model_clip_framemean — CLIP 이미지 + 프레임 평균 풀링 변형 (계획)

## 목적
`model_base/` (X-CLIP base patch32)와 **CLIP ViT-B/32 foundation과 WIT-400M
image-text contrastive pretraining을 그대로 유지한 채 X-CLIP의 learned
temporal attention / prompt encoder만 제거**한 변형으로, "video-aware
temporal modeling이 영화 V/A 예측에 실제로 기여하는가"를 분리 측정한다.

나머지 구성요소 — PANNs CNN14 오디오 백본, gated concat fusion, VA / Mood
(K=4) / Gate entropy heads, loss, dataset, trainer, LOMO 9-fold 프로토콜 —
는 `model_base/`와 동일.

## 비교 구도

이전에 고려했던 VideoMAE 안은 "pretraining data + objective + architecture"
의 3중 차이로 ablation 축 분리가 불가능하여 폐기했다. 본 변형은 PANNs vs AST
(AudioSet 내 CNN vs Transformer)와 같은 **단일 축 ablation** 을 실현한다.

| 축 | X-CLIP (baseline) | CLIP frame-mean | 단일 축? |
|---|---|---|---|
| Foundation | CLIP ViT-B/32 | CLIP ViT-B/32 | ✓ 동일 |
| Pretraining data | WIT-400M + HD-VILA | WIT-400M | ≈ (X-CLIP은 추가 adaptation) |
| Objective | image-text + video-text contrastive | image-text contrastive | ≈ |
| **Aggregation** | **learned temporal attention + prompt encoder** | **uniform frame mean** | **✓ 이 축 하나가 타겟** |
| Output dim | 512 | 512 | ✓ 동일 |

연구 질문: "X-CLIP의 video-aware temporal attention layer가 영화 감정 예측
(특히 arousal)에 의미 있는 기여를 하는가? 단순 프레임 평균 집계만으로도
충분한가?"

## 기대 효과
- σ=OFF 베이스라인(X-CLIP): mean CCC 0.541, CCC_A 0.502.
- Arousal은 shot pace / motion energy / 장면 전환 속도에 의존하므로 X-CLIP의
  temporal attention이 유리할 것으로 기대되지만, 단순 프레임 평균도 scene
  semantic은 충분히 잡을 수 있어 valence 쪽은 차이가 작을 가능성.
- 유의미한 Δ 기준: |Δ mean CCC| > 0.03, paired t-test p<0.05.

## 구현 상태
- [x] 폴더 구조 + 공유 config/model 재사용 설계 (`model_base/`의 factory 인자 활용)
- [x] `config.py` — `TrainCogConfigClipFrameMean(TrainCogConfig)`:
      clip_model_name, clip_num_frames, clip_pool 필드 (visual_dim=512 유지)
- [x] `model.py` — `AutoEQModelClipFrameMean` (AutoEQModelCog 서브클래스;
      visual_dim=512로 기존 구조와 완전 호환이라 추가 코드 불요)
- [x] `run_train.py` — `_base_run_train.main(..., model_cls=AutoEQModelClipFrameMean,
      config_cls=TrainCogConfigClipFrameMean)` 래퍼
- [x] `run_lomo.py` — `_base_run_lomo.main(..., run_train_main=run_train.main)` 래퍼

## 남은 작업

### 1. CLIP frame-mean feature 사전 계산 (`scripts/precompute_clipimg_framemean_features.py`)
- 입력: `dataset/autoEQ/CCMovies/windows/<film>/<wid>.mp4` 또는 `.pt`
  (기존 X-CLIP precompute와 같은 프레임 소스)
- 모델: `transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")`
- 추출 절차:
  1. window별로 uniform 8 frames sampling (num_frames=8)
  2. 각 frame을 224×224로 리사이즈 후 `CLIPImageProcessor`로 정규화
  3. `CLIPVisionModel` forward → `pooler_output` (B, 512)
  4. 8 frames의 pooler_output을 mean-pool → (512,)
- 출력: `data/features/ccmovies_clipimg/ccmovies_visual.pt` (dict[wid → (512,)])
  + 오디오/metadata는 베이스라인 `data/features/ccmovies/`에서 byte-for-byte 복사
  + `manifest.json` (model revision, 512-dim, 8-frame pool 기록)

### 2. LOMO 9-fold 학습 실행
```
python -m model.autoEQ.train_pseudo.model_clip_framemean.run_lomo \
    --feature_dir data/features/ccmovies_clipimg \
    --split_name ccmovies --movie_set ccmovies \
    --epochs 40 \
    --output_dir runs/ablation_clipimg_lomo9_sigma_off \
    --base_seed 42 \
    --sigma_filter_threshold -1.0 \
    --modality_dropout_p 0.05 --feature_noise_std 0.03 \
    --mixup_prob 0.5 --mixup_alpha 0.4 \
    --label_smooth_eps 0.05 --label_smooth_sigma_threshold 0.15 \
    --lambda_mood 0.3
```
σ=OFF 공정 조건 + 베이스라인과 동일한 aug 세팅으로 돌려 encoder 변수만 단일 축.

### 3. 베이스라인 대비 통계 비교 (`scripts/compare_ablation.py` 재사용)
- 입력: `runs/ccmovies_lomo9_sigma_off/lomo_report.json`,
  `runs/ablation_clipimg_lomo9_sigma_off/lomo_report.json`
- 산출: per-fold ΔCCC 표, paired t-test (n=9), 사분면 acc 비교, JSON 요약.

## 가드레일
- **Audio feature 공정성**: CLIP frame-mean 실험에서도 베이스라인의 PANNs
  `ccmovies_audio.pt`와 metadata를 **byte-for-byte 복사**해 사용. precompute
  스크립트가 이를 강제.
- **체크포인트 버전 고정**: `openai/clip-vit-base-patch32` — HF revision SHA와
  transformers 버전을 manifest.json에 기록.
- **Frame sampling**: 베이스라인 X-CLIP과 동일한 프레임 선택 인덱스 사용
  (uniform 8 frames across window). 임의 변경은 frame 내용 차이로 변수 오염.
- **Image preprocessing**: CLIPImageProcessor의 기본 설정 유지 (224×224,
  CLIP mean/std 정규화). 수동 override 금지.
