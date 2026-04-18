# V3.6 Bandit v2 실험 보고서 — Negative Result

**작성일**: 2026-04-18 (트랙 5 종료 시점)
**브랜치**: `feat/v3-6-bandit-v2`
**결론**: V3.6 기각, V3.5.5 (htdemucs 기반) 최종 채택.

---

## 1. 실험 목적

영화 도메인 특화 분리기(Bandit v2, DnR 학습)가 음악 도메인 분리기(htdemucs, MUSDB 학습) 대비 본 프로젝트(Mood-EQ)에 더 적합한지 검증.

**가설**: 영화 도메인으로 학습된 3-stem(speech/music/sfx) 분리기는 htdemucs의 2-stem(vocals/no_vocals)보다 도메인 정합도가 높아, V3.5.5의 vocal leak·no_vocals 혼재 문제를 구조적으로 개선할 수 있다.

---

## 2. 실험 설계

### 모델 비교

| 항목 | Bandit v2 (V3.6 후보) | htdemucs (V3.5.5 채택본) |
|---|---|---|
| 출처 | Watcharasupat et al., 2024 ([arXiv:2407.07275](https://arxiv.org/abs/2407.07275)) | Défossez, 2022 (htdemucs from `demucs==4.0.1`) |
| 학습 데이터 | DnR v3 (Divide and Remaster, 영화 음향) | MUSDB18-HQ (음악) |
| 파라미터 | 37.0M | ~83.6M |
| 입력 SR | 48,000 Hz | 44,100 Hz |
| Stems | speech / music / sfx (3) | vocals / drums / bass / other (4 → 2 합성) |
| 체크포인트 | `checkpoint-eng.ckpt` (영어 영화 특화) | demucs 패키지 내장 |

### 테스트 설정

- **씬**: `scene_topgun_category_eq/original.wav` (11.000s, 48kHz stereo, 528,000 samples)
- **프레임워크**: ZFTurbo Music-Source-Separation-Training (Bandit v2 통합 추론 인터페이스)
- **체크포인트 출처**: Zenodo `https://zenodo.org/records/12701995/files/checkpoint-eng.ckpt`
- **체크포인트 SHA256**: `3d14c48399ddb42be6fca6c316adc1b7d67d0e648e4e937359811aa6fa8b8cd2`
- **체크포인트 크기**: 446,680,129 B (446 MB)
- **추론 환경**: CPU (`--force_cpu`), 의존성 변경 시 venv 재현 필요

### 평가 지표

1. **재결합 SNR**: `20 log10(rms(original) / rms(original − Σ stems))` — 분리 정확도 정량
2. **Stem 에너지 비율**: 도메인 인식의 정성적 검증
3. **청취 판정**: 사용자 1차 평가 (대사 명료도 / 음악 정서 / 자연스러움)

---

## 3. 결과 요약

### 3.1 Step A2 (기본 설정: `num_overlap=4`, `chunk_size=384000`)

**파일 레벨**

| stem | peak dBFS | rms dBFS | 에너지 비율 |
|---|---|---|---|
| speech | -14.72 | -43.83 | 0.4% |
| music | -2.87 | -24.31 | 36.3% |
| sfx | -1.88 | -21.89 | 63.3% |

**재결합 검증 vs htdemucs**

| 지표 | Bandit v2 | htdemucs | 차이 |
|---|---|---|---|
| 재결합 SNR | **+30.42 dB** | +28.39 dB | **+2.03 dB (Bandit 우세)** |
| Residual RMS | -49.49 dBFS | -47.46 dBFS | -2.03 dB (Bandit 더 깨끗) |
| 처리 시간 (11s 클립) | ~200s | ~13s | **15배 느림** |

**청취 판정**: **htdemucs 우세** (수치 우위에도 불구하고 청취상 열세).

### 3.2 Step A2b (하이퍼파라미터 튜닝)

청취 열세의 원인을 segment 경계 artifact 또는 chunk context 부족으로 추정하여 추론 hyperparameter 튜닝 시도.

| Scenario | num_overlap | chunk_size | SNR (dB) | Δ vs A2 | 처리 시간 (s) | 비고 |
|---|---|---|---|---|---|---|
| **A2 (baseline)** | 4 | 384,000 | **+30.42** | — | 189.19 | — |
| **Scen A** | **8** | 384,000 | +29.76 | -0.66 | 488.71 | overlap 강화로 transient smoothing |
| **Scen B** | 4 | **576,000** | **+26.88** | **-3.54** | 133.46 | chunk 확대로 OOD 입력 → music peak -3.4dB |
| Scen C (조합) | — | — | — | — | — | 스펙 분기 판정에 따라 스킵 (A < A2) |

**모든 튜닝 시나리오에서 SNR 악화**. 튜닝으로 +0.5 dB 이상 개선된 시나리오 없음.

**청취 판정**: 튜닝본 어느 것도 htdemucs 대비 우세하지 않음. 튜닝 후에도 htdemucs 채택 결론 유지.

---

## 4. 학술적 관찰

### 4.1 수치와 청취의 괴리
재결합 SNR +2.03 dB 우세에도 청취 품질이 열세. Source separation 평가에서 **SDR / SNR 단독 신뢰의 위험성**을 시사. 이는 BSS Eval 류 메트릭이 perceptual quality와 약상관임을 보여주는 실험적 증거 (Le Roux et al., 2019, "SDR — half-baked or well done?"와 일관).

### 4.2 3-stem 재합성 artifact
dialogue / music / sfx 3-stem이 htdemucs 2-stem(vocals / no_vocals) 대비 재합성 위상 일관성에서 더 큰 부담. 3개 분리 신호의 위상이 모두 일관해야 원본이 복원되는데, 영화 도메인의 광범위한 spectral overlap (예: 음악 + 폭발음 + 대사가 같은 주파수 대역에서 공존)에서 작은 위상 오차가 청감상 부각될 수 있음.

### 4.3 DnR 학습 데이터의 한계
DnR 데이터셋은 영화 도메인 특화이나, 실제 프로페셔널 영화 믹싱(탑건 트레일러 = 메이저 스튜디오 5.1 채널 다운믹스)과는 distribution gap 존재 추정. DnR은 효과음 / 음악 / 대사가 비교적 명확하게 분리된 학습 샘플 위주이나, 실제 트레일러는 의도적으로 layered 됨.

### 4.4 Out-of-Distribution 민감성
chunk_size 50% 확장 (384k → 576k samples = 8s → 12s)만으로 SNR -3.54 dB 악화. 분리 모델의 추론 설정이 학습 설정과 강결합. 이는 production system에서 input length가 학습 분포에 강하게 제약됨을 시사.

### 4.5 처리 시간 트레이드오프
Bandit v2 200s/11s vs htdemucs 13s/11s = **15.4x 느림**. CPU 추론 기준으로 본 프로젝트의 batch processing 허용 범위(real-time factor 5x 이내) 초과. GPU 가속 시 차이 축소 가능하나, V3.5.5가 이미 충분히 좋은 결과를 내는 상황에서 GPU 인프라 도입을 정당화하기 어려움.

---

## 5. 결론 및 후속 방향

### 결정
**V3.5.5 (htdemucs + per-stem EQ + V3.3 Compressor) 최종 채택.**

근거:
1. 정량: Bandit v2 SNR +2 dB 우세는 청취 결과와 역상관
2. 정성: 1차 청취 + 튜닝 후 청취 모두 htdemucs 우세
3. 실용: 처리 시간 15배 차이로 production 부적합

### 부수 가치
- Bandit v2 자체는 기각이나, **영화 도메인 source separation 연구의 baseline**으로 활용 가치 있음
- V3.5.5 채택 결정의 강건성 확인 (alternative와 비교 후 선택)

### Future Work
- **Bandit v2 fine-tuning**: 실제 프로페셔널 트레일러 데이터로 fine-tuning 후 재검토
- **htdemucs domain adaptation**: 영화 데이터로 추가 학습하여 V3.5.5 자체 개선
- **Banquet (4-stem)**: 뮤지컬 singing voice를 별도 분리하는 모델 검토 (Watcharasupat 후속 연구)
- **Hybrid pipeline**: 일반 씬은 htdemucs, 라라랜드급 뮤지컬 씬은 Bandit v2 → 씬 카테고리별 분리기 선택

---

## 6. 실험 자산 (Reproducibility)

### 클론
```bash
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git tools/mss-training
```

### 체크포인트 다운로드
```bash
mkdir -p tools/mss-training/checkpoints/bandit_v2
curl -L -o tools/mss-training/checkpoints/bandit_v2/checkpoint-eng.ckpt \
  "https://zenodo.org/api/records/12701995/files/checkpoint-eng.ckpt/content"
```

### 체크포인트 prefix strip (PyTorch Lightning → ZFTurbo bare format)
```python
import torch
ckpt = torch.load('checkpoints/bandit_v2/checkpoint-eng.ckpt',
                  map_location='cpu', weights_only=False)
sd = ckpt['state_dict']
sd_clean = {(k[6:] if k.startswith('model.') else k): v
            for k, v in sd.items()
            if not k.startswith('loss_handler.')}
torch.save(sd_clean, 'checkpoints/bandit_v2/checkpoint-eng-bare.ckpt')
```

### 의존성 (venv 추가, asteroid 제외 — pesq Cython 빌드가 MSVC 의존)
```
pytorch_lightning ml_collections einops loralib spafe
segmentation_models_pytorch timm matplotlib wandb pyyaml
```

### 추론 명령
```bash
cd tools/mss-training
PYTHONIOENCODING=utf-8 PYTHONPATH=. ../../venv/Scripts/python.exe inference.py \
  --model_type bandit_v2 \
  --config_path configs/config_dnr_bandit_v2_mus64.yaml \
  --start_check_point checkpoints/bandit_v2/checkpoint-eng-bare.ckpt \
  --input_folder ../../tmp/bandit_test/input_topgun \
  --store_dir ../../tmp/bandit_test/topgun_cat_eq \
  --force_cpu --disable_detailed_pbar
```

### 튜닝 config (Step A2b)
- `tools/mss-training/configs/config_dnr_bandit_v2_mus64_overlap8.yaml` — Scenario A
- `tools/mss-training/configs/config_dnr_bandit_v2_mus64_chunk576k.yaml` — Scenario B
- `tools/mss-training/configs/config_dnr_bandit_v2_mus64_combined.yaml` — Scenario C (미실행)

### 산출물 (`.gitignore` 처리, 로컬에만 보존)
- `tmp/bandit_test/topgun_cat_eq/topgun_original/{speech,music,sfx}.wav` — Scenario A2 baseline
- `tmp/bandit_test/topgun_cat_eq/tuned_A/topgun_original/...` — Scenario A
- `tmp/bandit_test/topgun_cat_eq/tuned_B/topgun_original/...` — Scenario B

---

## 7. 측정 데이터 원본

### Step A2 baseline (탑건 category_eq)
```
파일             sr     n_smp     peak     rms
original         48000  528000   -1.87  -19.07
speech           48000  528000  -14.72  -43.83
music            48000  528000   -2.87  -24.31
sfx              48000  528000   -1.88  -21.89

Recombination (speech + music + sfx):
  ref      peak=-1.87  rms=-19.07 dBFS
  recomb   peak=-1.88  rms=-19.16
  residual rms = -49.49 dBFS
  SNR          = +30.42 dB

Stem energy ratio:
  speech:   0.4%
  music:   36.3%
  sfx:     63.3%

Processing time: 200.0s (model load 10.82s + inference 189.19s)
```

### Step A2b 비교표 (탑건 category_eq)
```
metric         A2_baseline      tuned_A      tuned_B
SNR(dB)             +30.42       +29.76       +26.88
residual            -49.49       -48.83       -45.96
speech_peak         -14.72       -15.89       -18.77
music_peak           -2.87        -2.87        -6.27
sfx_peak             -1.88        -1.88        -2.03
processing(s)       189.19       488.71       133.46
num_overlap              4            8            4
chunk_size          384000       384000       576000
```

---

## 8. Negative Result의 가치

본 보고서는 "실패"가 아닌 **negative result**로 분류. 학술 정직성과 reproducibility 관점에서:

- 후속 연구자가 동일 모델을 재시도할 때 본 결과를 baseline으로 활용 가능
- "수치는 좋은데 청취가 나쁘다"는 SOTA 분리 모델의 일반 현상에 대한 데이터 포인트 추가
- htdemucs를 영화 도메인에 그대로 사용한 결정의 정당성 확인 (alternative와 비교 후 선택)

이 실험에 투입된 ~30분 (설치 + 분리 + 튜닝 + 측정)은 V3.5.5 채택 결정의 신뢰도를 강화한 합리적 비용.
