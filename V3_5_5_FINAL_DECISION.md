# V3.5.5 최종 채택 결정 문서

**작성일**: 2026-04-18
**브랜치**: `feat/v3-6-bandit-v2` (PR 시점 재명명 예정)
**관련 문서**: `V3_5_5_PIPELINE_DESIGN.md`, `V3_6_BANDIT_V2_NEGATIVE_RESULT.md`

---

## 1. 결정

**V3.5.5 (htdemucs 2-stem 분리 + per-stem EQ + V3.3 Compressor) 최종 채택.**

본 문서는 Mood-EQ 시스템의 공식 처리 파이프라인을 V3.5.5로 확정하고, 정식 MUSHRA 평가 인프라(4조건) 구축을 완료한 시점의 결정 기록이다.

---

## 2. 결정 경로 (버전 history)

| 버전 | 특징 | 상태 | 트랙 / 문서 |
|---|---|---|---|
| **V3.1** | ±3 dB 프리셋 (계획서 baseline) | 기준치 | 초기 계획서 |
| **V3.2** | ±4 dB dramatic (V-shape/tilt 강화) | 일부 채택 | (V3.1/V3.2 병존) |
| **V3.3** | ±6 dB 확대 + Compressor 후처리 | 정식 채택 | 트랙 2 (`feat: V3.3 EQ ±6dB...`, 0d7165c) |
| **V3.5.5** | Demucs 2-stem + per-stem EQ + Compressor | **1차 채택** | 트랙 3 (`V3_5_5_PIPELINE_DESIGN.md`, 9486d73) |
| V3.6 | Bandit v2 3-stem (영화 도메인 분리기) | **기각** | 트랙 5 (`V3_6_BANDIT_V2_NEGATIVE_RESULT.md`) |
| **V3.5.5** | (최종 확정) | **정식 채택** | 본 문서 |

---

## 3. 채택 근거

### 3.1 정량 지표

| 항목 | 수치 |
|---|---|
| Demucs 재결합 SNR (탑건 category_eq) | **+28.39 dB** |
| Demucs 재결합 SNR (라라랜드 wonder) | **+31.42 dB** |
| Residual RMS (탑건) | -47.46 dBFS |
| Residual RMS (라라랜드) | -50.73 dBFS |
| LUFS 매칭 후 peak 헤드룸 (탑건) | -4.17 dBFS |
| LUFS 매칭 후 peak 헤드룸 (라라랜드) | -4.60 dBFS |
| 처리 시간 (11s 클립, CPU) | ~13 s (real-time factor 1.2×) |

### 3.2 정성 지표 (1차 청취, HALT 5 통과)

| 평가 축 | 판정 |
|---|---|
| 대사 명료도 | ★ (우세) |
| 음악 정서 강조 | ★ (우세) |
| 전체 자연스러움 | ★ (우세) |

### 3.3 비교 검증

| 비교군 | 방법 | 결과 | 근거 |
|---|---|---|---|
| V3.3 풀믹스 EQ | 1차 청취 | **V3.5.5 우세** | 트랙 3 HALT 5 |
| V3.6 Bandit v2 | 1차 청취 + 하이퍼파라미터 튜닝 | **V3.5.5 우세** | `V3_6_BANDIT_V2_NEGATIVE_RESULT.md` |

**구조적 강점** (V3.3 대비):
- 대사(vocals)와 음악·SFX(no_vocals)를 분리 후 **차등 EQ** 가능 → 대사 명료도 + 음악 정서 강조의 trade-off 구조적 해결
- V3.3의 dialogue protection(density 기반 voice 대역 감쇠)을 유지 (B-1 정책) — 시변 EQ 처리 일관성 보존, 격리 변수는 스템 분리 단일

**구조적 강점** (V3.6 Bandit v2 대비):
- 15배 빠른 처리 (200s vs 13s / 11s 클립)
- 수치(SNR)와 청취의 일관성 (Bandit v2는 SNR +2 dB 우세에도 청취 열세)
- 학습 도메인 distribution gap 최소 (htdemucs MUSDB18은 "깨끗한" 음악이지만 재합성 위상 일관성 우수)

---

## 4. 정식 평가 프로그램 (4조건 MUSHRA)

### 4.1 구성

| 조건 | 라벨 | 파일 | 설명 |
|---|---|---|---|
| **Reference** | 원본 (매칭) | `original_matched.wav` | LUFS 매칭된 원본 (블라인드로 hidden reference 역할) |
| **Anchor** | 3.5 kHz LPF | `anchor_matched.wav` | ITU-R BS.1534-3 표준 low anchor (Butterworth 8차) |
| **V3.3** | ±6 dB + Compressor | `v3_3_matched.wav` | 풀믹스 mood EQ 비교군 |
| **V3.5.5** | Demucs + EQ + Compressor | `v3_5_5_matched.wav` | **채택본** |

### 4.2 Loudness Matching

| 씬 | Target LUFS (v3_3) | 4조건 최대 Δ | 4조건 최대 peak |
|---|---|---|---|
| 탑건 category_eq | -16.95 | 0.38 LU (anchor) | -1.00 dBFS (anchor) |
| 라라랜드 wonder | -15.61 | 0.00 LU | -3.65 dBFS (v3_3) |

**MUSHRA 표준 기준 충족**:
- 4조건 LUFS 편차 ≤ 0.5 LU ✅
- 모든 파일 peak ≤ -1.0 dBFS (보수적 헤드룸) ✅
- Anchor stopband (≥3.5 kHz) ≤ -40 dBFS (FFT 검증): **-77 dBFS (탑건) / -75 dBFS (라라랜드)** ✅

### 4.3 평가 페이지

- **파일**: `evaluation/webmushra/simple_player_v3.html`
- **접속**: `http://localhost:8765/simple_player_v3.html` (Python http.server 기준)
- **기능**:
  - 블라인드 라벨 (세션마다 A~D 무작위 셔플)
  - JND 캘리브레이션 톤 (1 dB JND 임계 확인)
  - Web Audio 기반 즉시 전환 (5ms crossfade)
  - 결과 JSON 다운로드 (session_id, blind_maps, scores, notes)

### 4.4 응답 로깅 포맷 (요약)
```json
{
  "session_id": "mood_eq_planb_<iso-timestamp>",
  "metadata": { "mode": "plan_b_html_player_v3", ... },
  "blind_maps": { "topgun_category_eq": { "reference": "B", "anchor": "D", ... } },
  "scores": { "topgun_category_eq": { "v3_5_5": {"score": 88, "blindLabel": "A"}, ... } },
  "notes": "..."
}
```

### 4.5 실행 방법
```bash
cd /d/Homecinema/evaluation/webmushra
python -m http.server 8765
# 브라우저: http://localhost:8765/simple_player_v3.html
```

---

## 5. 기술 스택 (V3.5.5 최종)

### 5.1 처리 파이프라인
```
original.wav (48kHz stereo, 11s 큐레이션 씬)
  │
  ├─► Demucs (htdemucs 4.0.1, 48k → 44.1k → 48k 리샘플)
  │     ├─► vocals.wav
  │     └─► no_vocals.wav (drums + bass + other 합산)
  │
  ├─► apply_timevarying_eq_array  (playback.py)
  │     ├─► vocals: VOCALS_CLARITY_EQ (10밴드, 씬 무관 고정, ±2.5 dB 보수)
  │     └─► no_vocals: V3.3 EQ (compute_effective_eq + dialogue protection + confidence scaling)
  │                    (시변, timeline.json scenes 기반 mood blending)
  │
  ├─► Compressor (pedalboard)
  │     threshold -12 dB, ratio 3:1, attack 10 ms, release 100 ms, makeup +3 dB
  │
  └─► clipping-safe normalize (peak > 0.99 → 0.95)

Output: v3_5_5.wav → loudness_match.py → v3_5_5_matched.wav
```

### 5.2 주요 파일

| 파일 | 역할 |
|---|---|
| `model/autoEQ/inference/eq_engine.py` | EQ 프리셋 (V3.1/V3.2/V3.3), blend + confidence + dialogue protection |
| `model/autoEQ/inference/playback.py` | `apply_timevarying_eq_array` (메모리 인터페이스) + `apply_timevarying_eq` (파일) |
| `model/autoEQ/inference/smoothing.py` | EMA + 크로스페이드 (cut 0.3s / dissolve 2.0s) |
| `tools/run_v3_5_5_pipeline.py` | V3.5.5 wrapper: Demucs stems + 시변 EQ + Compressor |
| `tools/loudness_match.py` | pyloudnorm ITU-R BS.1770-4 integrated loudness 매칭 |
| `tools/generate_anchor.py` | MUSHRA 표준 3.5 kHz LPF low anchor |
| `tools/run_v3_3_pipeline.py` | V3.3 풀믹스 기준군 (V3.5.5와 동일 compressor 사양) |
| `evaluation/webmushra/simple_player_v3.html` | 4조건 MUSHRA 평가 페이지 |
| `V3_5_5_PIPELINE_DESIGN.md` | 설계 문서 (트랙 3) |
| `V3_6_BANDIT_V2_NEGATIVE_RESULT.md` | V3.6 기각 근거 (트랙 5) |

### 5.3 의존성 (venv)
- Python 3.11, torch 2.11.0+cpu, torchaudio 2.11.0+cpu
- demucs 4.0.1, pedalboard, scipy, numpy
- pyloudnorm 0.2.0 (ITU-R BS.1770-4)
- (트랙 5 추가, V3.5.5 자체엔 미사용: pytorch_lightning 2.6.1 등)

---

## 6. 5~6번째 비선형 메커니즘 (연구 가치)

원 계획서의 5가지 비선형 메커니즘:
1. 확률 블렌딩 (probabilistic preset blending)
2. Density-aware gain modulation (대사 보호)
3. Confidence-based scaling
4. Scene-reset EMA smoothing
5. Transition-aware crossfade (cut / dissolve 차등)

**V3.5.5로 추가된 6번째 메커니즘**:
6. **Source-aware processing pipeline** — 믹스 신호를 분리기로 분해 후 스템별 차등 처리. 대사와 음악·SFX의 음향학적 특성이 다름을 전제로, 하나의 EQ가 모두를 만족시키는 "풀믹스 최적화" 한계를 돌파.

**학술적 의미**: 지금까지의 Mood-EQ는 signal-level 비선형(1~5)에 한정됐으나, V3.5.5는 source-level 비선형을 도입. 이는 "신호 → 의미적 구성요소 → 차등 처리"의 representation hierarchy를 파이프라인에 명시.

---

## 7. 한계 및 Future Work

### 7.1 알려진 한계
- **뮤지컬 singing voice**: 라라랜드 wonder 씬에서 허밍/스캣이 vocals vs no_vocals에 부분적으로 leak. htdemucs는 "노래"와 "음악 반주"의 경계가 학습 분포에 약함 (MUSDB18은 팝/록 위주).
- **처리 시간 스케일**: 11s 클립 기준 13s는 충분하나, 풀 트레일러(~130s) 기준 ~150s로 real-time interactive 부적합. 배치 전용.
- **씬 카테고리 선택 결정성**: `category_eq` 같은 모호한 씬에서 dominant mood 선택이 휴리스틱. V3.5.5 wrapper의 시변 blending이 이를 완화하나, 경계 씬에서는 여전히 약점.

### 7.2 Future Work 후보
1. **Bandit v2 fine-tuning** — 프로페셔널 영화 트레일러 데이터로 재학습하여 청취 품질 개선 후 재검토 (V3.7 가능성)
2. **htdemucs domain adaptation** — 영화 데이터로 추가 학습하여 V3.5.5 자체 강화
3. **Banquet (4-stem) 검토** — 뮤지컬 singing voice를 별도 stem으로 분리 (뮤지컬 씬 전용 파이프라인)
4. **Hybrid pipeline** — 씬 카테고리에 따라 분리기 선택 (일반 씬 htdemucs, 뮤지컬 씬 Banquet 등)
5. **개인화 JND 캘리브레이션** (트랙 4) — 청취자 개인의 JND 임계값에 맞춘 EQ 강도 조절
6. **GPU 가속** — 배치 처리 속도 개선으로 실시간 interactive 영역 진입

---

## 8. 변경 파일 전체 (PR 시점 정리용)

### 신규 (커밋 대상)
- `V3_5_5_PIPELINE_DESIGN.md` (트랙 3 커밋 `fd7044e` / 원본 `9486d73`)
- `V3_6_BANDIT_V2_NEGATIVE_RESULT.md` (트랙 5 커밋 `e2b971a`)
- `V3_5_5_FINAL_DECISION.md` (본 문서)
- `tools/run_v3_5_5_pipeline.py`
- `tools/loudness_match.py`
- `tools/generate_anchor.py`

### 수정 (커밋 대상)
- `model/autoEQ/inference/playback.py` (`apply_timevarying_eq_array` 신규)
- `model/autoEQ/inference/eq_engine.py` (`PRESET_VERSIONS["v3_5_5"]` alias)
- `evaluation/webmushra/simple_player_v3.html` (4조건 MUSHRA 재구성)
- `evaluation/webmushra/configs/mood_eq_curated.yaml` (v3_5_5 stimuli 추가)
- `.gitignore` (`tools/mss-training/`, `tmp/` 추가)

### 로컬 전용 (gitignore)
- `tmp/demucs_test/htdemucs/{topgun_cat_eq,lalaland_wonder}/{vocals,no_vocals}.wav` — Step 2/3 stems
- `tmp/bandit_test/topgun_cat_eq/**/*.wav` — 트랙 5 negative result 증거
- `tools/mss-training/` — ZFTurbo 프레임워크 clone (446MB checkpoint 포함)
- `evaluation/webmushra/configs/resources/audio/scene_*/*.wav` — 모든 오디오 자산 (재생성 가능)

---

## 9. 다음 단계 (사용자 몫)

1. **본인 1차 MUSHRA 세션** — 4조건 블라인드 평가, 결과 JSON 저장
2. **참가자 모집** — 5~10명 (이어폰 필수, 청취 경험 최소 1시간 이상 권장)
3. **결과 통계 분석**
   - 조건별 평균 / std / 95% CI
   - Wilcoxon signed-rank test (V3.3 vs V3.5.5)
   - 씬 × 조건 × 평가축 매트릭스
4. **최종 PR 준비**
   - 브랜치 재명명 (`feat/v3-6-bandit-v2` → `feat/v3-5-5-final` 등)
   - 변경 파일 정리 / squash rebase
   - `DEMUCS_UPGRADE_REVIEW.md` 업로드 (미존재, 사용자 보관본)

---

## 10. 결정자 서명

**결정**: V3.5.5 (Demucs 2-stem + per-stem EQ + V3.3 Compressor) 정식 채택.

**근거 강도**: 1차 청취 A/B + V3.6 대체안 실험 + MUSHRA 인프라 완료 = **강건(robust)**. 다중 평가 축(정량 SNR + 정성 청취 + alternative 비교) 모두 일관된 결론.

**영속성**: 본 결정은 향후 정식 MUSHRA 통계 분석 결과에 따라 재평가 가능하나, 현 단계에서 production-ready 파이프라인으로 확정.

---

*작성 환경*: Windows 10, venv Python 3.11, torch 2.11.0+cpu.
*평가 환경*: 이어폰 + 조용한 실내, Web Audio API 기반 5 ms crossfade 즉시 전환.
*문서 대조 필요 시*: `V3_5_5_PIPELINE_DESIGN.md` (설계), `V3_6_BANDIT_V2_NEGATIVE_RESULT.md` (대안 기각 근거).
